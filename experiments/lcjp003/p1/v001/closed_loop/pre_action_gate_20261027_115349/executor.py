import os,sys,time,json,pathlib,datetime,traceback,fcntl
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA');R=ROOT/'experiments/lcjp003';P=R/'p1/v001';C=P/'closed_loop'
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def write(name,obj):
 p=C/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def fail(e):
 write('failure.json',{'time':now(),'type':type(e).__name__,'message':str(e)[:600],'frames':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
def can_start(st,checkpoint_exists):
 return st.get('stage')=='training_complete_pending_closed_loop' and st.get('completed_updates')==3000 and checkpoint_exists
def run():
 from adapters import LiveRender,LiveProposals,LiveInput,FrozenSmolFeatures,LCJP003,ModelConfig,Dataset,sha,torch,np,h5py,CAMS
 from video import overlay,encode_video,page
 import cv2
 build=json.loads((C/'build_lock.json').read_text())
 for path,digest in build['files'].items():assert sha(path)==digest,'Closed-loop dependency changed: '+path
 assert json.loads((C/'causal_window_validation.json').read_text())['passed']
 assert (C/'video_validation_complete.json').exists()
 source=json.loads((P/'source_lock.json').read_text())
 for name,digest in source.items():assert sha(P/name)==digest,'Locked source changed: '+name
 saved=torch.load(P/'step3000.pt',map_location='cpu',weights_only=False)
 assert saved['step']==3000
 assert saved['config_sha256']==sha(P/'config.json')
 assert saved['source_lock_sha256']==sha(P/'source_lock.json')
 assert all(int(v['step'])==3000 for v in saved['optimizer']['state'].values())
 for x in saved['model'].values():
  if torch.is_floating_point(x):assert torch.isfinite(x).all()
 write('frozen_checkpoint.json',{'time':now(),'step':3000,'sha256':sha(P/'step3000.pt'),'inputs_sha256':sha(P/'inputs.json'),'evaluation_lock_sha256':sha(P/'evaluation_lock.json')})
 device='cuda';model=LCJP003(ModelConfig()).to(device).eval();model.load_state_dict(saved['model']);del saved
 encoder=FrozenSmolFeatures(R/'weights/smolvlm2',device=device)
 lock=json.loads((P/'evaluation_lock.json').read_text())['closed_loop']
 write('state.json',{'stage':'gpu_input_gate','time':now()})
 work=C/'gpu_gate';eid=lock['episodes'][0]
 render=LiveRender(eid,work,capacity=4);proposals=LiveProposals(device=device);proposals.frame(render,0)
 adapter=LiveInput(eid,work,capacity=4,device=device,encoder=encoder)
 fr=adapter.frame(0);batch=adapter.batch(0);ref=Dataset(cached=True);rf=ref.frame(eid,0)
 differences={k:bool(np.array_equal(fr[k],rf[k],equal_nan=True)) for k in ['xyz_base','xyz_norm','relation','object_id','point_valid','rgb','state','projected_uv','view_visible']}
 sem_diff=float(np.max(np.abs(adapter.frames[0]['semantic']-rf['semantic'])))
 write('gpu_input_gate.json',{'time':now(),'exact_fields':differences,'semantic_max_abs':sem_diff})
 render.close()
 if not all(differences.values()) or sem_diff!=0:raise RuntimeError('Live GPU input differs from locked cached frame; formal trials not started')
 gen=torch.Generator(device='cpu').manual_seed(lock['noise_seeds'][0])
 with torch.inference_mode():pred=model(batch,torch.randn(1,1,16,7,generator=gen).to(device))
 assert tuple(pred['actions'].shape)==(1,1,16,7)
 del proposals,adapter,batch,pred;torch.cuda.empty_cache()
 write('ready.json',{'time':now(),'gpu_input_exact':True,'checkpoint_step':3000})
 assets=ROOT/'docs/LCJP003_P1_closed_loop_assets';assets.mkdir(exist_ok=True)
 results=[]
 for trial,(eid,seed) in enumerate(zip(lock['episodes'],lock['noise_seeds'])):
  folder=C/f'trial_{trial:02d}';folder.mkdir(exist_ok=True);completed=folder/'result.json'
  if completed.exists():results.append(json.loads(completed.read_text()));continue
  render=LiveRender(eid,folder,capacity=501);proposals=LiveProposals(device=device)
  adapter=LiveInput(eid,folder,capacity=501,device=device,encoder=encoder)
  gen=torch.Generator(device='cpu').manual_seed(seed)
  rawvideo=folder/'dual_camera.mp4';writer=cv2.VideoWriter(str(rawvideo),cv2.VideoWriter_fourcc(*'mp4v'),20,(1024,576))
  assert writer.isOpened()
  count=0;records=[];success=False;reason='max_steps';start=time.monotonic()
  try:
   for t in range(500):
    tick=time.monotonic();proposals.frame(render,t);after_prop=time.monotonic()
    fr=adapter.frame(t);batch=adapter.batch(t);after_input=time.monotonic()
    with torch.inference_mode():out=model(batch,torch.randn(1,1,16,7,generator=gen).to(device))
    torch.cuda.synchronize();after_model=time.monotonic()
    writer.write(overlay(fr,out,t,trial));count+=1
    np.savez_compressed(folder/f'scores_{t:04d}.npz',scores=out['scores'][0].cpu().numpy(),xyz=fr['xyz_base'],object_id=fr['object_id'],point_valid=fr['point_valid'],projected_uv=fr['projected_uv'],view_visible=fr['view_visible'])
    action=out['actions'][0,0,0].cpu().numpy()*adapter.ds.action_std+adapter.ds.action_mean
    success,done=render.step(action,t+1)
    row={'frame':t,'action':action.tolist(),'success':success,'env_done':done,'proposal_s':after_prop-tick,'input_s':after_input-after_prop,'model_s':after_model-after_input,'seconds':time.monotonic()-tick}
    records.append(row)
    with (folder/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    write('state.json',{'stage':'closed_loop','trial':trial,'step':t+1,'time':now(),'mean_step_s':(time.monotonic()-start)/(t+1)})
    if t==19:
     write('eta20.json',{'time':now(),'trial':trial,'measured_steps':20,'mean_step_s':(time.monotonic()-start)/20,'max_remaining_trial_s':480*(time.monotonic()-start)/20,'early_success_can_shorten':True})
    if success:reason='success';break
    if done:reason='env_done';break
   # Append final observed post-action RGB, without reusing prior scores.
   rgb=np.concatenate([cv2.resize(render.out[cam+'/rgb'][len(records)],(512,512),interpolation=cv2.INTER_NEAREST) for cam in CAMS],axis=1)
   banner=np.zeros((64,1024,3),np.uint8);cv2.putText(banner,'Final post-action observation | '+reason,(14,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(230,230,230),1)
   writer.write(cv2.cvtColor(np.concatenate([banner,rgb],axis=0),cv2.COLOR_RGB2BGR));count+=1
  finally:writer.release();render.close()
  dst=assets/f'trial_{trial:02d}.mp4';decoded=encode_video(rawvideo,dst,count)
  item={'trial':trial,'episode':eid,'seed':seed,'steps':len(records),'success':success,'stop_reason':reason,'decoded_frames':decoded,'seconds':time.monotonic()-start,'video_relative':'LCJP003_P1_closed_loop_assets/'+dst.name,'checkpoint_step':3000,'time':now()}
  completed.write_text(json.dumps(item,indent=2));results.append(item);page(results)
  write('results.json',{'time':now(),'completed_trials':len(results),'successes':sum(x['success'] for x in results),'trials':results})
  if trial==0:write('first_video.json',item)
  del proposals,adapter,batch,out;torch.cuda.empty_cache()
 write('complete.json',{'time':now(),'trials':20,'successes':sum(x['success'] for x in results),'results':results})
def main():
 handle=(C/'executor.lock').open('a')
 try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('Closed-loop executor already active')
 while True:
  if (P/'failure.json').exists():raise RuntimeError('Training failure present; closed-loop not started')
  st=json.loads((P/'state.json').read_text())
  if can_start(st,(P/'step3000.pt').exists()):break
  write('state.json',{'stage':'waiting_for_training','time':now(),'training_completed_updates':st.get('completed_updates',st.get('step'))})
  time.sleep(30)
 run()
if __name__=='__main__':
 try:main()
 except Exception as e:fail(e)
