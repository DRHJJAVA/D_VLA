
from prepare import *
from datetime import datetime,timezone,timedelta
def stamp():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'executor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((OUT/'action_input_gate.json').read_text())['all_inputs_exact']
 ck=RUN/'checkpoints/step3000.pt'
 assert sha(ck)=='9a84bf5b21c6437f567c1208e3f876e61a32d73e53c5475a155b712ece7b8446'
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=JointNetwork(common.build_expert('cpu').lm_expert)
 state=torch.load(ck,map_location='cpu',weights_only=False);assert int(state['step'])==3000
 model.load_state_dict(state['model'],strict=True);model=model.cuda().eval();del state
 adapter=legacy_adapter()
 r3vis=sys.modules.get('visibility')
 sys.modules['visibility']=adapter.module_from(ROOT/'experiments/lcjp003/src/visibility.py','visibility')
 sys.modules['data']=adapter.module_from(ROOT/'experiments/lcjp003/src/data.py','data')
 sys.modules['visibility']=r3vis
 torch.backends.cudnn.allow_tf32=True
 proposals=adapter.LiveProposals('cuda')
 rows=json.loads((ROOT/'experiments/contact_joint_object/manifests/episodes.json').read_text())['episodes']
 taskfiles=sorted((VENDOR/'init_files/libero_object').glob('*.pruned_init'))
 norm=json.loads((RUN/'normalization.json').read_text())
 sd=torch.tensor(norm['std'],device='cuda');mu=torch.tensor(norm['mean'],device='cuda')
 results=[];shared_sam=None;times=[]
 for init_index in range(5):
  for task_index,taskfile in enumerate(taskfiles):
   trial=init_index*10+task_index;task=taskfile.stem
   trialdir=OUT/'trials'/('trial_%03d'%trial)
   if (trialdir/'result.json').exists():
    results.append(json.loads((trialdir/'result.json').read_text()));continue
   if trialdir.exists():raise RuntimeError('partial_trial_exists_preserve_and_review_before_resume')
   row=next(dict(e) for e in rows if e['id']==task+'__demo_0');row['frames']=605
   trialdir.mkdir(parents=True)
   render=adapter.LiveRender(row['id'],trialdir,capacity=605)
   states=torch.load(taskfile,map_location='cpu',weights_only=False)
   render.env.set_init_state(states[init_index])
   for _ in range(5):render.env.step(np.zeros(7,np.float32))
   render.capture(0)
   words=torch.load(OLD2/'cache'/(row['task']+'_words.pt'),map_location='cpu',weights_only=False)
   builder,_=input_builder(trialdir,words);builder.actions={row['id']:np.zeros((605,7),np.float32)}
   if shared_sam is not None:builder.sam=shared_sam
   live=live_geometry(trialdir);cache=trialdir/'r2/cache'/row['id'];cache.mkdir(parents=True,exist_ok=True)
   gen=torch.Generator(device='cpu').manual_seed(20261028+trial)
   plan=None;success=False;done=False;records=[];started=time.time()
   for t in range(600):
    if (OUT/'STOP').exists():
     save('state.json',{'status':'paused_at_control_boundary','trial':trial,'step':t,'time':stamp()});render.close();return
    ts=time.perf_counter();torch.backends.cudnn.allow_tf32=True
    proposals.frame(render,t)
    original_h5=h5py.File
    def inference_read_guard(path,*args,**kwargs):
     if '/data/raw/' in str(path):raise RuntimeError('expert_demonstration_read_forbidden_during_policy_inference')
     return original_h5(path,*args,**kwargs)
    h5py.File=inference_read_guard
    try:base=builder._cache_frame(row,t)
    finally:h5py.File=original_h5
    torch.save(base,cache/(str(t)+'.pt'));shared_sam=builder.sam
    d,_,objects,meta=live.policy_input(row,t)
    obs_seconds=time.perf_counter()-ts
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    d={k:v.cuda() for k,v in d.items()}
    with torch.inference_mode():
     enc=model.encode(d);z=torch.randn(1,16,6,generator=gen).cuda()
     a,g=model.generate(enc,z,plan=plan)
     physical=a[0]*sd+mu
     grip=g[0].argmax(-1)
     action=torch.cat([physical[0],(grip[:1].float()*2-1)]).cpu().numpy()
     assert np.isfinite(action).all()
     # Controller applies its registered input clipping; record both proposed and executed.
     executed=np.clip(action,-1,1)
     shifted=torch.cat([a[0,1:],a[0,-1:]],0)
     shifted_grip=torch.cat([grip[1:],grip[-1:]],0).float()
     plan=torch.cat([shifted,shifted_grip[:,None]],-1).detach()
     chosen=int(enc['roles'].argmax())
     np.savez_compressed(trialdir/('policy_%04d.npz'%t),alpha=enc['alpha'].cpu().numpy(),roles=enc['roles'].cpu().numpy(),xyz=d['xyz'].cpu().numpy(),valid=d['valid'].cpu().numpy(),action=action,executed=executed)
    policy_seconds=time.perf_counter()-ts-obs_seconds
    builder.actions[row['id']][t]=executed
    result=render.step(executed,t+1)
    success=bool(render.env.check_success())
    if isinstance(result,tuple):done=bool(result[-1]) if isinstance(result[-1],(bool,np.bool_)) else False
    elapsed=time.perf_counter()-ts;times.append(elapsed)
    rec={'step':t+1,'success':success,'selected_candidate':chosen,'objects':len(objects),'observation_seconds':obs_seconds,'policy_seconds':policy_seconds,'total_seconds':elapsed,'action':action.tolist(),'executed_action':executed.tolist()}
    records.append(rec)
    if trial==0 and t==3:save('control_gate.json',{'passed':True,'real_predict_step_observe_cycles':4,'future_demonstration_reads':0,'new_observations_used':True,'time':stamp()})
    with (trialdir/'steps.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    save('state.json',{'status':'running','trial':trial,'task':task,'initial_state_index':init_index,'control_steps':t+1,'completed_trials':len(results),'successes':sum(x['success'] for x in results),'time':stamp()})
    if len(times)==20:
     save('eta20.json',{'mean_step_seconds':float(np.mean(times)),'median_step_seconds':float(np.median(times)),'first_trial_worst_remaining_seconds':float((600-(t+1))*np.mean(times)),'all_50_worst_control_seconds':float((50*600-20)*np.mean(times)),'note':'upper estimate assuming all trials reach600 steps; success terminates early, shared load can change'})
    if success or done:break
   render.close()
   res={'trial':trial,'task':task,'initial_state_index':init_index,'success':success,'steps':len(records),'termination':'success' if success else ('environment_done' if done else '600_step_timeout'),'elapsed_seconds':time.time()-started,'time':stamp()}
   (trialdir/'result.json').write_text(json.dumps(res,ensure_ascii=False,indent=2));results.append(res)
   per_task={p.stem:{'completed':sum(x['task']==p.stem for x in results),'successes':sum(x['task']==p.stem and x['success'] for x in results)} for p in taskfiles}
   save('results.json',{'completed':len(results),'planned':50,'successes':sum(x['success'] for x in results),'success_rate':sum(x['success'] for x in results)/len(results),'per_task':per_task,'trials':results,'full_500_benchmark':False})
 save('complete.json',{'completed':50,'successes':sum(x['success'] for x in results),'time':stamp()})
 save('state.json',{'status':'complete','completed_trials':50,'successes':sum(x['success'] for x in results),'time':stamp()})
if __name__=='__main__':
 try:main()
 except Exception as e:save('failure.json',{'type':type(e).__name__,'summary':str(e)[:500],'time':stamp(),'frames':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
