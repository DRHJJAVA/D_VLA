import os,sys,time,json,pathlib,random,fcntl,concurrent.futures
import torch,numpy as np
from data import ROOT,RUN,CFG,atomic_json,fail,load_sample,prepare_one,path_for,digest,obs_file
from runtime import setup,seed,build,move,optimizer,checkpoint,restore_rng
from objective import objective
EPISODES={e['id']:e for e in json.loads((RUN/'episodes.json').read_text())}
def state(status,step,**kwargs):
 atomic_json('state.json',dict(status=status,completed_updates=step,time=time.time(),pid=os.getpid(),automatic_closed_loop=False,**kwargs))
def read_record(rec,previous=True):
 e=EPISODES[rec['episode']];t=int(rec['frame'])
 prepare_one((e,t));q=load_sample(rec);previous_q=None
 if previous and t>0:
  try:prior_observation=obs_file(e,t-1)
  except FileNotFoundError:prior_observation=None
  if prior_observation is not None and (ROOT/'experiments/lcjp003/r2/v2/v001/cache'/e['id']/(str(t-1)+'.pt')).exists():
   pre=dict(rec,frame=t-1);prepare_one((e,t-1));previous_q=load_sample(pre)
 return q,previous_q
def prior_plan(model,previous,noise,step):
 if previous is None:return None
 with torch.no_grad():
  en=model.encode(previous['inputs'],step=step)
  m,g,_=model.generate(en,noise)
  plan=torch.cat([m,g.argmax(-1).float()[...,None]*2-1],-1)
  return torch.cat([plan[:,1:],plan[:,-1:]],1).detach()
def evaluate(model,step):
 started=time.perf_counter();was=model.training;model.eval();state('evaluating',step)
 records=json.loads((RUN/'evaluation_windows.json').read_text());out=[]
 scale=torch.tensor(json.loads((RUN/'normalization.json').read_text())['std'][:6],device='cuda')
 with torch.no_grad():
  for i,rec in enumerate(records):
   q,pre=read_record(rec,previous=False);q=move(q);pre=move(pre) if pre else None
   gen=torch.Generator(device='cpu').manual_seed(CFG['seed']+100000+i)
   noise=torch.randn(8,16,6,generator=gen).cuda()
   plan=prior_plan(model,pre,noise,step)
   d,y=q['inputs'],q['labels'];e=model.encode(d,step=step)
   motion,grip,sw=model.generate(e,noise,plan)
   error=(motion-y['target'][None]).abs();delta=error*scale[None,None]
   valid=e['points']['valid'];scores=e['point_score'][valid]
   row={**rec,'normalized_motion_mae':float(error.mean()),'translation_action_mae':float(delta[...,:3].mean()),
    'rotation_action_mae':float(delta[...,3:].mean()),'gripper_error':float((grip.argmax(-1)!=y['grip'][None]).float().mean()),
    'lambda_mean':float(e['lambda'].mean()),'predicted_target_points':e['points']['point_count'],
    'score_mean':float(scores.mean()) if len(scores) else None,'score_std':float(scores.std(unbiased=False)) if len(scores) else None,
    'score_min':float(scores.min()) if len(scores) else None,'score_max':float(scores.max()) if len(scores) else None}
   # Teacher agreement is explicitly not independent recognition accuracy.
   pred=e['seg']['logits'].sigmoid()>.5;truth=y['mask']>.5;known=y['mask_weight']>0
   inter=(pred&truth&known).sum();union=((pred|truth)&known).sum()
   row['weak_teacher_mask_iou']=float(inter/union.clamp_min(1))
   row['grip_switch_error']=float(((sw.sigmoid()>.5)!=y['switch'][None].bool()).float().mean())
   for mode in ['uniform','shuffle','wrong']:
    modified=model.encode(d,score_override=mode,step=step)
    mo,_,_=model.generate(modified,noise,plan)
    row[mode+'_mae']=float((mo-y['target'][None]).abs().mean())
    row[mode+'_action_change']=float((mo-motion).abs().mean())
   out.append(row)
 summary={}
 for split in ['train','dev']:
  rr=[x for x in out if x['split']==split];keys=[k for k,v in rr[0].items() if isinstance(v,(int,float)) and k!='frame']
  summary[split]={k:float(np.mean([x[k] for x in rr if x.get(k) is not None])) for k in keys if any(x.get(k) is not None for x in rr)}
 report={'step':step,'time':time.time(),'seconds':time.perf_counter()-started,'same_80_windows':True,'same_noise':True,
 'predicted_mask_only':True,'independent_window_previous_plan_reset':True,'weak_teacher_iou_is_not_ground_truth_accuracy':True,'summary':summary,'records':out}
 atomic_json('eval%04d.json'%step,report)
 model.train(was);return report
def main():
 lock=open(RUN/'train.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (RUN/'STOP').exists(),'paused'
 gates=json.loads((RUN/'checks/network_and_gradient.json').read_text());assert gates['passed']
 source=json.loads((RUN/'source_lock.json').read_text())
 for name,value in source['files'].items():assert digest(ROOT/name)==value,'source_changed:'+name
 setup();seed(CFG['seed']);model=build();opt=optimizer(model)
 schedule=json.loads((RUN/'sampling_schedule.json').read_text())
 start=0
 if (RUN/'checkpoints/last.pt').exists():
  ck=torch.load(RUN/'checkpoints/last.pt',map_location='cpu',weights_only=False)
  assert ck['source_lock_sha256']==digest(RUN/'source_lock.json'),'resume_source_mismatch'
  model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng']);start=ck['step']
 else:checkpoint(model,opt,0)
 if not (RUN/'eval0000.json').exists():evaluate(model,0)
 model.train();times=[]
 gw=torch.tensor(CFG.get('grip_class_weights',[1.,1.]),device='cuda')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures={}
  def submit(idx):
   if idx<len(schedule):futures[idx]=[pool.submit(read_record,r) for r in schedule[idx]]
  submit(start);submit(start+1)
  for index in range(start,len(schedule)):
   step=index+1
   if (RUN/'STOP').exists():checkpoint(model,opt,index);state('paused_by_request',index);return
   state('training',index,next_update=step)
   t=time.perf_counter();batch=[f.result() for f in futures.pop(index)];submit(index+2)
   data_seconds=time.perf_counter()-t
   opt.zero_grad(set_to_none=True);components=[];forward=0.;backward=0.;prior=0.;h2d=0.
   for raw,pre in batch:
    t=time.perf_counter();q=move(raw);previous=move(pre) if pre else None;torch.cuda.synchronize();h2d+=time.perf_counter()-t
    noise=torch.randn(8,16,6,device='cuda')
    t=time.perf_counter()
    plan=prior_plan(model,previous,noise,step) if random.random()>=CFG['previous_plan_dropout'] else None
    torch.cuda.synchronize();prior+=time.perf_counter()-t
    t=time.perf_counter();loss,metrics,e,_=objective(model,q['inputs'],q['labels'],noise,step,plan,gw)
    assert torch.isfinite(loss),'nonfinite_loss'
    torch.cuda.synchronize();forward+=time.perf_counter()-t
    t=time.perf_counter();(loss/8).backward();torch.cuda.synchronize();backward+=time.perf_counter()-t
    components.append(metrics)
   t=time.perf_counter()
   grads={}
   for tag in ['segment','score','motion','grip']:
    gs=[p.grad for n,p in model.named_parameters() if n.startswith(tag) and p.grad is not None]
    grads[tag]=sum(float(g.abs().sum()) for g in gs)
   assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'nonfinite_gradient'
   torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['optimizer']['grad_clip']);opt.step()
   torch.cuda.synchronize();optim=time.perf_counter()-t
   elapsed=data_seconds+h2d+prior+forward+backward+optim;times.append(elapsed)
   row={'step':step,'time':time.time(),'seconds':elapsed,'data_wait':data_seconds,'data_h2d':h2d,'previous_plan':prior,
    'forward':forward,'backward':backward,'optimizer_check':optim,'available_previous_windows':sum(pre is not None for _,pre in batch),'gradients_l1':grads,
    'losses':{k:float(np.mean([x[k] for x in components])) for k in components[0]}}
   with (RUN/'runtime_timings.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if step%10==0 or step in [20,50]:checkpoint(model,opt,step)
   if step in [20,50]:
    seconds=float(np.mean(times[-min(20,len(times)):]))
    eval_sec=json.loads((RUN/'eval0000.json').read_text())['seconds']
    atomic_json('eta%04d.json'%step,{'step':step,'time':time.time(),'mean_update_seconds':seconds,
     'to_step250_seconds':max(0,250-step)*seconds+eval_sec,'to_step3000_seconds':(3000-step)*seconds+((3000-step)//250+1)*eval_sec,
     'cold_preparation_excluded':True,'shared_gpu_load_may_change':True,'closed_loop_included':False})
   if step%250==0:evaluate(model,step)
   state('training',step)
 checkpoint(model,opt,len(schedule));state('training_complete_pending_acceptance',len(schedule))
if __name__=='__main__':
 try:main()
 except Exception as ex:
  previous=json.loads((RUN/'state.json').read_text()).get('completed_updates',0) if (RUN/'state.json').exists() else 0
  fail('train',ex);state('failed',previous);raise SystemExit(1)
