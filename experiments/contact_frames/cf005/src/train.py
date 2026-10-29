from pathlib import Path
import json,datetime,hashlib,importlib.util,traceback,sys,os
import numpy as np,torch
from torch.nn import functional as F
R=Path('/mnt/nvme1/drh/D_VLA'); B=R/'experiments/contact_frames/cf005'; C=R/'experiments/contact_frames/cf003'
torch.set_num_threads(4)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mod(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 cfg=json.loads((B/'config.json').read_text()); old=mod(C/'src/train_and_evaluate.py','old_functions'); net=mod(B/'src/model_all_train_frames.py','new_model')
 norm=json.loads((C/'normalization.json').read_text());mean=np.array(norm['mean'],np.float32);std=np.array(norm['std'],np.float32)
 sources=[B/'config.json',B/'src/model_all_train_frames.py',B/'src/train.py']
 inputs=[C/'normalization.json',C/'src/train_and_evaluate.py']
 data={}; tensors={}; contexts={}
 for ep in cfg['train_episodes']:
  path=B/'data'/ep/'artifacts/dataset.npz';inputs.append(path);d=dict(np.load(path));data[ep]=d;contexts[ep]=old.context(d)
  tensors[ep]={k:torch.as_tensor(d[k],device='cuda',dtype=torch.float32) for k in ['images','points','target','weight','dy','dynvalid']}
  tensors[ep]['states']=torch.as_tensor((d['states']-mean)/std,device='cuda')
 for seed in cfg['seeds']:
  inputs.extend([C/f'runs/no_world/seed_{seed}/initial.pt',B/f'plans/seed_{seed}.json',C/f'runs/no_world/seed_{seed}/last.pt'])
 lock={str(p):sha(p) for p in sources+inputs};save(B/'training_inputs_locked.json',lock)
 save(B/'normalization.json',norm)
 pre=[]
 for seed in cfg['seeds']:
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
  initial=torch.load(C/f'runs/no_world/seed_{seed}/initial.pt',map_location='cpu',weights_only=False)
  model=net.ContactFrameNet(50).cuda();model.load_state_dict(initial['model'])
  control=mod(C/'src/model_no_world.py','control_model').ContactFrameNet(50).cuda();control.load_state_dict(initial['model'])
  ep=cfg['train_episodes'][0];dt=tensors[ep];ix=torch.as_tensor(contexts[ep][[5,20,60]],device='cuda')
  model.eval();control.eval()
  with torch.no_grad():
   aa=model(dt['images'][ix],dt['points'][ix],dt['states'][ix]);bb=control(dt['images'][ix],dt['points'][ix],dt['states'][ix])
   diff=max(float((a-b).abs().max()) for a,b in zip(aa,bb))
  assert diff==0
  z=torch.zeros(3,device='cuda',requires_grad=True);w=torch.tensor([1,0,1.],device='cuda')
  (F.binary_cross_entropy_with_logits(z,torch.ones_like(z),reduction='none')*w).mean().backward();assert z.grad[1]==0
  pre.append({'seed':seed,'eval_matches_baseline_initial_max_abs':diff,'initial_tensor_hash':initial['initial_tensor_hash'],'unknown_bce_gradient_zero':True})
  del control
  out=B/f'runs/all_train_frames/seed_{seed}';out.mkdir(parents=True,exist_ok=True)
  torch.save(initial,out/'initial.pt')
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
  plan=json.loads((B/f'plans/seed_{seed}.json').read_text());assert len(plan)==600
  history=[];model.train()
  for step,item in enumerate(plan,1):
   ep=item['episode'];frames=item['frames'];d=tensors[ep];ix=torch.as_tensor(contexts[ep][frames],device='cuda')
   opt.zero_grad(set_to_none=True);logit,dy=model(d['images'][ix],d['points'][ix],d['states'][ix])
   bce=(F.binary_cross_entropy_with_logits(logit.reshape(-1),d['target'][frames],reduction='none')*d['weight'][frames]).mean()
   aux=(F.smooth_l1_loss(dy,d['dy'][frames],reduction='none').mean(-1)*d['dynvalid'][frames]).mean()
   loss=bce+.2*aux;assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
   history.append({'step':step,'episode':ep,'bce':float(bce),'aux':float(aux),'loss':float(loss)})
   if step%100==0:
    save(out/'history.json',history);save(B/'stage_state.json',{'stage':'training','seed':seed,'step':step,'updated':now()})
  ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'steps':600,'state_mean':mean,'state_std':std,'config':cfg,'initial_tensor_hash':initial['initial_tensor_hash'],'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
  torch.save(ck,out/'last.pt');save(out/'history.json',history)
 save(B/'audits/preflight.json',pre)
 weights={str(p):sha(p) for p in sorted((B/'runs').glob('*/seed_*/last.pt'))};save(B/'evaluation_weights_locked.json',weights)
 save(B/'stage_state.json',{'stage':'evaluating','updated':now()})
 summaries=[]
 def predict(model,d,ix):
  st=(d['states']-mean)/std;prob=[]
  with torch.no_grad():
   for j in range(0,len(ix),16):
    k=ix[j:j+16]
    z,_=model(torch.as_tensor(d['images'][k],device='cuda'),torch.as_tensor(d['points'][k],device='cuda'),torch.as_tensor(st[k],device='cuda',dtype=torch.float32))
    prob.extend(z.sigmoid().cpu().numpy().reshape(-1).tolist())
  return np.asarray(prob,np.float32)
 for ep in cfg['train_episodes']+cfg['fixed_review_episodes']:
  is_train=ep in data
  src=B/'data'/ep/'artifacts/dataset.npz' if is_train else C/'evaluation_no_world'/ep/'artifacts/dataset.npz'
  d=dict(np.load(src));n=len(d['states'])
  ix=old.context(d) if is_train else np.load(C/'evaluation_no_world'/ep/'artifacts/frame_predictions.npz')['context_indices']
  allprob=[]
  for seed in cfg['seeds']:
   ck=torch.load(B/f'runs/all_train_frames/seed_{seed}/last.pt',map_location='cpu',weights_only=False)
   m=net.ContactFrameNet(50).cuda().eval();m.load_state_dict(ck['model']);pr=predict(m,d,ix);allprob.append(pr)
   if is_train:np.savez_compressed(B/f'runs/all_train_frames/seed_{seed}/{ep}_prediction.npz',probability=pr)
  probs=np.array(allprob);lo=probs.min(0);hi=probs.max(0);avg=probs.mean(0)
  reliable=d['geometry_reliable']&d['state_valid']
  pos=(lo>=.8)&reliable;neg=(hi<=.2)&reliable&(d['distance']-2*d['sigma']>.006);unknown=~(pos|neg)
  dest=B/'evaluation_all_train_frames'/ep/'artifacts';dest.mkdir(parents=True,exist_ok=True)
  (dest/'dataset.npz').symlink_to(src.resolve())
  np.savez_compressed(dest/'frame_predictions.npz',probabilities=probs,mean=avg,low=lo,high=hi,contact_candidate=pos,observed_separation_candidate=neg,unknown=unknown,context_indices=ix)
  w=d['weight'];target=d['target'];a=np.clip(avg,1e-7,1-1e-7);weakneg=(w>0)&(target<=.2)
  row={'episode':ep,'training_episode':is_train,'frames':n,'candidate':int(pos.sum()),'separation':int(neg.sum()),'unknown':int(unknown.sum()),'weighted_weak_bce':float((w*(-target*np.log(a)-(1-target)*np.log(1-a))).sum()/max(w.sum(),1e-9)),'weak_negative_frames':int(weakneg.sum()),'unanimous_high_on_weak_negative':int((weakneg&(lo>=.8)).sum()),'mean_seed_spread':float((hi-lo).mean())}
  rr={'white_yellow_mug_demo_5':(44,52,'negative'),'porcelain_mug_demo_9':(56,95,'positive'),'tomato_sauce_demo_7':(52,132,'positive')}
  if ep in rr:
   aa,bb,label=rr[ep];sl=slice(aa,bb+1)
   row['human_review_only']={'range':[aa,bb],'label':label,'seed_high_counts':(probs[:,sl]>=.8).sum(1).tolist(),'candidate':int(pos[sl].sum()),'separation':int(neg[sl].sum()),'unknown':int(unknown[sl].sum()),'unanimous_high':int((lo[sl]>=.8).sum()),'scores_mean':avg[sl].tolist()}
  summaries.append(row)
 save(B/'summary.json',summaries)
 changes=[p for p,h in lock.items() if sha(Path(p))!=h];assert not changes
 assert weights=={str(p):sha(p) for p in sorted((B/'runs').glob('*/seed_*/last.pt'))}
 save(B/'audits/completion.json',{'time':now(),'steps_per_seed':600,'seeds':cfg['seeds'],'training_inputs_unchanged':True,'weights_frozen_before_review_evaluation':True,'review_ranges_used_in_training':False,'initial_and_plan_control':'cf003/no_world','unknown_gradient_zero':True})
 save(B/'stage_state.json',{'stage':'training_evaluation_complete','updated':now()})
if __name__=='__main__':
 try:main()
 except Exception as e:
  (B/'failure.log').write_text(traceback.format_exc())
  save(B/'failure.json',{'type':type(e).__name__,'message':str(e),'time':now()})
  save(B/'stage_state.json',{'stage':'failed','exception_type':type(e).__name__,'updated':now()});sys.exit(1)
