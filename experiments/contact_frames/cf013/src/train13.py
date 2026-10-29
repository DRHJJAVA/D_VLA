from common13 import *
from model13 import *
import random,time,copy
def batch(data,motion,items,mu,sd):
 b=old.make_batch(data,items,mu,sd)
 b['motion96']=torch.stack([motion[e]['motion96'][t] for e,t in items])
 b['point_motion']=torch.stack([motion[e]['point_motion'][t] for e,t in items])
 return b
def predict(model,b,arm):
 return model(b['images'],b['state'],b['grid'],b['motion96'],b['point_motion'],arm=='motion_residual')
def metrics(pred):
 rows=[]
 for hi,h in enumerate([1,2,4]):
  mask=pred['mask'][:,hi];err=np.linalg.norm(pred['mean'][:,hi]-pred['target'][:,hi],axis=-1)
  sd=pred['sigma'][:,hi];absdiff=np.abs(pred['mean'][:,hi]-pred['target'][:,hi])
  for subset in ['all_targets','motion_valid']:
   k=mask if subset=='all_targets' else mask&pred['motion_valid']
   n=int(k.sum())
   rows.append(dict(horizon=h,subset=subset,count=n,error_mm=float(err[k].mean()*1000) if n else None,nll=float(pred['nll'][:,hi][k].mean()) if n else None,coverage95=float((absdiff[k]<=1.96*sd[k,None]).mean()) if n else None,width95_mm=float((3.92*sd[k]).mean()*1000) if n else None))
 return rows
def evaluate(model,data,motion,ep,mu,sd,arm,out):
 model.eval();fr=data[ep]['eligible'].cpu().numpy().astype(int);parts={k:[] for k in ['mean','sigma','target','mask','nll','velocity','motion_valid']}
 with torch.no_grad():
  for start in range(0,len(fr),4):
   times=fr[start:start+4];b=batch(data,motion,[(ep,int(t)) for t in times],mu,sd);mean,sigma=predict(model,b,arm)
   vals={'mean':mean*.01,'sigma':sigma*.01,'target':b['target']*.01,'mask':b['mask'],'nll':gaussian(mean,sigma,b['target']),'velocity':b['point_motion'][...,:3],'motion_valid':b['point_motion'][...,3].bool()}
   for k,v in vals.items():parts[k].append(v.detach().cpu().numpy())
 pred={k:np.concatenate(v) for k,v in parts.items()};pred['frames']=fr
 out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'predictions.npz',**pred)
 rows=metrics(pred);save(out/'metrics.json',rows)
 return rows
def preflight(data,motion):
 ep=CFG['train_episodes'][0];pp=PREV/'runs'/ep/'seed_2026';plan=json.loads((pp/'plan.json').read_text())
 mu=torch.tensor(plan['norm_mean'],device='cuda');sd=torch.tensor(plan['norm_std'],device='cuda')
 initial=torch.load(pp/'initialization.pt',map_location='cpu',weights_only=False)
 old.seedall(2026);m=MotionPredictor(initial).cuda();ref=SurfacePredictor().cuda();ref.load_state_dict(initial)
 items=[(ep,int(data[ep]['eligible'][0]))];b=batch(data,motion,items,mu,sd)
 with torch.no_grad():
  a,s=predict(m,b,'motion_absolute');x,y=at_points(ref(b['images'],b['state'],True),b['grid'])
  initerr=max(float((a-x).abs().max()),float((s-y).abs().max()))
  assert initerr<1e-6,('initial_function',initerr)
  r,rs=predict(m,b,'motion_residual')
  expected=b['point_motion'][None] if False else b['point_motion'][:,None,:,:3]/.01*torch.tensor([1,2,4],device='cuda')[None,:,None,None]*b['point_motion'][:,None,:,3:4]
  assert torch.allclose(r-a,expected,atol=2e-6)
  b2=dict(b);b2['target']=torch.randn_like(b['target'])*100;b2['rgb']=torch.randn_like(b['rgb'])
  aa,ss=predict(m,b2,'motion_absolute');assert torch.equal(a,aa) and torch.equal(s,ss)
 a=a.detach().requires_grad_();s=s.detach().requires_grad_();nll=gaussian(a,s,b['target']);loss=nll[b['mask']].mean();loss.backward()
 assert torch.all(a.grad[~b['mask']]==0) and torch.all(s.grad[~b['mask']]==0)
 assert torch.isfinite(a.grad).all()
 save(BASE/'audits/preflight.json',dict(time=now(),initial_function_max_diff=initerr,residual_identity=True,future_loss_data_input_invariance=True,unknown_nll_gradient_zero=True,parameters=sum(p.numel() for p in m.parameters())))
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.3)
 verify(json.loads((BASE/'motion_locked.json').read_text()));verify(json.loads((BASE/'training_sources_locked.json').read_text()))
 data=old.load_data();motion={}
 for ep in CFG['train_episodes']:
  z=np.load(BASE/'data'/ep/'motion_features.npz');motion[ep]={k:torch.from_numpy(z[k]).float().cuda() for k in ['motion96','point_motion']}
 preflight(data,motion)
 results=[]
 for held in CFG['train_episodes']:
  for seed in CFG['seeds']:
   prev=PREV/'runs'/held/f'seed_{seed}';out=BASE/'runs'/held/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
   plan=json.loads((prev/'plan.json').read_text())
   assert held not in plan['train'] and len(plan['steps'])==600
   assert all(e in plan['train'] for step in plan['steps'] for e,t in step)
   initial=torch.load(prev/'initialization.pt',map_location='cpu',weights_only=False)
   old.seedall(seed);m=MotionPredictor(initial);newinitial=copy.deepcopy(m.state_dict());ih=old.parameter_hash(newinitial)
   torch.save(newinitial,out/'initialization.pt');save(out/'plan.json',{**plan,'cf011_initialization_sha256':plan['initialization_sha256'],'initialization_sha256':ih,'original_plan_sha256':sha(prev/'plan.json')})
   mu=torch.tensor(plan['norm_mean'],device='cuda');sd=torch.tensor(plan['norm_std'],device='cuda')
   for arm in CFG['arms']:
    target=out/arm
    assert not (target/'last.pt').exists(),('already_trained',held,seed,arm)
    target.mkdir();old.seedall(seed);model=MotionPredictor(initial).cuda();model.load_state_dict(newinitial)
    assert old.parameter_hash(model.state_dict())==ih
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[]
    for step,items in enumerate(plan['steps'],1):
     model.train();b=batch(data,motion,items,mu,sd);mean,sigma=predict(model,b,arm)
     nll=gaussian(mean,sigma,b['target']);ln=nll[b['mask']].mean();lp=old.photo_loss(mean,b);loss=ln+.1*lp
     assert torch.isfinite(loss),('nonfinite_loss',held,seed,arm,step)
     opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
     history.append(dict(step=step,nll=float(ln.detach()),photo=float(lp.detach()),loss=float(loss.detach()),grad=float(grad)))
     if step%100==0:
      save(target/'history.json',history);stage('training',episode=held,seed=seed,arm=arm,step=step,completed_runs=len(results))
      print(json.dumps(dict(episode=held,seed=seed,arm=arm,step=step,loss=history[-1]['loss'])),flush=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),steps=600,init_hash=ih,config=CFG),target/'last.pt')
    save(target/'history.json',history);rows=evaluate(model,data,motion,held,mu,sd,arm,target)
    results.append(dict(episode=held,seed=seed,arm=arm,metrics=rows));save(BASE/'results.json',results)
    del model,opt;torch.cuda.empty_cache()
 verify(json.loads((BASE/'input_locked.json').read_text()));verify(json.loads((BASE/'motion_locked.json').read_text()))
 stage('training_complete',runs=len(results),updates=len(results)*600)
if __name__=='__main__':main()
