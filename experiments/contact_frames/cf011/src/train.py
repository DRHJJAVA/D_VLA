from common import *
import numpy as np,torch,random,copy
from torch.nn import functional as F
from model import SurfacePredictor,at_points,gaussian
torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
DEVICE='cuda'
def seedall(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def parameter_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def load_data():
 data={}
 for ep in CFG['train_episodes']:
  with np.load(BASE/'data'/f'{ep}.npz') as z:
   data[ep]={k:torch.from_numpy(z[k].copy()).to(DEVICE) for k in z.files if k not in ['center','pixels']}
 return data
def make_batch(data,items,mu,sd):
 hist=[];st=[];gr=[];ta=[];mask=[];xyz=[];col=[];future_rgb=[];future_d=[];future_K=[];future_C=[]
 for ep,t in items:
  d=data[ep];hist.append(d['frames'][t-3:t+1].reshape(28,96,96));st.append((d['states'][t]-mu)/sd)
  gr.append(d['grid'][t]);ta.append(d['target'][t]/.01);mask.append(d['mask'][t]);xyz.append(d['xyz'][t]);col.append(d['color'][t])
  ix=torch.tensor([t+k for k in CFG['horizons']],device=DEVICE)
  future_rgb.append(d['rgb'][ix]);future_d.append(d['depth'][ix]);future_K.append(d['K'][ix]);future_C.append(d['C'][ix])
 return dict(images=torch.stack(hist),state=torch.stack(st),grid=torch.stack(gr),target=torch.stack(ta),mask=torch.stack(mask),xyz=torch.stack(xyz),color=torch.stack(col),rgb=torch.stack(future_rgb),depth=torch.stack(future_d),K=torch.stack(future_K),C=torch.stack(future_C))
def photo_loss(mean,b):
 B,H,N,_=mean.shape
 pred=b['xyz'][:,None]+mean*.01
 C=b['C'];K=b['K']
 cam=torch.einsum('bhij,bhnj->bhni',C[:,:,:3,:3].transpose(-1,-2),pred-C[:,:,:3,3,None].transpose(-1,-2))
 pix=torch.einsum('bhij,bhnj->bhni',K,cam)
 xy=pix[...,:2]/pix[...,2:].clamp_min(1e-5)
 grid=2*(xy+.5)/96-1
 colors=F.grid_sample(b['rgb'].flatten(0,1),grid.reshape(B*H,1,N,2),align_corners=False).squeeze(2).transpose(1,2).reshape(B,H,N,3)
 depth=F.grid_sample(b['depth'].flatten(0,1),grid.reshape(B*H,1,N,2),align_corners=False,mode='nearest').reshape(B,H,N)
 valid=b['mask']&(cam[...,2]>.01)&(grid.abs().amax(-1)<=1)&(depth>0)&((cam[...,2]-depth).abs()<=.01)
 # Visibility masks apply only to image reconstruction, never the 3D likelihood.
 err=(colors-b['color'][:,None]).abs().mean(-1)
 return (err*valid).sum()/valid.sum().clamp_min(1)
def preflight(data):
 ep=CFG['train_episodes'][0];d=data[ep];t=int(d['eligible'][0])
 mu=torch.zeros(50,device=DEVICE);sd=torch.ones(50,device=DEVICE)
 b=make_batch(data,[(ep,t)],mu,sd);m=SurfacePredictor().to(DEVICE)
 m.eval()
 with torch.no_grad():
  a=m(b['images'],b['state'],False);b2=m(b['images'],b['state']+torch.randn_like(b['state']),False)
  assert torch.equal(a,b2)
  mean,s=at_points(a,b['grid']);assert bool(((s>=.2)&(s<=2)).all())
  old=d['frames'][t+1:].clone();d['frames'][t+1:]=torch.randn_like(old)
  images2=make_batch(data,[(ep,t)],mu,sd)['images'];d['frames'][t+1:]=old
  assert torch.equal(images2,b['images'])
 m.train();pred=m(b['images'],b['state'],True);mean,s=at_points(pred,b['grid'])
 nll=gaussian(mean,s,b['target']);nll.retain_grad()
 loss=(nll*b['mask']).sum()/b['mask'].sum()+CFG['photometric_weight']*photo_loss(mean,b);loss.backward()
 assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
 assert bool((nll.grad[~b['mask']]==0).all())
 return dict(parameters=sum(p.numel() for p in m.parameters()),future_frame_input_invariance=True,unconditional_state_invariance=True,unknown_nll_gradient_zero=True,finite_backward=True,sigma_m_range=CFG['sigma_m'])
def evaluate(model,data,ep,mu,sd,arm,out):
 model.eval();d=data[ep];all_t=d['eligible'].cpu().numpy().tolist()
 means=[];sigmas=[];targets=[];masks=[];nlls=[];points=[];zero=[];velocity=[]
 with torch.no_grad():
  for start in range(0,len(all_t),8):
   ts=all_t[start:start+8];b=make_batch(data,[(ep,t) for t in ts],mu,sd)
   mean,s=at_points(model(b['images'],b['state'],arm=='robot_conditioned'),b['grid'])
   means.append((mean*.01).cpu().numpy());sigmas.append((s*.01).cpu().numpy());targets.append((b['target']*.01).cpu().numpy());masks.append(b['mask'].cpu().numpy())
   nlls.append(gaussian(mean,s,b['target']).cpu().numpy())
   for t in ts:
    vel=d['xyz'][t]-d['xyz'][t-1];valid=d['point_valid'][t]&d['point_valid'][t-1]
    v=torch.stack([vel*k for k in CFG['horizons']]);v[~valid[None].expand(3,-1)]=float('nan')
    velocity.append(v.cpu().numpy())
 mean=np.concatenate(means);sigma=np.concatenate(sigmas);target=np.concatenate(targets);mask=np.concatenate(masks);nll=np.concatenate(nlls);vel=np.stack(velocity)
 out.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(out/'predictions.npz',frames=np.array(all_t),mean=mean,sigma=sigma,target=target,mask=mask,nll=nll,history_velocity=vel)
 rows=[]
 for j,k in enumerate(CFG['horizons']):
  v=mask[:,j];e=np.linalg.norm(mean[:,j]-target[:,j],axis=-1);z=np.linalg.norm(target[:,j],axis=-1);ve=np.linalg.norm(vel[:,j]-target[:,j],axis=-1)
  def stats(sel):
   if not sel.any():return dict(count=0)
   return dict(count=int(sel.sum()),error_mean_mm=float(e[sel].mean()*1000),error_median_mm=float(np.median(e[sel])*1000),nll=float(nll[:,j][sel].mean()),sigma_mean_mm=float(sigma[:,j][sel].mean()*1000),axis95_coverage=float((np.abs(mean[:,j]-target[:,j])<=1.96*sigma[:,j,:,None])[sel].mean()),zero_motion_mean_mm=float(z[sel].mean()*1000))
  rows.append(dict(horizon=k,all=stats(v),moving_gt2mm=stats(v&(z>.002)),low_motion_le2mm=stats(v&(z<=.002)),historical_velocity_mean_mm=float(ve[v&np.isfinite(ve)].mean()*1000) if (v&np.isfinite(ve)).any() else None))
 save(out/'metrics.json',dict(episode=ep,arm=arm,rows=rows,unit='pseudo-correspondence observations, not independent contacts',history_baseline='uses offline track identities, diagnostic only'))
 model.train()
 return rows
def main():
 stage('training_preflight')
 torch.cuda.set_per_process_memory_fraction(CFG['gpu_fraction'])
 verify(json.loads((BASE/'input_locked.json').read_text()));verify(json.loads((BASE/'data_locked.json').read_text()))
 data=load_data();seedall(2026)
 save(BASE/'audits/preflight.json',dict(time=now(),**preflight(data)))
 all_results=[]
 for held in CFG['train_episodes']:
  training=[ep for ep in CFG['train_episodes'] if ep!=held]
  assert held not in training
  s=torch.cat([data[ep]['states'][data[ep]['eligible']] for ep in training])
  mu=s.mean(0);sd=s.std(0).clamp_min(.01)
  for seed in CFG['seeds']:
   seedall(seed);init=copy.deepcopy(SurfacePredictor().state_dict());ih=parameter_hash(init)
   rng=np.random.default_rng(seed)
   eligible={ep:data[ep]['eligible'].cpu().numpy() for ep in training}
   # Every consecutive three updates visits all three training episodes, with four frames from one event.
   order=[]
   for cycle in range(CFG['steps']//3):order.extend(rng.permutation(training).tolist())
   assert len(order)==CFG['steps']
   plan=[[(ep,int(t)) for t in rng.choice(eligible[ep],size=CFG['batch_size'],replace=True)] for ep in order]
   fold=BASE/'runs'/held/f'seed_{seed}';fold.mkdir(parents=True,exist_ok=True)
   save(fold/'plan.json',dict(train=training,held_out=held,seed=seed,steps=plan,initialization_sha256=ih,norm_mean=mu.cpu().tolist(),norm_std=sd.cpu().tolist()))
   torch.save(init,fold/'initialization.pt')
   for arm in CFG['arms']:
    out=fold/arm;out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():raise RuntimeError('refuse_to_overwrite_completed_run')
    seedall(seed);model=SurfacePredictor().to(DEVICE);model.load_state_dict(init)
    assert parameter_hash(model.state_dict())==ih
    opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
    history=[]
    for step,items in enumerate(plan,1):
     b=make_batch(data,items,mu,sd);opt.zero_grad(set_to_none=True)
     mean,sigma=at_points(model(b['images'],b['state'],arm=='robot_conditioned'),b['grid'])
     nll=gaussian(mean,sigma,b['target']);loss3=(nll*b['mask']).sum()/b['mask'].sum().clamp_min(1)
     phot=photo_loss(mean,b);loss=loss3+CFG['photometric_weight']*phot
     if not torch.isfinite(loss):raise FloatingPointError('nonfinite_loss')
     loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
     history.append(dict(step=step,episode=items[0][0],nll=float(loss3.detach()),photometric=float(phot.detach()),total=float(loss.detach())))
     if step%50==0:
      save(out/'history.json',history)
      stage('training',held_out=held,seed=seed,arm=arm,step=step,completed_runs=len(all_results))
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),steps=CFG['steps'],init_hash=ih,config=CFG),out/'last.pt')
    save(out/'history.json',history)
    rows=evaluate(model,data,held,mu,sd,arm,out)
    all_results.append(dict(held_out=held,seed=seed,arm=arm,rows=rows))
    save(BASE/'results_progress.json',dict(time=now(),runs=all_results))
    del model,opt
 verify(json.loads((BASE/'input_locked.json').read_text()));verify(json.loads((BASE/'data_locked.json').read_text()))
 save(BASE/'summary.json',dict(time=now(),runs=all_results,n_runs=len(all_results),manual_training_labels=False,contact_classifier_trained=False))
 stage('training_complete_pending_audit',completed_runs=len(all_results))
if __name__=='__main__':
 try:main()
 except Exception as e:failure(e);raise SystemExit(1)
