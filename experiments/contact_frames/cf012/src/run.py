from common12 import *
import numpy as np,torch,random,copy,itertools
def stats(pred):
 m=pred['mask'];sig=pred['sigma'];err=pred['mean']-pred['target']
 rows=[]
 for j,h in enumerate(CFG['horizons']):
  v=m[:,j];s=sig[:,j];e=err[:,j]
  rows.append(dict(horizon=h,count=int(v.sum()),error_mean_mm=float(np.linalg.norm(e,axis=-1)[v].mean()*1000),sigma_mean_mm=float(s[v].mean()*1000),q=float(((e**2).sum(-1)/(3*s**2))[v].mean()),floor_fraction=float((s[v]<.0021).mean()),ceiling_fraction=float((s[v]>.0199).mean())))
 return rows
def load_pred(p):
 with np.load(p) as z:return {k:z[k].copy() for k in ['frames','mean','sigma','target','mask','nll']}
def fit_diagnostics(data):
 result=[]
 for held in CFG['episodes']:
  for seed in CFG['seeds']:
   fold=PREV/'runs'/held/f'seed_{seed}';plan=json.loads((fold/'plan.json').read_text())
   mu=torch.tensor(plan['norm_mean'],device='cuda');sd=torch.tensor(plan['norm_std'],device='cuda')
   for arm in CFG['arms']:
    model=old.SurfacePredictor().cuda()
    ck=torch.load(fold/arm/'last.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
    for ep in plan['train']:
     out=BASE/'fit_diagnostics'/held/f'seed_{seed}'/arm/ep
     old.evaluate(model,data,ep,mu,sd,arm,out)
     result.append(dict(outer_episode=held,seed=seed,arm=arm,training_episode=ep,rows=stats(load_pred(out/'predictions.npz'))))
    stage('fit_diagnostics',outer_episode=held,seed=seed,arm=arm,completed_models=len(result)//3)
    del model,ck
 save(BASE/'fit_diagnostics/summary.json',dict(time=now(),rows=result,scope='training-episode predictions only; no manual ranges'))
def train_pairs(data):
 results=[]
 for pi,pair in enumerate(itertools.combinations(CFG['episodes'],2)):
  training=list(pair);remaining=[e for e in CFG['episodes'] if e not in pair]
  s=torch.cat([data[e]['states'][data[e]['eligible']] for e in training]);mu=s.mean(0);sd=s.std(0).clamp_min(.01)
  for seed in CFG['seeds']:
   old.seedall(seed);init=copy.deepcopy(old.SurfacePredictor().state_dict());ih=old.parameter_hash(init)
   rng=np.random.default_rng(seed);order=[]
   for _ in range(CFG['steps']//2):order.extend(rng.permutation(training).tolist())
   plan=[[(ep,int(t)) for t in rng.choice(data[ep]['eligible'].cpu().numpy(),size=CFG['batch_size'],replace=True)] for ep in order]
   fold=BASE/'inner'/f'pair_{pi}'/f'seed_{seed}';fold.mkdir(parents=True,exist_ok=True)
   save(fold/'plan.json',dict(training=training,evaluation=remaining,seed=seed,steps=plan,initialization_hash=ih,norm_mean=mu.cpu().tolist(),norm_std=sd.cpu().tolist()))
   torch.save(init,fold/'initialization.pt')
   for arm in CFG['arms']:
    out=fold/arm;out.mkdir(exist_ok=True)
    if (out/'last.pt').exists():raise RuntimeError('refuse_overwrite_inner_training')
    old.seedall(seed);model=old.SurfacePredictor().cuda();model.load_state_dict(init)
    assert old.parameter_hash(model.state_dict())==ih
    opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
    hist=[]
    for step,items in enumerate(plan,1):
     b=old.make_batch(data,items,mu,sd);opt.zero_grad(set_to_none=True)
     mean,sigma=old.at_points(model(b['images'],b['state'],arm=='robot_conditioned'),b['grid'])
     nll=old.gaussian(mean,sigma,b['target'])
     loss3=(nll*b['mask']).sum()/b['mask'].sum().clamp_min(1)
     photo=old.photo_loss(mean,b);loss=loss3+CFG['photometric_weight']*photo
     if not torch.isfinite(loss):raise FloatingPointError('nonfinite_loss')
     loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
     hist.append(dict(step=step,episode=items[0][0],nll=float(loss3.detach()),photo=float(photo.detach()),total=float(loss.detach())))
     if step%100==0:
      save(out/'history.json',hist)
      stage('training_inner_models',pair=pi,training=training,seed=seed,arm=arm,step=step,completed_runs=len(results))
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),steps=600,init_hash=ih,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),config=CFG),out/'last.pt')
    save(out/'history.json',hist)
    for ep in remaining:
     old.evaluate(model,data,ep,mu,sd,arm,out/ep)
    results.append(dict(pair=pi,training=training,evaluation=remaining,seed=seed,arm=arm))
    save(BASE/'inner_progress.json',dict(time=now(),runs=results))
    del model,opt
 return results
def calibrate():
 stage('calibrating_from_inner_predictions')
 records=[]
 for held in CFG['episodes']:
  for arm in CFG['arms']:
   sources=[];q=[]
   for pi,pair in enumerate(itertools.combinations(CFG['episodes'],2)):
    if held in pair:continue
    calib=next(e for e in CFG['episodes'] if e not in pair and e!=held)
    assert held not in pair and held!=calib
    for seed in CFG['seeds']:
     folder=BASE/'inner'/f'pair_{pi}'/f'seed_{seed}'
     pred=folder/arm/calib/'predictions.npz'
     a=load_pred(pred);rows=stats(a);q.append([x['q'] for x in rows])
     sources.append(dict(training=list(pair),calibration_episode=calib,seed=seed,prediction=str(pred),prediction_sha256=sha(pred),checkpoint=str(folder/arm/'last.pt'),checkpoint_sha256=sha(folder/arm/'last.pt')))
   assert len(q)==9
   alpha=np.sqrt(np.maximum(np.mean(q,axis=0),CFG['minimum_variance_numeric']))
   records.append(dict(held_out=held,arm=arm,alpha=alpha.tolist(),sources=sources))
 save(BASE/'calibration_frozen.json',dict(time=now(),records=records,formula='sqrt(mean_event_seed(||residual||^2/(3*sigma^2)))'))
 save(BASE/'calibration_locked.json',lock([BASE/'calibration_frozen.json']))
def apply_calibration():
 stage('evaluating_frozen_calibration')
 frozen=json.loads((BASE/'calibration_frozen.json').read_text());result=[]
 for row in frozen['records']:
  ep=row['held_out'];arm=row['arm'];alpha=np.array(row['alpha'],np.float64)
  for seed in CFG['seeds']:
   a=load_pred(PREV/'runs'/ep/f'seed_{seed}'/arm/'predictions.npz')
   sig=a['sigma']*alpha[None,:,None]
   error=a['mean']-a['target']
   nll=.5*(error/sig[...,None])**2
   nll=nll.sum(-1)+3*np.log(sig/.01)+1.5*np.log(2*np.pi)
   out=BASE/'calibrated'/ep/f'seed_{seed}'/arm;out.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(out/'predictions.npz',frames=a['frames'],mean=a['mean'],sigma=sig,raw_sigma=a['sigma'],target=a['target'],mask=a['mask'],nll=nll)
   metrics=[]
   for j,h in enumerate(CFG['horizons']):
    v=a['mask'][:,j];e=error[:,j]
    r=dict(horizon=h,count=int(v.sum()),alpha=float(alpha[j]),error_mm=float(np.linalg.norm(e,axis=-1)[v].mean()*1000))
    for label,s,ll in [('raw',a['sigma'][:,j],a['nll'][:,j]),('calibrated',sig[:,j],nll[:,j])]:
     d=dict(nll=float(ll[v].mean()),sigma_mm=float(s[v].mean()*1000))
     for name,z in [('50',.6744897501960817),('80',1.2815515655446004),('95',1.959963984540054)]:
      d['coverage'+name]=float((np.abs(e)<=z*s[...,None])[v].mean())
      d['width'+name+'_mm']=float((2*z*s[v]).mean()*1000)
     r[label]=d
    metrics.append(r)
   save(out/'metrics.json',dict(held_out=ep,seed=seed,arm=arm,rows=metrics))
   result.append(dict(held_out=ep,seed=seed,arm=arm,rows=metrics))
 save(BASE/'calibrated_summary.json',dict(time=now(),runs=result))
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(CFG['gpu_fraction'])
 verify(json.loads((PREV/'completed_locked.json').read_text()))
 verify(json.loads((BASE/'source_locked.json').read_text()))
 data=old.load_data()
 stage('fit_diagnostics')
 fit_diagnostics(data)
 result=train_pairs(data)
 assert len(result)==36
 calibrate();apply_calibration()
 verify(json.loads((BASE/'source_locked.json').read_text()));verify(json.loads((PREV/'completed_locked.json').read_text()))
 save(BASE/'training_complete.json',dict(time=now(),runs=36,updates=21600))
 stage('calibration_complete_pending_baseline_audit',runs=36)
if __name__=='__main__':
 try:main()
 except Exception as e:failure(e);raise SystemExit(1)
