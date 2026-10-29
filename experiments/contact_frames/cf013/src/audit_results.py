from common13 import *
from model13 import MotionPredictor
from train13 import metrics
import collections,math
def main():
 assert json.loads((BASE/'stage_state.json').read_text())['stage'] in ['training_complete','training_complete_pending_audit']
 runs=json.loads((BASE/'results.json').read_text());assert len(runs)==24
 rows=[];hashchecks={};checks=[];max_nll=0.;max_target=0.
 for lp in ['input_locked.json','motion_locked.json','training_sources_locked.json']:
  hashchecks[lp]=verify(json.loads((BASE/lp).read_text()))
 for tag in ['cf011','cf012']:
  lp=ROOT/'experiments/contact_frames'/tag/'completed_locked.json'
  hashchecks[tag]=verify(json.loads(lp.read_text()))
 for ep in CFG['train_episodes']:
  d=np.load(PREV/'data'/f'{ep}.npz');motion=np.load(BASE/'data'/ep/'motion_features.npz')
  for seed in CFG['seeds']:
   root=BASE/'runs'/ep/f'seed_{seed}';pr=PREV/'runs'/ep/f'seed_{seed}'
   p=json.loads((root/'plan.json').read_text());o=json.loads((pr/'plan.json').read_text())
   for k in ['train','held_out','seed','steps','norm_mean','norm_std']:assert p[k]==o[k],('plan',k)
   assert p['original_plan_sha256']==sha(pr/'plan.json')
   init=torch.load(root/'initialization.pt',map_location='cpu',weights_only=False)
   assert old.parameter_hash(init)==p['initialization_sha256']
   oldinit=torch.load(pr/'initialization.pt',map_location='cpu',weights_only=False)
   for k,v in oldinit.items():
    vv=init['base.'+k]
    if k=='encoder.0.weight':
     assert torch.equal(vv[:,:28],v) and torch.count_nonzero(vv[:,28:])==0
    else:assert torch.equal(vv,v)
   assert torch.count_nonzero(init['local.2.weight'])==0 and torch.count_nonzero(init['local.2.bias'])==0
   for arm in CFG['arms']:
    rd=root/arm;hist=json.loads((rd/'history.json').read_text())
    assert len(hist)==600 and [x['step'] for x in hist]==list(range(1,601))
    ck=torch.load(rd/'last.pt',map_location='cpu',weights_only=False)
    assert ck['steps']==600 and ck['init_hash']==p['initialization_sha256']
    assert all(int(v['step'])==600 for v in ck['optimizer']['state'].values())
    z=np.load(rd/'predictions.npz');fr=z['frames'];mask=z['mask'];target=z['target']
    assert np.array_equal(fr,d['eligible'])
    assert np.array_equal(mask,d['mask'][fr])
    dif=float(np.max(np.abs(target-d['target'][fr])));max_target=max(max_target,dif);assert dif<1e-6
    assert np.array_equal(z['motion_valid'],motion['point_motion'][fr,:,3].astype(bool))
    assert np.allclose(z['velocity'],motion['point_motion'][fr,:,:3],atol=0,rtol=0)
    mu=z['mean'].astype(np.float64)/.01;sd=z['sigma'].astype(np.float64)/.01;tar=target.astype(np.float64)/.01
    nll=.5*np.sum((mu-tar)**2,axis=-1)/sd**2+3*np.log(sd)+1.5*np.log(2*np.pi)
    nd=float(np.max(np.abs(nll-z['nll'])));max_nll=max(max_nll,nd);assert np.allclose(nll,z['nll'],atol=2e-4,rtol=2e-5)
    for row in metrics(z):rows.append(dict(episode=ep,seed=seed,arm=arm,**row))
    # Identical saved target samples for all baselines; invalid causal velocities are not filled as observations.
    ref=np.load(pr/'robot_conditioned/predictions.npz')
    assert np.array_equal(ref['frames'],fr) and np.array_equal(ref['mask'],mask)
    for hi,h in enumerate([1,2,4]):
     for subset in ['all_targets','motion_valid']:
      k=mask[:,hi] if subset=='all_targets' else mask[:,hi]&z['motion_valid']
      if arm==CFG['arms'][0]:
       for label,pred in [('cf011_robot',ref['mean'][:,hi]),('static',np.zeros_like(target[:,hi]))]:
        err=np.linalg.norm(pred-target[:,hi],axis=-1)
        entry=dict(episode=ep,seed=seed,arm=label,horizon=h,subset=subset,count=int(k.sum()),error_mm=float(err[k].mean()*1000),nll=None,coverage95=None,width95_mm=None)
        if label=='cf011_robot':
         entry.update(nll=float(ref['nll'][:,hi][k].mean()),coverage95=float((np.abs(pred-target[:,hi])[k]<=1.96*ref['sigma'][:,hi][k,None]).mean()),width95_mm=float((3.92*ref['sigma'][:,hi][k]).mean()*1000))
        rows.append(entry)
      if arm==CFG['arms'][0] and subset=='motion_valid':
       err=np.linalg.norm(z['velocity']*h-target[:,hi],axis=-1)
       rows.append(dict(episode=ep,seed=seed,arm='constant_velocity',horizon=h,subset=subset,count=int(k.sum()),error_mm=float(err[k].mean()*1000),nll=None,coverage95=None,width95_mm=None))
    checks.append(dict(episode=ep,seed=seed,arm=arm,steps=600,plan_exact=True,initializer_exact=True,optimizer_steps_exact=True))
 groups=collections.defaultdict(list)
 for row in rows:groups[(row['episode'],row['arm'],row['horizon'],row['subset'])].append(row)
 means=[]
 for (ep,arm,h,subset),rr in groups.items():
  v=dict(episode=ep,arm=arm,horizon=h,subset=subset,seeds=len(rr),count=rr[0]['count'])
  for metric in ['error_mm','nll','coverage95','width95_mm']:
   vv=[x[metric] for x in rr if x[metric] is not None];v[metric]=float(np.mean(vv)) if vv else None
  means.append(v)
 aggregates=[]
 for arm in ['cf011_robot',*CFG['arms'],'static','constant_velocity']:
  for subset in ['all_targets','motion_valid']:
   rr=[x for x in means if x['arm']==arm and x['subset']==subset]
   if not rr:continue
   v=dict(arm=arm,subset=subset,event_horizon_cells=len(rr))
   for metric in ['error_mm','nll','coverage95','width95_mm']:
    vv=[x[metric] for x in rr if x[metric] is not None];v[metric]=float(np.mean(vv)) if vv else None
   aggregates.append(v)
 save(BASE/'comparison.json',dict(time=now(),unit='episode-horizon-seed equal weighted; pseudo-correspondence dynamics, not contact truth',rows=rows,means=means,aggregate=aggregates))
 save(BASE/'audits/completion.json',dict(time=now(),runs=24,updates=14400,checks=checks,hash_checks=hashchecks,max_independent_nll_difference=max_nll,max_target_difference_m=max_target,manual_ranges_used=False,test_access=False,contact_model_changed=False))
 stage('audited_pending_decision')
 print(json.dumps(aggregates),flush=True)
if __name__=='__main__':main()
