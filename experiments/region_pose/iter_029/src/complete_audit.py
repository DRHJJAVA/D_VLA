from pathlib import Path
import sys,json,hashlib,ast,random,numpy as np,torch
B=Path(__file__).resolve().parents[1];R=B.parents[2]
sys.path.insert(0,str(B/'src'))
import train as t
from model import RegionPoseNet
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2))
def check(lock,root):
 d=json.loads(lock.read_text())
 for f,h in d.items():
  p=Path(f) if Path(f).is_absolute() else root/f
  assert sha(p)==h,('hash',str(lock),f)
 return len(d)
cfg=json.loads((B/'config.json').read_text());m=json.loads((B/'manifest.json').read_text())
checks={n:check(B/n,R) for n in ['source_locked.json','input_locked.json']}
checks['training_inputs_locked.json']=check(B/'training_inputs_locked.json',B)
for it in ['016','027','028']:checks[it]=check(B.parent/('iter_'+it)/'completed_locked.json',R)
init=json.loads((B/'initialization/locked.json').read_text())
assert sha(B/'initialization/initial.pt')==init['checkpoint_sha256']
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
rows=t.loadrows(m,torch.device('cuda'))
def met(score,d):
 out=t.metrics(score,d);y=d['target'].cpu().numpy().astype(bool);known=d['known'].cpu().numpy()
 for b in [.01,.02,.05,.1,.2,.3]:
  ids=np.argsort(-np.asarray(score),kind='stable')[:max(1,int(np.ceil(len(score)*b)))]
  suffix=str(int(b*100));out['all_coverage'+suffix]=float(y[ids].sum()/max(1,y.sum()))
  out['all_unknown_fraction'+suffix]=float((~known[ids]).mean())
  out['all_known_negative_fraction'+suffix]=float((known[ids]&~y[ids]).mean())
 return out
def aggregate(rr):
 keys=[k for k,v in rr[0].items() if isinstance(v,(int,float)) and k not in ['offset']]
 eps=[]
 for ep in sorted({r['episode'] for r in rr}):
  ar=[r for r in rr if r['episode']==ep]
  eps.append(dict(episode=ep,**{k:float(np.mean([r[k] for r in ar])) for k in keys}))
 return dict(mean={k:float(np.mean([r[k] for r in eps])) for k in keys},episodes=eps,offsets=[dict(offset=o,**{k:float(np.mean([r[k] for r in rr if r['offset']==o])) for k in keys}) for o in [15,10,5,1]],rows=rr)
summaries=[];predmax=0.;training_checks=[]
for run in cfg['runs']:
 name=run['group']+'_'+run['variant'];D=B/'runs'/name
 hist=json.loads((D/'history.json').read_text());plan=json.loads((B/'plans'/(name+'.json')).read_text())
 assert len(hist)==run['epochs']
 actual=[dict(epoch=h['epoch'],**r) for h in hist for r in h['selected_queries']]
 assert actual==plan and len(actual)==240
 assert hist[-1]['optimizer_steps']==240
 assert json.loads((D/'initialization.json').read_text())['state_sha256']==init['state_sha256']
 best=torch.load(D/'best.pt',map_location='cpu',weights_only=False);last=torch.load(D/'last.pt',map_location='cpu',weights_only=False)
 assert best['epoch']==min(hist,key=lambda h:h['val_loss'])['epoch']
 assert last['epoch']==run['epochs'] and last['input_lock']==best['input_lock']
 assert max(float(v['step']) for v in last['optimizer']['state'].values())==240
 assert all(k in last for k in ['rng_python','rng_numpy','rng_cpu','rng_cuda'])
 training_checks.append(dict(run=name,epochs=len(hist),steps=240,best_epoch=best['epoch'],plan_exact=True,optimizer_rng_present=True))
 for ckname,ck in [('best',best),('last',last)]:
  net=RegionPoseNet().cuda();net.load_state_dict(ck['state']);net.eval()
  rr=[]
  outdir=B/'evaluation'/name/ckname;outdir.mkdir(parents=True)
  for row,d in rows:
   with torch.inference_mode():
    a,b=t.forward(net,d);logit=a if run['variant']=='object_only' else b
    score=logit.sigmoid().cpu().numpy();headloss=float(t.loss(logit,d))
   rec=dict(key=row['key'],episode=row['episode'],event=row['event'],offset=row['offset'],split=row['split'],in_training_group=row['split']=='train' and (run['group']=='expanded10' or row['episode'] in cfg['control_episodes']),loss=headloss,**met(score,d))
   rr.append(rec)
   np.savez_compressed(outdir/(row['key']+'.npz'),score=score,logit=logit.cpu().numpy(),target=d['target'].cpu().numpy(),known=d['known'].cpu().numpy())
   if ckname=='best' and row['split']=='val':
    old=np.load(D/'val_predictions'/(row['key']+'.npz'))
    diff=float(np.max(np.abs(old['score']-score)));predmax=max(predmax,diff);assert diff==0
  result=dict(run=name,checkpoint=ckname,epoch=ck['epoch'],validation=aggregate([r for r in rr if r['split']=='val']),training_group=aggregate([r for r in rr if r['in_training_group']]),all_training_diagnostic=aggregate([r for r in rr if r['split']=='train']))
  dump(outdir/'summary.json',result);summaries.append(result)
  del net
# All-query baselines, fixed saved causal geometry, no future reference pose.
base=[]
for row,d in rows:
 source=np.load(row['target_source']);rawroot=B.parent/('iter_027' if row['split']=='train' else 'iter_028')
 sr=next(r for r in json.loads((rawroot/'summary.json').read_text())['rows'] if r['event']==row['event'] and r['query']==row['query'])
 raw=np.load(R/sr['source_file']);sel=source['selected_index']
 geo=raw['early_geometry_score'][sel]
 cen=-np.linalg.norm(d['xyz'].cpu().numpy(),axis=1)
 rand=[met(np.random.default_rng(2026+j).random(len(geo)),d) for j in range(64)]
 base.append(dict(key=row['key'],episode=row['episode'],offset=row['offset'],split=row['split'],current_geometry=met(geo,d),center=met(cen,d),random_mean={k:float(np.mean([r[k] for r in rand])) for k in rand[0]},random_repetitions=64))
dump(B/'evaluation/baselines.json',base)
reuse=[json.loads(p.read_text()) for p in (B/'audits').glob('*_feature_reuse.json')]
old=[r for r in reuse if r['existing']]
dump(B/'audits/completion_audit.json',dict(checks=checks,training=training_checks,initialization_sha256=init['state_sha256'],validation_prediction_max_abs_diff=predmax,feature_comparisons=len(old),feature_exact_reuse=sum(r['mode']=='exact_verified_reuse' for r in old),fresh_current_frame=len(reuse)-sum(r['mode']=='exact_verified_reuse' for r in old),feature_max_difference=max(r['feature_max_abs_diff'] for r in old),test_read=False,stage2_accepted=False))
dump(B/'evaluation/summary.json',summaries)
dump(B/'stage_state.json',dict(stage='evaluation_complete_pending_review'))
print('EVALUATION_AUDIT_COMPLETE',len(summaries),flush=True)
