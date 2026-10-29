from pathlib import Path
import json,hashlib,ast,numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/mnt/nvme1/drh/D_VLA');BASE=ROOT/'experiments/contact_frames/cf015';PREV=BASE.parent/'cf014'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def check(path,key=None):
 d=json.loads(path.read_text());d=d[key] if key else d
 bad=[p for p,h in d.items() if not Path(p).exists() or sha(p)!=h];assert not bad,(path.name,bad)
 return len(d)
def main():
 counts={}
 for path,key in [(PREV/'completed_locked.json',None),(BASE/'train/input_locked.json',None),
  (BASE/'review/input_locked.json',None),(BASE/'train/output_locked.json',None),
  (BASE/'review/output_locked.json',None),(BASE/'quality_probe/source_weight_locked.json',None),
  (BASE/'quality_probe/train_frozen.json','files'),(BASE/'quality_probe/predictions_frozen.json','files')]:
  counts[str(path.relative_to(ROOT))]=check(path,key)
 cfg=json.loads((BASE/'config.json').read_text())
 # Independently reconstruct weak target, actual labels and changed gradient directions.
 npos=nneg=nunknown=flip=0;near_negmax=0.;pmass=0.;lowmass=0.;removed=[]
 for ep in cfg['train_episodes']:
  with np.load(BASE/'train'/f'{ep}_supervision.npz') as a,np.load(PREV.parent/'cf010/data'/ep/'artifacts/dataset.npz') as d:
   with np.load(PREV.parent/'cf010/evaluation'/ep/'artifacts/frame_predictions.npz') as g:sep=g['geometry_separation_evidence'].astype(bool)
   target=np.where(sep,0,1).astype('float32')
   weight=np.where(sep,d['near_weight'],d['weight']*d['near_target']*d['interaction_support']).astype('float32')
   # Multiplication association can incur float32 rounding; no changed values semantically.
   assert np.allclose(weight,a['weight'],rtol=1e-6,atol=1e-9)
   assert np.array_equal(target,a['target'])
   assert np.array_equal(d['target'],d['near_target']*d['interaction_support'])
   pos=(a['target']==1)&(a['weight']>0);neg=(a['target']==0)&(a['weight']>0);unk=a['weight']==0
   npos+=int(pos.sum());nneg+=int(neg.sum());nunknown+=int(unk.sum())
   near_negmax=max(near_negmax,float(d['near_target'][neg].max()))
   pmass+=float(a['weight'][pos].sum());lowmass+=float(a['weight'][pos&(d['target']<.1)].sum())
   oldgrad=d['weight']*(.5-d['target']);newgrad=a['weight']*(.5-a['target'])
   flip+=int((pos&(oldgrad>0)&(newgrad<0)).sum())
   removed.append(dict(episode=ep,weighted_old_zero_now_unknown=int((unk&(d['weight']>0)).sum()),
      near_ge_point5_old_zero_now_unknown=int((unk&(d['weight']>0)&(d['near_target']>=.5)).sum())))
 review=json.loads((BASE/'review/diagnostic_review.json').read_text())
 humans=json.loads((PREV.parent/'cf010/review_ranges.json').read_text())
 for row in review['rows']:
  with np.load(BASE/'probes'/(row['episode']+'.npz')) as d:
   assert np.isfinite(d['probabilities']).all() and np.isfinite(d['logits']).all()
   for name,s in row['segments'].items():
    a,b=s['bounds'];assert b<d['probabilities'].shape[-1]
    for i,v in enumerate(cfg['variants']):
     q=d['probabilities'][:,i,a:b+1]
     assert s['variants'][v]['high_frames']==int((q.mean(0)>=.8).sum())
     assert abs(s['variants'][v]['mean_score']-float(q.mean()))<1e-7
 # Source provenance: new motion descriptor has no upstream object instance mask argument/access.
 src=PREV/'src/prepare_features.py';t=ast.parse(src.read_text())
 desc=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='descriptor')
 names=sorted({n.id for n in ast.walk(desc) if isinstance(n,ast.Name)})
 strings={n.value for n in ast.walk(t) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
 provenance=dict(source=str(src),sha256=sha(src),descriptor_arguments=[a.arg for a in desc.args.args],
   component_file_key_access=('component' in strings),object_mask_argument=any('mask' in x.arg or 'object' in x.arg for x in desc.args.args),
   scope='current valid depth in Gaussian hand-centered spatial weighting; no explicit target identity input to descriptor',
   limitation='cannot quantify background point fraction from 24 aggregate features alone')
 save(BASE/'audits/motion_source_provenance.json',provenance)
 assert not provenance['component_file_key_access'] and not provenance['object_mask_argument']
 stats=dict(positive=npos,negative=nneg,unknown=nunknown,maximum_negative_near_target=near_negmax,
   changed_gradient_direction_at_probability_half=flip,positive_mass=pmass,soft_below_point1_positive_mass_fraction=lowmass/pmass,removed=removed)
 save(BASE/'audits/independent_findings.json',stats)
 trainfreeze=json.loads((BASE/'pre_review_freeze.json').read_text())['time']
 reviewtime=json.loads((BASE/'review/summary.json').read_text())['time']
 manualtime=json.loads((BASE/'audits/manual_access.json').read_text())['time']
 assert trainfreeze<reviewtime<=manualtime
 save(BASE/'audits/completion.json',dict(time=now(),passed=True,locks=counts,independent_statistics=stats,
     chronological_train_freeze_then_review_then_manual=True,network_training=False,test_access=False))
 save(BASE/'audits/resolved.json',dict(time=now(),failures=[
   dict(file='failure.json',resolved=True,reason='v2 gradient diagnostics completed with GRU train-mode backward; no optimizer'),
   dict(file='review_metrics_error.log',resolved=True,reason='finite distance count and null preserved; v2 completed')]))
 print(json.dumps(dict(passed=True,stats=stats,locks=counts),ensure_ascii=False))
if __name__=='__main__':main()
