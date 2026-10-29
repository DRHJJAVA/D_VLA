import sys,json,traceback,re,hashlib
from pathlib import Path
import numpy as np,h5py,torch
R=Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003');sys.path.insert(0,str(R/'src'))
from data import P0Dataset
from features import FrozenSmolFeatures
out={}
f=FrozenSmolFeatures.__new__(FrozenSmolFeatures);f.device='cpu'
f.image_grid=lambda rgb:torch.stack([torch.ones(768,32,32),torch.full((768,32,32),99.)])
rgb=np.zeros((1,2,256,256,3),np.uint8)
uv=np.array([[[[0,0],[0,0]],[[255,255],[255,255]],[[0,128],[0,128]],[[128,128],[128,128]]]],np.float32)
vis=np.zeros((1,4,2),bool);vis[:,:,0]=True
v=f.point_features(rgb,uv,vis)
assert np.max(abs(v-1))==0,'source_features_changed_by_border_or_invisible_camera'
vis[:]=True;both=f.point_features(rgb,uv,vis)
assert np.max(abs(both-50))==0
out['feature_boundary']={'status':'passed','source_only_values':v[0,:,0].tolist(),'both_visible_values':both[0,:,0].tolist(),'actual_production_method':True}
audit=json.loads((R/'p0/data_audit.json').read_text());ds=P0Dataset()
orig=h5py.Dataset.__getitem__;runs=[]
for s in audit['samples']:
 if 'boundary' in s['case']:continue
 ep=s['episode_id'];t=s['current_frame'];ds._cache.clear()
 baseline=ds.load_window(ep,t)
 calls=[0]
 def guard(obj,key):
  calls[0]+=1
  if '/cache/rgbd/' in obj.file.filename:
   first=key[0] if isinstance(key,tuple) else key
   if not isinstance(first,(int,np.integer)) or int(first)>t:raise RuntimeError('future_observation_read')
  if '/cache/observations/' in obj.file.filename:
   m=re.search(r'/frames/(\d+)(?:/|$)',obj.name)
   if m and int(m.group(1))>t:raise RuntimeError('future_observation_read')
  return orig(obj,key)
 ds._cache.clear();h5py.Dataset.__getitem__=guard
 guarded=ds.load_window(ep,t)
 positive=False
 try:
  with h5py.File(ds.old/'cache/rgbd'/(ep+'.h5'),'r') as h:h['agentview/depth_m'][t+1]
 except RuntimeError:positive=True
 finally:h5py.Dataset.__getitem__=orig
 assert positive
 equal={k:bool(np.array_equal(baseline[k],guarded[k],equal_nan=True)) for k in baseline if isinstance(baseline[k],np.ndarray)}
 assert all(equal.values())
 oldactions=ds.actions[ep].copy()
 ds.actions[ep][t+1:]=777;ds._cache.clear();changed=ds.load_window(ep,t);ds.actions[ep]=oldactions
 condition={k:bool(np.array_equal(baseline[k],changed[k],equal_nan=True)) for k in baseline if isinstance(baseline[k],np.ndarray) and k not in ['actions','actions_raw','action_valid']}
 assert all(condition.values()) and not np.array_equal(baseline['actions'],changed['actions'])
 runs.append({'episode':ep,'frame':t,'arrays_checked':len(equal),'guard_calls':calls[0],'future_positive_control_blocked':positive,'guarded_input_identical':True,'future_action_perturbation_condition_identical':True})
out['causality']={'status':'passed','runs':runs}
out['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'src').glob('*.py')}
out['status']='passed'
(R/'p0/visibility_v3_contracts.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
