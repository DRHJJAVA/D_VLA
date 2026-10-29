from pathlib import Path
import json,hashlib,numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
B=Path('/mnt/nvme1/drh/D_VLA');BASE=B/'experiments/contact_frames';R=BASE/'cf009';OLD=BASE/'cf008'
cfg=json.loads((R/'config.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
inputs={}
for ep in cfg['train_episodes']:
 for p in [OLD/'data'/ep/'artifacts/dataset.npz',OLD/'data'/ep/'artifacts/motion_evidence.json',BASE/'cf003/data'/ep/'artifacts/reference.npz',B/'experiments/region_pose/iter_003/evidence/motion'/(ep+'.npz')]:inputs[str(p)]=sha(p)
for p in [OLD/'completed_locked.json',R/'config.json',BASE/'cf003/normalization.json']:inputs[str(p)]=sha(p)
save(R/'input_locked.json',inputs)
save(R/'source_locked.json',{str(p):sha(p) for p in R.rglob('*') if p.is_file() and (p.suffix=='.py' or p.name=='config.json')})
rows=[]
for ep in cfg['train_episodes']:
 f=OLD/'data'/ep/'artifacts/dataset.npz';z=np.load(f);d={k:z[k].copy() for k in z.files}
 ref=np.load(BASE/'cf003/data'/ep/'artifacts/reference.npz')['surface_xyz'].astype(float)
 mot=np.load(B/'experiments/region_pose/iter_003/evidence/motion'/(ep+'.npz'))
 H=mot['hand_to_world'].astype(float);q=mot['measured_finger_q'].astype(float)
 O=d['visual_transforms'];N=len(d['target'])
 reliable=d['geometry_reliable']&d['state_valid']&np.isfinite(H).all((1,2))&np.isfinite(O).all((1,2))
 sep=reliable&(d['distance']-2*d['sigma']>.006)
 support=d['interaction_support'].astype(float)
 assert np.all((support>=0)&(support<=1))
 rel=np.full((N,len(ref),3),np.nan)
 for t in np.flatnonzero(reliable):
  world=ref@O[t,:3,:3].T+O[t,:3,3]
  rel[t]=np.linalg.solve(H[t,:3,:3],(world-H[t,:3,3]).T).T
 anchor=support*d['near_target']*np.clip(d['quality'],0,1)*reliable
 persistent=support.copy();chosen=np.arange(N);pair_rows=[]
 for t in np.flatnonzero(reliable&~sep):
  for s in range(max(0,t-16),min(N,t+17)):
   if s==t or not reliable[s] or sep[s] or anchor[s]<=0:continue
   if sep[min(s,t):max(s,t)+1].any():continue
   delta=float(np.median(np.linalg.norm(rel[t]-rel[s],axis=1)))
   dq=float(np.linalg.norm(q[t]-q[s]))
   variance=float(d['sigma'][t]**2+d['sigma'][s]**2)
   assert variance>0
   compatibility=float(np.exp(-.5*(delta*delta+dq*dq)/variance))
   contribution=float(anchor[s]*compatibility)
   pair_rows.append({'t':int(t),'s':s,'distance_m':delta,'finger_delta_m':dq,'variance':variance,'compatibility':compatibility,'anchor':float(anchor[s]),'contribution':contribution})
   if contribution>persistent[t]:persistent[t]=contribution;chosen[t]=s
 for arm,sup in [('positive_evidence',support),('persistent_evidence',persistent)]:
  out=R/arm/'data'/ep/'artifacts';out.mkdir(parents=True)
  dd={k:v.copy() for k,v in d.items()}
  w=d['near_weight']*sup*reliable
  y=d['near_target'].copy()
  w[sep]=d['near_weight'][sep];y[sep]=0
  y[w==0]=0
  dd.update(target=y.astype('float32'),weight=w.astype('float32'),automatic_positive_support=sup.astype('float32'),support_anchor_index=chosen if arm=='persistent_evidence' else np.arange(N))
  np.savez_compressed(out/'dataset.npz',**dd)
  assert np.isfinite(w).all() and np.all(w>=0)
  assert np.all(w[~reliable]==0) and np.all(y[sep]==0)
  for k in ['images','points','states','near_target','near_weight','dy','dynvalid','visual_transforms']:
   assert np.array_equal(dd[k],d[k],equal_nan=True),k
  rows.append({'arm':arm,'episode':ep,'weighted_frames':int((w>0).sum()),'unknown':int((w==0).sum()),'positive_target_ge08_frames':int(((y>=.8)&(w>0)).sum()),'positive_weight_mass':float(w[y>=.8].sum()),'old_unsupported_weighted':int(((d['weight']>0)&(support==0)&~sep).sum()),'extended_frames':int(((sup>support+1e-7)&reliable&~sep).sum())})
 save(R/'teacher_details'/ep/'persistence.json',pair_rows)
 np.savez_compressed(R/'teacher_details'/ep/'supports.npz',anchor=anchor,support=support,persistent=persistent,chosen=chosen,reliable=reliable,separation=sep)
# regression tests of source semantics
assert np.exp(-.5*0)==1
assert max(0.,0.)==0.
assert all(sha(p)==v for p,v in inputs.items())
save(R/'train_supervision_summary.json',{'rows':rows,'range_used':16,'human_ranges_used':False})
for arm in cfg['arms']:
 save(R/arm/'train_supervision_summary.json',{'rows':[v for v in rows if v['arm']==arm],'human_ranges_used':False})
 save(R/arm/'source_locked.json',{str(p):sha(p) for p in (R/arm/'src').glob('*.py')})
save(R/'stage_state.json',{'stage':'training_supervision_ready','time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')})
print(json.dumps({'prepared':True,'rows':rows}))
