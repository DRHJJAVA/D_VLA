import pathlib,sys,json,traceback,time
import numpy as np,h5py
from geometry import ROOT,RUN,CAMS,load_objects
from visibility import visible_reference,fuse_visible_features
try:
 e=json.loads((RUN/'episodes.json').read_text())[0];records=[];total_one=total_two=total_none=0
 for t in [13,38,67,91,117]:
  objects,meta=load_objects(e,t);xyz=np.concatenate([o['xyz'] for o in objects]);cams=[]
  with h5py.File(ROOT/'experiments/contact_joint_object/cache/rgbd'/f"{e['id']}.h5",'r') as h:
   i=meta['index']
   for name in CAMS:
    z=h[name];K=z['K'][()];K=K[i] if K.ndim==3 else K
    T=z['camera_to_world'][()];T=T[i] if T.ndim==3 else T
    cams.append(dict(K=K,camera_to_world=T,depth=z['depth_m'][i],robot_mask=z['robot_mask'][i]))
  v=visible_reference(xyz,cams);vis=v['visible'];w=v['weights']
  assert np.all(w[~vis]==0)
  known=vis.any(1);assert np.allclose(w[known].sum(1),1,atol=1e-6)
  assert np.all(v['current_unknown']==~known)
  f=np.ones((len(xyz),2,4),np.float32);a=fuse_visible_features(f,v)
  f[~vis]=100000.;b=fuse_visible_features(f,v)
  assert np.max(np.abs(a-b))==0
  residuals=[]
  for j,c in enumerate(cams):
   uv=v['uv'][vis[:,j],j].astype(int);p=xyz[vis[:,j]]
   cam=(p-c['camera_to_world'][:3,3])@c['camera_to_world'][:3,:3]
   residuals.extend(np.abs(cam[:,2]-c['depth'][uv[:,1],uv[:,0]]).tolist())
   assert not c['robot_mask'][uv[:,1],uv[:,0]].any()
  count=vis.sum(1);one=int((count==1).sum());two=int((count==2).sum());none=int((count==0).sum())
  total_one+=one;total_two+=two;total_none+=none
  blocked=[dict(c,robot_mask=np.ones_like(c['robot_mask'],bool)) for c in cams]
  vb=visible_reference(xyz,blocked);assert vb['current_unknown'].all() and not vb['weights'].any()
  records.append(dict(frame=t,points=len(xyz),one_visible=one,two_visible=two,neither_visible=none,max_depth_residual_m=max(residuals,default=0),occluded_feature_effect=0))
 assert total_one>0 and total_two>0
 out=dict(passed=True,scope='five training frames, geometric visibility and occlusion; not semantic accuracy',episode=e['id'],records=records,robot_overlay_violations=0,synthetic_all_occluded_unknown=True)
 (RUN/'checks/crossview_geometry.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
except Exception as exc:
 out=dict(error_type=type(exc).__name__,message=str(exc)[:300],frames=[dict(file=f.filename,line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)])
 (RUN/'checks/crossview_failure.json').write_text(json.dumps(out));print(json.dumps(out));sys.exit(1)
