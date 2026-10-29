from pathlib import Path
import json,numpy as np,h5py
from datetime import datetime,timezone,timedelta
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/stage012_v1';s=json.loads((O/'stage2_audit.json').read_text())
cf=R/'experiments/contact_frames/cf017'
def area(xyz,uv,step):
 uv=np.asarray(uv,dtype=np.int64);xyz=np.asarray(xyz)
 lut={tuple(uv[i]):xyz[i] for i in range(len(uv))}
 def tri(a,b,c):
  aa=np.linalg.norm(a-b);bb=np.linalg.norm(a-c);cc=np.linalg.norm(b-c)
  if max(aa,bb,cc)>.012:return 0.
  return np.linalg.norm(np.cross(b-a,c-a))/2
 a=0.;n=0
 for (u,v),p in lut.items():
  if u%step or v%step:continue
  q=lut.get((u+step,v));r=lut.get((u,v+step));t=lut.get((u+step,v+step))
  if q is None or r is None or t is None:continue
  v1=tri(p,q,r);v2=tri(t,q,r)
  a+=v1+v2
  n+=int(v1>0)+int(v2>0)
 return a,n
for name,rec in s['cross_task_localization'].items():
 base=cf/'cases'/name
 ref=np.load(base/'artifacts/reference.npz');mot=np.load(base/'motion.npz')
 t=rec['reference_frame'];rad=rec['hand_radius_m_diagnostic'];H=mot['hand_to_world'][t];q=mot['measured_finger_q'][t]
 centers=np.stack([H[:3,:3]@np.array([0,q[0],.093])+H[:3,3],H[:3,:3]@np.array([0,q[1],.093])+H[:3,3],H[:3,:3]@np.array([0,0,.097])+H[:3,3]])
 X=ref['surface_xyz'];uv=ref['surface_uv'];sel=np.min(np.linalg.norm(X[:,None,:]-centers[None,:,:],axis=2),axis=1)<rad
 a1,n1=area(X[sel],uv[sel],1);a2,n2=area(X[sel],uv[sel],2)
 rec['initial_source_triangulated_area_mm2']={'one_pixel_grid':a1*1e6,'two_pixel_grid':a2*1e6,'triangles_one_pixel':n1,'triangles_two_pixel':n2,'source_unique_by_initial_pixel':True,'occluded_or_unobserved_area_not_included':True,'not_true_contact_area':True}
 rec['repeated_identical_source_does_not_increase_unique_area']=True
 if name=='drawer_demo_0':
  fit=rec['local_fit']
  for frame in ['70','88']:
   # Cross-view heldout errors were already computed without heldout in fitting.
   fit[frame]['heldout_any_over_3mm']=any(v>3 for v in fit[frame]['heldout_near_errors_mm'])
s['created_beijing_area_addendum']=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
(O/'stage2_audit.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
print({k:{'area1_mm2':round(v['initial_source_triangulated_area_mm2']['one_pixel_grid'],2),'area2_mm2':round(v['initial_source_triangulated_area_mm2']['two_pixel_grid'],2),'triangles':v['initial_source_triangulated_area_mm2']['triangles_one_pixel']} for k,v in s['cross_task_localization'].items()})
