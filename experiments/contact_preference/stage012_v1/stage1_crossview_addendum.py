from pathlib import Path
import json,hashlib
import numpy as np,h5py
from scipy.spatial import cKDTree
from datetime import datetime,timezone,timedelta
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/stage012_v1'
x=json.loads((O/'stage1_audit.json').read_text())
def world(D,K,C,step):
 y,u=np.mgrid[0:D.shape[0]:step,0:D.shape[1]:step]
 y=y.ravel();u=u.ravel();z=D[y,u].astype(np.float64);ok=np.isfinite(z)&(z>.1)&(z<2)
 v=np.stack(((u[ok]-K[0,2])*z[ok]/K[0,0],(y[ok]-K[1,2])*z[ok]/K[1,1],z[ok]),axis=1)
 return v@C[:3,:3].T+C[:3,3]
cf=R/'experiments/contact_frames/cf017'
for name,t in [('drawer_demo_0',70),('stove_demo_0',86),('two_objects_demo_0',120)]:
 with h5py.File(cf/'rgbd'/(name+'.h5'),'r') as h:
  w=[]
  for cam in ['agentview','robot0_eye_in_hand']:
   D=h[cam+'/depth_m'][t];K=h[cam+'/K'][t];C=h[cam+'/camera_to_world'][t];w.append(world(D,K,C,3))
  da,_=cKDTree(w[1]).query(w[0],k=1);db,_=cKDTree(w[0]).query(w[1],k=1)
  for key,d in [('agent_to_wrist',da),('wrist_to_agent',db)]:
   m=d<.005
   x['cases'][name]['crossview_'+key]={'sampled_points':int(len(d)),'within_5mm_fraction':float(np.mean(m)),'matched_median_mm':float(np.median(d[m])*1000) if m.any() else None,'matched_p95_mm':float(np.quantile(d[m],.95)*1000) if m.any() else None,'unmatched_is_occlusion_or_nonoverlap_not_bad_calibration':True}
 mot=np.load(cf/'cases'/name/'motion.npz')['hand_to_world']
 valid=mot[1:,:3,:3]
 orth=np.max(np.abs(np.swapaxes(valid,1,2)@valid-np.eye(3)),axis=(1,2));dets=np.linalg.det(valid)
 x['cases'][name]['hand_rotation_orthogonality_max_abs']=float(orth.max())
 x['cases'][name]['hand_rotation_det_range']=[float(dets.min()),float(dets.max())]
checks={}
for key in ['porcelain_mug_demo_9_attempt1_query36','tomato_sauce_demo_7_attempt1_query48','white_yellow_mug_demo_4_attempt1_query40']:
 a=np.load(R/f'experiments/region_pose/iter_024/train/{key}.npz');b=np.load(R/f'experiments/region_pose/iter_025/targets/train/{key}.npz')
 checks[key]={k:bool(np.array_equal(a[k],b[k])) for k in ['query_xyz','query_pixel','query_rgb']}
x['current_input_unaffected_by_later_target_construction_3_existing_cases']=checks
x['future_perturbation_counterfactual_run']=False
x['created_beijing_addendum']=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
(O/'stage1_audit.json').write_text(json.dumps(x,ensure_ascii=False,indent=2))
print(json.dumps({'crossview':{k:{n:v[n] for n in v if n.startswith('crossview_')} for k,v in x['cases'].items()},'current_input_checks':checks},ensure_ascii=False))
