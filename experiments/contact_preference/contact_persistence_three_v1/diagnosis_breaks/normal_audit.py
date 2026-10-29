from pathlib import Path
import importlib.util,json,numpy as np,h5py,trimesh,traceback
D=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_preference/contact_persistence_three_v1')
sp=importlib.util.spec_from_file_location('nd',D/'src/analyze_v3.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
c=next(x for x in json.loads((D/'cases.json').read_text()) if x['id']=='switch');base=D/'data/switch';out=base/'estimates_v2';motion=np.load(base/'motion.npz');robot=M.A.RobotDepth(448);rows=[]
with h5py.File(c['rgbd'],'r') as h:
 for t in [74,75,76]:
  H=motion['hand_to_world'][t];q=motion['measured_finger_q'][t]
  robot.render(q,np.linalg.inv(H)@h['agentview/camera_to_world'][t],h['agentview/K'][t])
  mesh,_=M.A.M.geo.surface(robot,[2,5])
  for sf in [1,38,76]:
   a=np.load(out/f'{sf}_{t:03d}.npz');src=np.load(out/f'source_{sf}.npz')
   mask=(a['spread']<=.002)&src['normal_ok']&(a['distance']+a['spread']<=.003)
   ix=np.flatnonzero(mask)
   if not len(ix):continue
   close,dist,tri=trimesh.proximity.closest_point_naive(mesh,a['hand'][ix])
   nh=(src['normal']@a['fit_R'])@np.linalg.inv(H[:3,:3]).T
   cos=np.abs(np.einsum('ij,ij->i',nh[ix],mesh.face_normals[tri]))
   rows.append({'frame':t,'source':sf,'near_points':len(ix),'old_normal_pass':int((a['normal_cos'][ix]>=.7).sum()),'exact_triangle_normal_pass':int((cos>=.7).sum()),'max_cos_difference':float(np.max(np.abs(cos-a['normal_cos'][ix]))),'distance_difference_m':float(np.max(np.abs(dist-a['distance'][ix])))})
(D/'diagnosis_breaks/normal_triangle_audit.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
