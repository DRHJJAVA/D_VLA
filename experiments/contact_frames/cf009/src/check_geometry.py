from pathlib import Path
import json,numpy as np
B=Path('/mnt/nvme1/drh/D_VLA');R=B/'experiments/contact_frames/cf009'
cfg=json.loads((R/'config.json').read_text());mx=0.;rows=[]
for ep in cfg['train_episodes']:
 d=np.load(R.parent/'cf008/data'/ep/'artifacts/dataset.npz')
 ref=np.load(R.parent/'cf003/data'/ep/'artifacts/reference.npz')['surface_xyz'].astype(float)
 P=np.c_[ref,np.ones(len(ref))]
 mot=np.load(B/'experiments/region_pose/iter_003/evidence/motion'/(ep+'.npz'))
 H=mot['hand_to_world'];q=mot['measured_finger_q'];O=d['visual_transforms']
 pair=json.loads((R/'teacher_details'/ep/'persistence.json').read_text())
 ids=set([a[z] for a in pair for z in ['t','s']])
 rel={t:(P@O[t].T@np.linalg.inv(H[t]).T)[:,:3] for t in ids}
 for a in pair:
  t,s=a['t'],a['s']
  dist=np.median(np.sqrt(((rel[t]-rel[s])**2).sum(-1)))
  dq=np.sqrt(((q[t]-q[s])**2).sum())
  mx=max(mx,abs(dist-a['distance_m']),abs(dq-a['finger_delta_m']))
 assert mx<1e-10
 rows.append({'episode':ep,'pairs':len(pair)})
(R/'audits/geometry_reconstruction.json').write_text(json.dumps({'passed':True,'independent_homogeneous_coordinate_max_diff_m':mx,'rows':rows},indent=2))
print(json.dumps({'passed':True,'max_diff_m':mx}))
