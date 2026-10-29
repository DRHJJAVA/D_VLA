from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/region_pose/iter_012';P=B.parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def check(lock,root):
 d=json.loads(lock.read_text())
 for f,h in d.items():assert sha(root/f)==h,(lock,f)
 return len(d)
checks={}
for n in ['source_locked.json','input_locked.json']:checks[n]=check(B/n,R)
for sp in ['train','val']:checks[sp+'_lock']=check(B/sp/'locked.json',B/sp)
freeze=json.loads((B/'pre_validation_freeze.json').read_text())
assert freeze['train_summary_sha256']==sha(B/'train/summary.json') and freeze['train_lock_sha256']==sha(B/'train/locked.json')
assert freeze['source_lock_sha256']==sha(B/'source_locked.json') and freeze['input_lock_sha256']==sha(B/'input_locked.json')
rows=[]
for sp in ['train','val']:
 for c in json.loads((B/sp/'summary.json').read_text())['cases']:
  n=c['event']+'_query'+str(c['query']);z=np.load(B/sp/(n+'.npz'));frames=z['selected_frames']
  assert np.isfinite(z['source_xyz']).all() and np.isfinite(z['mapped_query_xyz']).all()
  assert len(np.unique(frames))==len(frames)
  source=json.loads((P/'iter_011'/sp/'audits'/(n+'.json')).read_text())
  expected=[source['selected_held_frames'][0]]
  for x in source['time_comparisons']:
   if not x['valid']:break
   expected.append(x['frames'][1])
  assert frames.tolist()==expected
  dist,ix=cKDTree(z['query_xyz']).query(z['mapped_query_xyz']);assert np.array_equal(ix,z['query_nearest_index']) and np.allclose(dist,z['distance_to_query'],atol=1e-15)
  reps=z['voxel_representative_index'];a=z['mapped_query_xyz']
  for j in reps:
   ids=np.flatnonzero((np.linalg.norm(a-a[j],axis=1)<=.002)&(dist<=.003))
   assert len(np.unique(z['frame'][ids]))>=2
  for t in frames:
   sel=z['frame']==t;h=z['source_hand_xyz'][sel];q=z['finger_q_'+str(t)]
   assert np.all((abs(h[:,0])<=.008)&(h[:,2]>=.0854)&(h[:,2]<=.1014)&(h[:,1]>=q[1]+.0005)&(h[:,1]<=q[0]-.0005))
  rows.append({k:c[k] for k in ['id','split','query','selected_frames','raw_points','near_query_points','supported_1mm_voxels','unsupported_or_unseen_points','query_visible_any','query_visible_two_frames','query_surface_points']})
assert len(rows)==9
for n in ['locked.json','display_revision_locked.json']:checks['review_'+n]=check(P/'reviews/iter_012'/n,R)
out={'source_input_split_and_display_locks_verified':checks,'pre_validation_freeze_verified':True,'candidate_provenance_and_distinct_frame_support_verified':True,'finite_volume_verified':True,'cases':rows,'training_performed':False,'not_ground_truth_audit':True}
(B/'audits/completion_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
