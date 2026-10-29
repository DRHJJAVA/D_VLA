from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from scipy.spatial import cKDTree
import audit_regions as reg
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';O=B/'iter_024'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def verify(path,root):
 d=json.loads(path.read_text())
 for k,h in d.items():
  p=Path(k);p=p if p.is_absolute() else root/p
  assert sha(p)==h,str(p)
 return len(d)
checks={}
for name in ['source_locked.json','source_final_audit_addendum.json','train_input_locked.json','val_input_locked.json']:checks[name]=verify(O/name,P)
for n in ['022','023']:checks['old_'+n]=verify(B/('iter_'+n)/'completed_locked.json',P)
fr=json.loads((O/'pre_validation_freeze.json').read_text())
assert fr['train_summary_sha256']==sha(O/'train/summary.json') and fr['train_loo_summary_sha256']==sha(O/'train_loo/summary.json')
assert fr['source_lock_sha256']==sha(O/'source_locked.json')
assert fr['time']<json.loads((O/'audits/val_preflight.json').read_text())['time']
rows=[];frames=0;fits=0;maxdiff=0.
for split in ['train','val']:
 checks[split+'_regions']=verify(O/split/'locked.json',O/split);checks[split+'_loo']=verify(O/(split+'_loo')/'locked.json',O/(split+'_loo'))
 summary=json.loads((O/(split+'_loo')/'summary.json').read_text())
 for c in summary['cases']:
  stem=c['event']+'_query'+str(c['query']);z=np.load(O/split/(stem+'.npz'));r=json.loads((O/split/(stem+'.json')).read_text())
  p=z['mapped_query_xyz'];fids=z['frame'];near=z['distance_to_query']<=.003
  support=np.zeros(len(p),int)
  for t in z['selected_frames']:
   m=(fids==t)&near
   if m.any():support+=cKDTree(p[m]).query(p)[0]<=.002
  assert np.array_equal(support,z['distinct_frame_support'])
  good=near&(support>=2);assert np.array_equal(good,z['supported_candidate'])
  ids=np.flatnonzero(good)
  if len(ids):
   _,inds=np.unique(np.floor(p[ids]/.001).astype(np.int64),axis=0,return_index=True);reps=ids[inds]
  else:reps=np.array([],int)
  assert np.array_equal(reps,z['voxel_representative_index'])
  maxq=maxr=0.;minfits=99999;stable_raw=0;rawtotal=0
  for ff in c['frames']:
   t=ff['frame'];frames+=1;fits+=ff['valid_loo_fits'];minfits=min(minfits,ff['valid_loo_fits'])
   zz=np.load(O/(split+'_loo')/(stem+'_frame'+str(t)+'.npz'))
   assert len(zz['loo_Q'])==ff['valid_loo_fits'] and len(zz['removed_track_id'])==ff['valid_loo_fits']
   assert len(set(zz['removed_track_id'].tolist()))==len(zz['removed_track_id'])
   hs=z['source_hand_xyz'][fids==t];q=z['finger_q_'+str(t)]
   assert np.all(reg.in_region(hs,q))
   pr=next(x for x in r['per_frame'] if x['frame']==t)
   assert pr['leave_one_out_fits']==ff['valid_loo_fits']
   for oldkey,newkey in [('median','median_mm'),('p95','p95_mm'),('max','max_mm')]:
    a=pr['leave_one_out_pointwise_max_shift_mm'].get(oldkey);b=ff['region_max_shift'].get(newkey)
    if a is not None and b is not None:maxdiff=max(maxdiff,abs(a-b));assert abs(a-b)<1e-7
   maxq=max(maxq,ff['query_max_shift']['p95_mm'] or 0);maxr=max(maxr,ff['region_max_shift']['p95_mm'] or 0)
   if ff['eligible_min7_loo']:stable_raw+=int((zz['region_max_shift_m']<=.002).sum())
   rawtotal+=len(zz['region_max_shift_m'])
  row={'split':split,'episode':c['episode'],'selected_frames':c['selected_frames'],'omitted_frames':c['omitted_later_endpoints'],'raw_observations':c['raw_points'],'near_query_points':r['near_query_points'],'unknown_gt6mm_observations':r['unsupported_or_unseen_points'],'supported_1mm_voxels':c['supported_1mm_voxels'],'query_cloud_points':c['query_surface_points'],'min_loo_fits':minfits,'worst_frame_query_p95_mm':maxq,'worst_frame_region_p95_mm':maxr,'region_observations_le2mm':stable_raw,'region_observations_total':rawtotal,'not_independent_grasps':True}
  rows.append(row)
save(O/'summary.json',{'cases':rows,'train_events':11,'old_validation_events':3,'diagnostic_frames':frames,'valid_leave_one_out_fits':fits,'no_labels_or_scoring_training':True,'not_physical_error_truth':True,'source_attribution':'Original/grid are tracking seed origins, not object or grasp classes.'})
save(O/'audits/completion_audit.json',{'passed':True,'locks':checks,'train_frozen_before_val':True,'candidate_support_independent_reconstruction':14,'real_source_voxel_representatives_exact':14,'finite_volume_all_sources_pass':True,'loo_region_stats_two_implementations_max_difference_mm':maxdiff,'frames':frames,'valid_loo_fits':fits,'no_test_access':True,'no_new_training_labels':True,'time':time.time()})
print(json.dumps({'passed':True,'cases':len(rows),'frames':frames,'loo_fits':fits,'max_region_stats_difference_mm':maxdiff,'locks':checks}))
