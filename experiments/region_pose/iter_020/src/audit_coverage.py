from pathlib import Path
import json,hashlib,time,collections
import numpy as np
from scipy.spatial import ConvexHull,QhullError
P=Path('/mnt/nvme1/drh/D_VLA')
O=P/'experiments/region_pose/iter_020'
D=P/'experiments/region_pose/iter_011/train'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rel(p): return str(p.relative_to(P))
def spatial(a):
 a=np.asarray(a,dtype=float)
 if not len(a): return {'n':0,'span':None,'hull_area_px2':None,'grid8_count':0}
 out={'n':len(a),'span':np.ptp(a,axis=0).tolist()}
 if a.shape[1]==2:
  try: area=float(ConvexHull(a).volume) if len(a)>=3 else 0.
  except QhullError: area=0.
  out.update(hull_area_px2=area,grid8_count=len(np.unique(np.floor(a/8).astype(int),axis=0)))
 else:
  out['rms_radius_m']=float(np.sqrt(np.mean(np.sum((a-a.mean(0))**2,axis=1))))
  out['axis_std_m']=np.std(a,axis=0).tolist()
 return out
def finite(d):
 if isinstance(d,dict): return {k:finite(v) for k,v in d.items()}
 if isinstance(d,list): return [finite(v) for v in d]
 if isinstance(d,float) and not np.isfinite(d): return None
 return d
train=read(P/'data/v3/manifests/stage2_episode_split.json')['splits']['train']
S=read(D/'summary.json'); cases={r['id']:r for r in S['cases']}
assert len(train)==20 and len(cases)==len(S['cases']) and set(cases)<=set(train)
R13=P/'experiments/region_pose/iter_013/targets/train/summary.json'
current={r['episode'] for r in read(R13)['rows']}
paths={P/'data/v3/manifests/stage2_episode_split.json',D/'summary.json',D/'locked.json',R13,
 P/'experiments/region_pose/iter_003/completed_locked.json',P/'experiments/region_pose/iter_013/completed_locked.json'}
for ep in train:
 paths.add(P/f'experiments/region_pose/iter_003/evidence/motion/{ep}.json')
 for p in (P/'experiments/region_pose/iter_003/evidence/fusion').glob(ep+'_attempt*.json'): paths.add(p)
 if ep in cases:
  r=cases[ep];key=r['event']+'_query'+str(r['query'])
  for typ in ['trajectories','correspondences']:
   p=D/typ/(key+'.npz')
   if p.exists(): paths.add(p)
assert all(p.exists() for p in paths)
inputs={rel(p):sha(p) for p in sorted(paths)}
write(O/'input_locked.json',inputs)
sources={rel(p):sha(p) for p in [O/'设计方案.md',O/'config.json',O/'src/audit_coverage.py']}
write(O/'source_locked.json',sources)
oldlocks=[(D,read(D/'locked.json')),(P/'experiments/region_pose/iter_003',read(P/'experiments/region_pose/iter_003/completed_locked.json')),(P/'experiments/region_pose/iter_013',read(P/'experiments/region_pose/iter_013/completed_locked.json'))]
checks=[]
for p in sorted(paths):
 if p.name.endswith('locked.json') or p.name=='stage2_episode_split.json': continue
 matches=[]
 for base,lock in oldlocks:
  for key,h in lock.items():
   q=Path(key) if Path(key).is_absolute() else base/key
   if q==p: matches.append(h)
 assert matches and all(h==sha(p) for h in matches),str(p)
 checks.append(rel(p))
write(O/'audits/prior_input_lock_check.json',{'checked_files':checks,'passed':True,'only_selected_training_files_hashed':True})
rows=[]; endpoints=[]; missing=[]; errors=[]
for ep in train:
 motion_path=P/f'experiments/region_pose/iter_003/evidence/motion/{ep}.json'
 m=read(motion_path);assert m['split']=='train'
 fusions=[]
 for p in sorted((P/'experiments/region_pose/iter_003/evidence/fusion').glob(ep+'_attempt*.json')):
  f=read(p)
  fusions.append({'path':rel(p),'keys':list(f),'status_fields':{k:v for k,v in f.items() if k in ['held','valid','status','reason','event','attempt','reference_frame']}})
 row={'episode':ep,'current_supervision':ep in current,'motion_source':rel(motion_path),'fusion_files':fusions}
 if ep not in cases:
  row['group']='no_011_held_case'
  row['attempts']=m['attempts']
  diag=m['frame_diagnostics']
  row['saved_motion_statistics']={
   'frames':len(diag),
   'max_observed_finite_voxels':max((x.get('finite_finger_region_observed_voxels',0) for x in diag),default=0),
   'frames_at_least_3_finite_voxels':sum(x.get('finite_finger_region_observed_voxels',0)>=3 for x in diag),
   'max_hand_frame_object_points':max((x.get('hand_frame_object_points',0) for x in diag),default=0)}
  missing.append(ep);rows.append(row);continue
 r=cases[ep];key=r['event']+'_query'+str(r['query'])
 row.update(event=r['event'],query=r['query'],fixed_endpoints=r['selected_held_frames'],
  group='early_correspondence_pass' if r['valid'] else 'early_correspondence_not_passed',
  recorded_reason=r.get('reason'),initial_corners=r.get('initial_corners',r.get('corners')),
  dual_bridge_difference=r.get('dual_bridge_difference'),time_comparisons=r.get('time_comparisons'))
 tpath=D/'trajectories'/(key+'.npz');cpath=D/'correspondences'/(key+'.npz')
 if not tpath.exists():
  row['trajectory_unavailable']='not_run_after_initial_corner_gate'
  assert row['initial_corners']<8
  row['endpoint_statistics']=[];rows.append(row);continue
 with np.load(tpath,allow_pickle=False) as z: tr={k:z[k] for k in z.files}
 with np.load(cpath,allow_pickle=False) as z: co={k:z[k] for k in z.files}
 initial=tr['initial_uv'];assert len(initial)==row['initial_corners']
 row['initial_spatial']=spatial(initial)
 row['query_cloud_spatial']=spatial(co['query_foreground_xyz'])
 row['trajectory_frames']=len(tr['frames'])
 row['visibility_is_per_frame_not_cumulative_survival']=True
 frame_to_i={int(f):i for i,f in enumerate(tr['frames'])}
 trace={int(x['frame']):x for x in r.get('trace',[])}
 row['endpoint_statistics']=[]
 for b in r.get('bridge_results',[]):
  frame=int(b['frame']);j=frame_to_i[frame];suf='_'+str(frame)
  uv=co['query_uv'+suf];ids=co['track_id'+suf]
  mask=co['inlier'+suf]
  n=len(ids);vis=int(tr['visible'][j].sum())
  assert n==b['endpoint_matches'] and len(uv)==n and len(mask)==n
  assert np.all(ids>=0) and np.all(ids<len(initial))
  assert np.allclose(uv,initial[ids],atol=1e-4)
  if 'inliers' in b: assert int(mask.sum())==b['inliers']
  if frame in trace: assert vis==trace[frame]['model_visible']
  er={'episode':ep,'event':r['event'],'frame':frame,'query':r['query'],
   'initial_corners':len(initial),'visible':vis,
   'trace_surviving_tracks':trace.get(frame,{}).get('surviving_tracks'),
   'tracked_before_endpoint_filter':b.get('tracked_before_endpoint_filter'),
   'endpoint_matches':n,'combined_endpoint_filter_removed':b.get('tracked_before_endpoint_filter',n)-n,
   'appearance_removed':None,'robot_removed':None,'depth_removed':None,
   'individual_filter_counts_unavailable':'not_saved_in_locked_evidence',
   'rigid_inliers':b.get('inliers'),'rigid_attempted':n>=8,
   'stored_inlier_mask_sum':int(mask.sum()),
   'ratio':b.get('ratio'),'median_inlier_mm':b.get('median_inlier_mm'),
   'second_axis_scale_mm':b.get('second_axis_scale_mm'),
   'endpoint_valid':b['valid'],'recorded_reason':b.get('reason'),
   'depth_neighborhood':b.get('depth_neighborhood'),
   'matches_spatial':spatial(uv),'matches_xyz_spatial':spatial(co['query_xyz'+suf]),
   'inliers_spatial':spatial(uv[mask]) if 'inliers' in b else None}
  area=row['initial_spatial']['hull_area_px2']
  er['match_hull_fraction_of_initial_hull']=er['matches_spatial']['hull_area_px2']/area if area else None
  er['inlier_hull_fraction_of_initial_hull']=er['inliers_spatial']['hull_area_px2']/area if area and er['inliers_spatial'] else None
  er['xyz_rms_fraction_of_saved_query_cloud']=er['matches_xyz_spatial'].get('rms_radius_m',0)/row['query_cloud_spatial']['rms_radius_m'] if n and row['query_cloud_spatial']['rms_radius_m'] else None
  row['endpoint_statistics'].append(er);endpoints.append(er)
 rows.append(row)
counts=dict(collections.Counter(r['group'] for r in rows))
passed={r['episode'] for r in rows if r['group']=='early_correspondence_pass'}
assert passed==current
assert counts=={'no_011_held_case':4,'early_correspondence_not_passed':10,'early_correspondence_pass':6}
failed=[r for r in rows if r['group']=='early_correspondence_not_passed']
early=[]
for r in rows:
 for er in r.get('endpoint_statistics',[])[:2]:
  early.append(er)
summary={'unit':'20_original_training_episodes','counts':counts,'current_supervision_matches_passed_cases':True,
 'missing_held_cases':missing,'zero_initial_corners':[r['episode'] for r in failed if r['initial_corners']==0],
 'endpoint_observations':len(endpoints),'endpoint_valid_count':sum(x['endpoint_valid'] for x in endpoints),
 'endpoint_less_than_8_matches':sum(x['endpoint_matches']<8 for x in endpoints),
 'endpoint_fitted_but_invalid':sum(x['rigid_attempted'] and not x['endpoint_valid'] for x in endpoints),
 'first_two_endpoint_observations':len(early),
 'first_two_pass_count':sum(x['endpoint_valid'] for x in early),
 'distinct_failed_episodes_with_both_early_endpoints_valid':[r['episode'] for r in failed if len(r.get('endpoint_statistics',[]))>=2 and all(e['endpoint_valid'] for e in r['endpoint_statistics'][:2])],
 'filter_attribution_available':False,'rows':rows,'new_labels':False,'tracking_inference':False,'training':False,
 'limits':['endpoint observations are not independent grasps','hull ratios are relative to initial corners, not object surface coverage','stored zero masks before fit are not measured zero rigid inliers','no physical identity truth or success-rate claim']}
write(O/'summary.json',finite(summary));write(O/'endpoints.json',finite(endpoints))
write(O/'audits/completion_audit.json',{'source_hashes_unchanged':all(sha(P/k)==h for k,h in sources.items()),
 'input_hashes_unchanged':all(sha(P/k)==h for k,h in inputs.items()),'input_files':len(inputs),
 'prior_locked_inputs_checked':len(checks),'episodes':len(rows),'endpoint_array_counts_and_track_ids_checked':len(endpoints),
 'current_supervision_set_equal':True,'validation_or_test_samples_read':False,'raw_rgbd_read':False,
 'unavailable_filter_fields_left_null':True,'passed':True})
write(O/'stage_state.json',{'stage':'audit_complete','time':time.time(),'training':False})
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False))
