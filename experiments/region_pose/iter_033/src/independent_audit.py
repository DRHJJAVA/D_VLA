from pathlib import Path
import numpy as np,h5py,json,hashlib,datetime,traceback
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';R=B/'iter_033'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def intervals(ids):
 groups=[]
 for i in ids:
  if not groups or i!=groups[-1][-1]+1:groups.append([int(i)])
  else:groups[-1].append(int(i))
 return groups
def countstat(x):return {'count':len(x),'longest':max(map(len,intervals(x)),default=0)}
try:
 assert json.loads((R/'stage_state.json').read_text())['stage']=='computed_pending_audit_report'
 source=json.loads((R/'input_locked.json').read_text())
 oldlock=json.loads((B/'iter_003/completed_locked.json').read_text())
 oldchecked=[]
 for row in json.loads((R/'cases.json').read_text()):
  for ext in ['json','npz']:
   rel='evidence/motion/'+row['id']+'.'+ext
   p=B/'iter_003'/rel
   assert sha(p)==oldlock[rel];oldchecked.append(rel)
 rows=[];point_checks=0;max_reproject=0.;max_depth=0.
 for row in json.loads((R/'cases.json').read_text()):
  ep=row['id'];z=np.load(R/'cases'/(ep+'.npz'));j=json.loads((R/'cases'/(ep+'.json')).read_text())
  old=json.loads((B/'iter_003/evidence/motion'/(ep+'.json')).read_text())
  n=len(z['hand_to_world']);offs=z['frame_offsets'];assert len(offs)==n+1 and offs[0]==offs[1]==0
  reps=z['representative_indices'];H=z['hand_to_world'];q=z['measured_finger_q'];nw=z['narrow_voxels'];full=z['full_voxels'];strict=z['full_012_strict_voxels']
  assert np.all(nw<=full) and np.all(strict<=full)
  assert len(reps)==int(full.sum())
  with h5py.File(row['rgbd'],'r') as f:
   for t in range(1,n):
    lo,hi=offs[t:t+2];p=z['points_world'][lo:hi];uv=z['pixel'][lo:hi];cam=z['camera'][lo:hi]
    hand=(p-H[t,:3,3])@H[t,:3,:3]
    valid=(abs(hand[:,0])<=.008)&(hand[:,2]>=.0854)&(hand[:,2]<=.1014)&(hand[:,1]>=q[t,1]+.0005)&(hand[:,1]<=q[t,0]-.0005)
    assert np.all(valid) and np.all(np.isfinite(p))
    unique,ki=np.unique(np.rint(hand/.001).astype(int),axis=0,return_index=True)
    actual=reps[(reps>=lo)&(reps<hi)]
    assert np.array_equal(actual,lo+ki) and len(actual)==full[t]
    for ci,cn in enumerate(['agentview','robot0_eye_in_hand']):
     use=cam==ci
     if not use.any():continue
     c=f[cn];K=c['K'][t];T=c['camera_to_world'][t];dep=c['depth_m'][t]
     cp=(p[use]-T[:3,3])@T[:3,:3]
     depvals=dep[uv[use,1],uv[use,0]]
     # Cloud convention audited from actual reprojection, with absolute depth.
     projected=(cp@K.T);projected=projected[:,:2]/projected[:,2:]
     err=float(np.max(np.abs(projected-uv[use])));derr=float(np.max(np.abs(cp[:,2]-depvals)))
     max_reproject=max(max_reproject,err);max_depth=max(max_depth,derr)
     assert err<1e-6 and derr<1e-8
     point_checks+=int(use.sum())
  attempts=[]
  for a in j['attempts']:
   start,end=a['start'],a['end'];ids=np.arange(start,end+1);diag=old['frame_diagnostics'];gap=q[:,0]-q[:,1]-.001
   gates=np.array([bool(gap[i]>0 and diag[i]['hand_frame_object_points']>=20 and diag[i]['previous_frame_surface_overlap'] is not None and diag[i]['previous_frame_surface_overlap']>=.5 and H[i,2,3]-H[start,2,3]>.025) for i in ids])
   nf=ids[gates&(nw[ids]>=3)].tolist();bf=ids[gates&(full[ids]>=3)].tolist();sf=ids[gates&(strict[ids]>=3)].tolist()
   assert nf==a['old_intersection']['frames'] and bf==a['full_intersection']['frames']
   accepted=[i for g in intervals(bf) if len(g)>=5 for i in g]
   assert accepted==a['candidate_accepted']
   sa=[i for g in intervals(sf) if len(g)>=5 for i in g]
   attempts.append({'attempt':a['attempt'],'start':start,'end':end,'old_accepted':len(a['old_accepted']),'full_candidate_accepted':len(accepted),'strict012_candidate_accepted':len(sa),'old_intersection':countstat(nf),'full_intersection':countstat(bf),'strict012_intersection':countstat(sf),'full_runs':[[g[0],g[-1],len(g)] for g in intervals(bf)],'strict012_runs':[[g[0],g[-1],len(g)] for g in intervals(sf)],'old_max_voxels':int(nw[ids].max()),'full_max_voxels':int(full[ids].max())})
  rows.append({'episode':ep,'frames':n,'points':len(z['points_world']),'representatives':len(reps),'robot_filter_differing_frames':int(np.sum(full!=strict)),'narrow_full_differing_frames':int(np.sum(full!=nw)),'attempts':attempts})
 assert all(sha(Path(k))==v for k,v in source.items())
 save(R/'audits/independent_reconstruction.json',{'episodes':rows,'old003_locked_files_checked':oldchecked,'source_input_unchanged':len(source),'real_source_points_reprojected':point_checks,'max_pixel_reprojection_error':max_reproject,'max_depth_error_m':max_depth,'frame0_unknown':True,'all_representatives_real_source_indices':True,'original_robot_filter_vs_012_separate':True,'new_labels':False,'validation_read':False,'test_read':False})
 print(json.dumps({'independent_audit_complete':True,'episodes':rows,'points':point_checks,'max_pixel_error':max_reproject,'max_depth_error_m':max_depth}),flush=True)
except Exception as e:
 save(R/'audit_failure.json',{'type':type(e).__name__,'message':str(e)})
 (R/'audit_traceback_remote.log').write_text(traceback.format_exc());print(json.dumps({'failed':True,'type':type(e).__name__,'message':str(e)}));raise SystemExit(1)
