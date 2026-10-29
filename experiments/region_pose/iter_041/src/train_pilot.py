import json, hashlib, datetime, traceback
from pathlib import Path
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA'); B=R/'experiments/region_pose'; D=B/'iter_041'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def runs(mask):
 out=[];start=None
 for i,b in enumerate(list(mask)+[False]):
  if b and start is None:start=i
  if not b and start is not None:out.append([start,i-1,i-start]);start=None
 return out
def independent_runs(mask):
 ids=np.flatnonzero(mask)
 if not len(ids):return []
 groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
 return [[int(g[0]),int(g[-1]),len(g)] for g in groups]
def select(motion,voxel,cfg):
 out=[]
 for a,b,n in runs(motion):
  if n<cfg['motion_min_run']:continue
  supported=[i for i in range(a,b+1) if voxel[i]>=cfg['region_min_voxels']]
  if len(supported)<2 or supported[-1]-supported[0]<cfg['endpoint_min_separation']:continue
  first=supported[0];endpoints=[]
  for off in cfg['endpoint_offsets']:
   options=[i for i in supported if i>=first+off]
   if options:endpoints.append(options[0])
  endpoints=sorted(set(endpoints+[supported[-1]]))
  out.append({'motion_start':a,'motion_end':b,'motion_frames':n,'observed_region_frames':supported,'endpoint_frames':endpoints,'unobserved_region_frames':[i for i in range(a,b+1) if i not in supported],'candidate_only':True})
 return out
def main():
 cfg=json.loads((D/'config.json').read_text())
 # Meaningful behavioral fixtures: intermittent visibility, broken motion, insufficient duration and pre-attempt isolation.
 assert select([1]*5,[3,0,3,0,0],cfg)
 assert not select([1]*4,[3,0,3,0],cfg)
 assert not select([1]*5,[3,0,0,0,0],cfg)
 assert not select([1,1,0,1,1],[3]*5,cfg)
 assert not select([0,0,1,1,1,1,1],[3,3,0,0,0,0,0],cfg)
 assert runs([0,1,1,0,1])==independent_runs([0,1,1,0,1])
 source=json.loads((B/'iter_033/summary.json').read_text())
 inputs=[B/'iter_033/summary.json',B/'iter_033/completed_locked.json']
 for ep in source['episodes']:
  name=ep['episode'];inputs += [B/f'iter_033/cases/{name}.npz',B/f'iter_003/evidence/motion/{name}.json']
 before={str(p.relative_to(R)):sha(p) for p in inputs}
 save(D/'input_locked.json',before)
 oldlock=json.loads((B/'iter_033/completed_locked.json').read_text())
 for path,h in oldlock.items():
  p=Path(path);p=p if p.is_absolute() else R/p
  expected=h if isinstance(h,str) else h['sha256']
  assert sha(p)==expected
 rows=[];perframe=[];old_equal=True
 for ep in source['episodes']:
  name=ep['episode'];assert ep['split']=='train'
  m=json.loads((B/f'iter_003/evidence/motion/{name}.json').read_text());assert m['split']=='train'
  with np.load(B/f'iter_033/cases/{name}.npz') as z:
   v=z['full_voxels'];H=z['hand_to_world'];n=len(v)
  frames=m['frame_diagnostics'];assert len(frames)==n
  gap=np.array([x.get('pad_gap_mm') for x in frames],float)
  cloud=np.array([x.get('hand_frame_object_points') for x in frames],float)
  overlap=np.array([x.get('previous_frame_surface_overlap') for x in frames],float)
  for attempt,old in zip(m['attempts'],ep['attempts']):
   a=attempt['start_frame'];b=attempt['end_frame'];assert (a,b)==(old['start'],old['end'])
   ids=np.arange(n);lift=H[:,2,3]-H[a,2,3]
   inside=(ids>=a)&(ids<=b)
   motion=inside&np.isfinite(gap)&(gap>0)&np.isfinite(cloud)&(cloud>=20)&np.isfinite(overlap)&(overlap>=.5)&np.isfinite(lift)&(lift>.025)
   original=motion&(v>=3);old_frames=np.flatnonzero(original).tolist()
   assert old_frames==old['full_intersection']['frames']
   assert runs(motion)==independent_runs(motion)
   original_runs=[z for z in runs(original) if z[2]>=5]
   original_accepted=[i for start,end,_ in original_runs for i in range(start,end+1)]
   assert original_accepted==old['candidate_accepted']
   candidates=select(motion,v,cfg)
   # Every old complete interval is contained by a new motion candidate.
   for start,end,_ in original_runs:
    assert any(x['motion_start']<=start and x['motion_end']>=end for x in candidates)
   row={'episode':name,'attempt':attempt['attempt'],'attempt_start':a,'attempt_end':b,'old_full_candidate':bool(original_runs),'old_full_runs':original_runs,'motion_runs':runs(motion),'new_candidate':bool(candidates),'candidates':candidates,'old_intersection_frames':old_frames,'not_success_ground_truth':True}
   rows.append(row)
   for i in range(a,b+1):
    perframe.append({'episode':name,'attempt':attempt['attempt'],'frame':i,'gap_mm':float(gap[i]) if np.isfinite(gap[i]) else None,'cloud_points':float(cloud[i]) if np.isfinite(cloud[i]) else None,'overlap':float(overlap[i]) if np.isfinite(overlap[i]) else None,'lift_mm':float(lift[i]*1000),'full_voxels':int(v[i]),'motion_ok':bool(motion[i]),'old_full_intersection':bool(original[i])})
  save(D/f'train_pilot/{name}.json',[r for r in rows if r['episode']==name])
 assert len(rows)==6
 empty=[r for r in rows if r['episode']=='tomato_sauce_demo_3' and r['attempt']==1]
 assert len(empty)==1
 summary={'training_episodes':5,'attempts':6,'old_candidate_attempts':sum(r['old_full_candidate'] for r in rows),'new_candidate_attempts':sum(r['new_candidate'] for r in rows),'newly_candidate_attempts':sum(r['new_candidate'] and not r['old_full_candidate'] for r in rows),'historical_empty_control_promoted':empty[0]['new_candidate'],'rows':rows,'new_labels':False,'new_training':False,'validation_read':False,'test_read':False,'not_success_rate':True}
 save(D/'train_pilot/summary.json',summary);save(D/'train_pilot/frame_conditions.json',perframe)
 assert all(sha(R/p)==h for p,h in before.items())
 save(D/'audits/train_pilot_audit.json',{'old_full_intersection_exact':True,'old_accepted_exact':True,'old_candidates_preserved':True,'independent_runs_equal':True,'behavioral_fixtures_pass':True,'input_hashes_unchanged':len(before),'prior_033_lock_files_verified':len(oldlock),'new_labels':False})
 save(D/'train_pilot_locked.json',{str(p.relative_to(R)):sha(p) for p in (D/'train_pilot').glob('*.json')})
 save(D/'stage_state.json',{'stage':'train_pilot_complete','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'next':'extend full-volume extraction to remaining 15 original training episodes, then apply frozen split evidence rule; validation not yet read'})
 print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(D/'failure.json',{'type':type(e).__name__,'traceback':traceback.format_exc()})
  print(json.dumps({'failure_type':type(e).__name__}))
  raise SystemExit(1)
