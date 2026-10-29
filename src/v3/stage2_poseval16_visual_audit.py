"""Visual-only audit after pose-validation first-frame predictions were hash-locked.
Full query RGB-D is used only here to construct independent interaction evidence.
No simulator contacts/object truth, actions, names, or text are opened.
"""
from pathlib import Path
import hashlib,json,sys,math
import h5py,numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_build_visual_interaction_448_3dcc import world
MAN=R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json'
LOCK=R/'outputs/v3_stage2/visual_frequency_poseval16/predictions_locked.json'
PRED=LOCK.parent
SURF=R/'data/v3/proposals/poseval16_surface_candidates'
OUT=R/'audits/v3/stage2_poseval16_visual_audit.json'
TARGETS=R/'audits/v3/stage2_poseval16_visual_targets'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def event_target(path,pixel,outer):
 with h5py.File(path) as f:
  depth=f['agentview/depth_m'][:]
  K=f['agentview/K'][:]
  T=f['agentview/camera_to_world'][:]
 first=depth[0];px=pixel[:,0];py=pixel[:,1];z=first[py,px]
 if not np.all((z>.05)&np.isfinite(z)):raise ValueError('invalid first depth')
 xyz=world(px,py,z,K[0],T[0])
 removed=(depth[:,py,px]>z[None,:]+.025).mean(1)
 onsets=[i for i in range(1,len(removed)-2)
   if removed[i-1]<=.2 and np.all(removed[i:i+3]>.2)]
 if not onsets:return None,onsets,removed
 x1,y1,x2,y2=map(int,outer)
 xlo=max(0,x1-30);ylo=max(0,y1-30)
 xhi=min(first.shape[1],x2+30);yhi=min(first.shape[0],y2+30)
 scores=[]
 for event in onsets:
  nearest=np.full(len(xyz),np.inf)
  for frame in range(max(0,event-8),event):
   now=depth[frame,ylo:yhi:2,xlo:xhi:2]
   old=first[ylo:yhi:2,xlo:xhi:2]
   sy,sx=np.nonzero((now<old-.02)&(now>.05))
   if len(sx)==0:continue
   pu=xlo+2*sx;pv=ylo+2*sy;zz=now[sy,sx]
   moving=world(pu,pv,zz,K[frame],T[frame])
   nearest=np.minimum(nearest,cKDTree(moving).query(xyz)[0])
  if np.isfinite(nearest).any():scores.append(nearest)
 if not scores:return None,onsets,removed
 distance=np.min(np.stack(scores),axis=0)
 target=np.exp(-np.minimum(distance,.2)/.02).astype(np.float32)
 return target,onsets,removed
def budget(score,target,fraction):
 k=max(1,int(math.ceil(len(score)*fraction)))
 top=np.argpartition(score,-k)[-k:]
 return dict(selected=k,random=k/len(score),
   visual_mass=float(target[top].sum()/max(target.sum(),1e-8)))
def main():
 lock=json.loads(LOCK.read_text())
 assert lock['status']=='poseval16_multicandidate_visual_frequency_locked'
 assert lock['no_query_future'] and lock['no_simulator_truth']
 entries=json.loads(MAN.read_text())['rows'];assert len(entries)==16
 rows=[]
 for e in entries:
  id=e['id'];pred=PRED/'rank_1'/(id+'.npz')
  assert sha(pred)==lock['prediction_sha256']['rank_1/'+id]
  with np.load(pred) as f:
   score=f['score'].copy();pixel=f['pixel'].copy()
   assert int(f['query_frame'])==0 and not bool(f['has_future'])
  with np.load(SURF/id/'rank_1.npz') as f:outer=f['outer_box'].copy()
  target,onsets,removed=event_target(e['rgbd_path'],pixel,outer)
  row=dict(id=id,points=len(pixel),onsets=onsets,
    max_removed_fraction=float(removed.max()),final_removed_fraction=float(removed[-1]))
  if target is None:
   row['status']='no_stable_visual_event'
  else:
   row['status']='visual_event_auditable'
   row['target_mean']=float(target.mean())
   row['target_mass']=float(target.sum())
   row['budget_10']=budget(score,target,.1)
   row['budget_20']=budget(score,target,.2)
   row['budget_30']=budget(score,target,.3)
   dest=TARGETS/(id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():
    with np.load(dest) as old:assert np.allclose(old['target'],target,rtol=0,atol=1e-7)
   else:
    tmp=dest.with_suffix('.partial.npz')
    if tmp.exists():raise RuntimeError('inspect partial: '+str(tmp))
    np.savez_compressed(tmp,target=target,pixel=pixel,onsets=np.asarray(onsets),
      prediction_sha256=np.asarray(sha(pred)))
    tmp.replace(dest)
  rows.append(row);print(json.dumps(row),flush=True)
 valid=[x for x in rows if x['status']=='visual_event_auditable']
 summary=dict(total=len(rows),auditable=len(valid),
   mean_top10_visual_mass=float(np.mean([x['budget_10']['visual_mass'] for x in valid])) if valid else None,
   mean_random10=float(np.mean([x['budget_10']['random'] for x in valid])) if valid else None,
   mean_top20_visual_mass=float(np.mean([x['budget_20']['visual_mass'] for x in valid])) if valid else None,
   mean_random20=float(np.mean([x['budget_20']['random'] for x in valid])) if valid else None,
   mean_top30_visual_mass=float(np.mean([x['budget_30']['visual_mass'] for x in valid])) if valid else None,
   mean_random30=float(np.mean([x['budget_30']['random'] for x in valid])) if valid else None)
 report=dict(status='pose_validation_visual_only_after_lock',reference='automatic full-sequence RGB-D interaction cue, not grasp truth',
  predictions_locked_sha256=sha(LOCK),no_simulator_truth=True,validation_future_audit_only=True,
  summary=summary,rows=rows)
 if OUT.exists():raise FileExistsError(OUT)
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
