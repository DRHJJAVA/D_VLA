"""Experimental RGB-D-only stable near-surface interaction cue, train/val only.
No action, contact, object truth, names, or held-out episode is read.
This is candidate visual evidence, not a grasp label or calibrated prediction.
"""
from pathlib import Path
import json,math
import h5py,numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
COMP=R/'data/v3/proposals/grabcut_components/train_val_448'
BASE=R/'data/v3/samples/visual_interaction_uniform448'
OUT=R/'data/v3/samples/stable_near_surface_cue_v1'
def world(px,py,z,K,T):
 cam=np.stack(((px-K[0,2])*z/K[0,0],(py-K[1,2])*z/K[1,1],z),1)
 return cam@T[:3,:3].T+T[:3,3]
def one(split,id):
 with np.load(BASE/split/(id+'.npz')) as b:
  xyz=b['xyz'].copy();pixel=b['pixel'].copy();onsets=b['onsets'].copy()
  legacy=b['target'].copy()
 with np.load(COMP/split/(id+'.npz')) as c:
  path=Path(str(c['rgbd_path']));outer=c['outer_box'].astype(int)
 with h5py.File(path) as f:
  depth=f['agentview/depth_m'];K=f['agentview/K'];T=f['agentview/camera_to_world']
  first=depth[0]
  x1,y1,x2,y2=outer
  xlo=max(0,x1-20);ylo=max(0,y1-20)
  xhi=min(first.shape[1],x2+20);yhi=min(first.shape[0],y2+20)
  event_cues=[];event_rows=[]
  for event in onsets:
   frame_dist=[];point_counts=[]
   for frame in range(max(0,int(event)-3),int(event)):
    now=depth[frame,ylo:yhi:2,xlo:xhi:2]
    old=first[ylo:yhi:2,xlo:xhi:2]
    sy,sx=np.nonzero((now<old-.02)&(now>.05))
    if not len(sx):
     frame_dist.append(np.full(len(xyz),.2));point_counts.append(0);continue
    pu=xlo+2*sx;pv=ylo+2*sy;zz=now[sy,sx]
    moving=world(pu,pv,zz,K[frame],T[frame])
    near=cKDTree(xyz).query(moving)[0]<.05
    moving=moving[near]
    point_counts.append(int(len(moving)))
    frame_dist.append(cKDTree(moving).query(xyz)[0] if len(moving) else np.full(len(xyz),.2))
   if not frame_dist:continue
   stable=np.max(np.stack(frame_dist),axis=0)
   evidence=np.exp(-np.minimum(stable,.2)/.012).astype(np.float32)
   event_cues.append(evidence)
   event_rows.append(dict(event=int(event),frames=list(range(max(0,int(event)-3),int(event))),moving_point_counts=point_counts))
  if not event_cues:raise ValueError(id)
  cue=np.max(np.stack(event_cues),axis=0)
  dest=OUT/split/(id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():
   with np.load(dest) as z:assert np.allclose(z['cue'],cue,atol=1e-7,rtol=0)
  else:
   temp=dest.with_suffix('.partial.npz')
   if temp.exists():raise RuntimeError(str(temp))
   np.savez_compressed(temp,cue=cue,pixel=pixel,xyz=xyz,onsets=onsets,
    legacy_visual_target=legacy,query_future_training_evidence=np.asarray(True),
    no_simulator_truth=np.asarray(True))
   temp.replace(dest)
  order=np.argsort(cue)[::-1]
  k=math.ceil(.2*len(cue))
  mass=float(cue[order[:k]].sum()/max(cue.sum(),1e-9))
  oldmass=float(legacy[order[:k]].sum()/max(legacy.sum(),1e-9))
  return dict(id=id,split=split,points=len(cue),events=event_rows,
   cue_mean=float(cue.mean()),cue_max=float(cue.max()),
   cue_effective_mass=float(cue.sum()),top20_own_mass=mass,
   top20_legacy_mass=oldmass,
   legacy_mean=float(legacy.mean()))
def main():
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   row=one(split,id);rows.append(row);print(json.dumps(row),flush=True)
 p=OUT/'manifest.json'
 record=dict(status='stable_near_surface_cue_v1_complete',rows=rows,
  train=len(S['train']),val=len(S['val']),
  definition='max across events of exp(-max last-three-frame near-moving-surface distances / 12mm); moving RGB-D surfaces restricted to within 50mm of first-frame object surface',
  experimental_only=True,no_simulator_truth=True,no_actions=True,
  source='complete RGB-D train/val sequences',
  caveat='may still include robot occlusion; must visually inspect before learning')
 if p.exists():assert json.loads(p.read_text())['status']==record['status']
 else:p.write_text(json.dumps(record,indent=2))
 print(json.dumps(dict(status='complete',count=len(rows))),flush=True)
if __name__=='__main__':main()
