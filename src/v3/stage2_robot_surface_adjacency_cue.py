"""Exploratory robot-object RGB-D adjacency cue on TRAIN/VAL full sequences.
Use trained self-supervised moving-manipulator separator at pre-lift frame,
exclude first-frame object points still depth-visible, then retain predicted
robot surface newly in front of the first frame and physically near object.
No simulator contacts/truth, actions, names, manual labels, or held-out data.
This is not a grasp label; inspect physically before any student training.
"""
from pathlib import Path
import json,sys,math
import cv2,h5py,numpy as np,torch
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_train_preinteraction_robot_separator import SmallUNet
from stage2_audit_robot_separator_contact_frames import infer
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
SRC=R/'data/v3/samples/visual_interaction_uniform448_3dcc'
COMP=R/'data/v3/proposals/foreground3d_components'
MODEL=R/'outputs/v3_stage2/preinteraction_robot_seg_v1'
OUT=R/'data/v3/samples/robot_surface_adjacency_cue_v1'
def world(px,py,z,K,T):
 cam=np.stack(((px-K[0,2])*z/K[0,0],(py-K[1,2])*z/K[1,1],z),1)
 return cam@T[:3,:3].T+T[:3,3]
def one(split,id,net,device):
 with np.load(SRC/split/(id+'.npz')) as z:
  xyz=z['xyz'].copy();pixel=z['pixel'].copy();onsets=z['onsets'].copy()
 with np.load(COMP/split/(id+'.npz')) as z:
  path=Path(str(z['rgbd_path']));box=z['outer_box'].astype(int)
 with h5py.File(path) as f:
  first=f['agentview/depth_m'][0]
  masks=[];events=[];scores=[]
  for event in onsets:
   t=int(event)-1
   now=f['agentview/depth_m'][t]
   rgb=f['agentview/rgb'][t]
   K=f['agentview/K'][t];T=f['agentview/camera_to_world'][t]
   prob=infer(net,rgb,now,device)
   robot=prob>.5
   still=(now[pixel[:,1],pixel[:,0]]>.05)&(np.abs(now[pixel[:,1],pixel[:,0]]-first[pixel[:,1],pixel[:,0]])<.015)
   robot[pixel[still,1],pixel[still,0]]=False
   x1,y1,x2,y2=box
   xlo=max(0,x1-40);ylo=max(0,y1-40);xhi=min(first.shape[1],x2+40);yhi=min(first.shape[0],y2+40)
   view=robot[ylo:yhi:2,xlo:xhi:2]
   dnow=now[ylo:yhi:2,xlo:xhi:2]
   d0=first[ylo:yhi:2,xlo:xhi:2]
   yy,xx=np.nonzero(view&(dnow>.05)&(dnow<d0-.012))
   pu=xlo+2*xx;pv=ylo+2*yy;zz=dnow[yy,xx]
   if len(zz):
    moving=world(pu,pv,zz,K,T)
    dist=cKDTree(xyz).query(moving)[0]
    keep=dist<.035
    moving=moving[keep];pu=pu[keep];pv=pv[keep]
   else:moving=np.empty((0,3),np.float32)
   if len(moving):
    near=cKDTree(moving).query(xyz)[0]
    score=np.exp(-np.minimum(near,.2)/.012).astype(np.float32)
   else:score=np.zeros(len(xyz),np.float32)
   scores.append(score)
   events.append(dict(event=int(event),frame=t,
    predicted_robot_pixels=int(robot.sum()),
    visible_first_object_points=int(still.sum()),
    near_robot_surface_points=int(len(moving)),
    mean_cue=float(score.mean()),
    robot_near_pixel=np.stack((pu,pv),axis=1).astype(np.int16).tolist()))
  cue=np.max(np.stack(scores),axis=0)
  dest=OUT/split/(id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():
   with np.load(dest) as old:assert np.allclose(old['cue'],cue,atol=1e-7)
  else:
   temp=dest.with_suffix('.partial.npz')
   if temp.exists():raise RuntimeError(str(temp))
   np.savez_compressed(temp,cue=cue,xyz=xyz,pixel=pixel,onsets=onsets,
    event_cues=np.stack(scores),query_future_training_evidence=np.asarray(True),
    no_simulator_truth=np.asarray(True))
   temp.replace(dest)
  k=max(1,math.ceil(.2*len(cue)));top=np.argpartition(cue,-k)[-k:]
  return dict(id=id,split=split,points=len(xyz),events=events,
   cue_mean=float(cue.mean()),top20_own_mass=float(cue[top].sum()/max(cue.sum(),1e-8)))
def main():
 summary=json.loads((MODEL/'summary.json').read_text())
 assert summary['status']=='auxiliary_robot_separator_pseudo_training_complete'
 device='cuda' if torch.cuda.is_available() else 'cpu'
 state=torch.load(summary['checkpoint'],map_location=device,weights_only=False)
 net=SmallUNet().to(device);net.load_state_dict(state['model']);net.eval()
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   row=one(split,id,net,device);rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='events'}),flush=True)
 manifest=OUT/'manifest.json'
 record=dict(status='robot_surface_adjacency_cue_v1_complete',
  source='TRAIN/VAL complete RGB-D; first-frame object surface and pre-lift visual robot predictions',
  no_simulator_truth=True,no_actions=True,no_names=True,no_manual_labels=True,
  experimental_only=True,model_checkpoint=summary['checkpoint'],
  caveat='near robot surface is not verified contact; still-visible object exclusion and geometry may miss hidden/wrapped areas',
  rows=rows)
 if manifest.exists():
  old=json.loads(manifest.read_text());assert old['status']==record['status']
 else:manifest.write_text(json.dumps(record,indent=2))
 print(json.dumps(dict(status='complete',episodes=len(rows))),flush=True)
if __name__=='__main__':main()
