"""Rebuild visual-only interaction targets on uniform 448 RGB-D first-frame objects.
The complete TRAIN/VAL RGB-D sequences supply automatic evidence; no simulator truth,
robot actions, names, or hand-labeled regions are read. Validation target is audit-only."""
from pathlib import Path
import json,math,sys
import h5py,numpy as np,torch
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_extract_dino import model,preproc
from stage2_local_teacher import crop_box
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
E={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
COMP=R/'data/v3/proposals/grabcut_components/train_val_448'
OUT=R/'data/v3/samples/visual_interaction_uniform448'
def world(px,py,z,K,T):
 cam=np.stack(((px-K[0,2])*z/K[0,0],(py-K[1,2])*z/K[1,1],z),1)
 return cam@T[:3,:3].T+T[:3,3]
def build(split,id,net):
 e=E[id]
 with np.load(COMP/split/(id+'.npz')) as f:
  pixel=f['pixel'].copy();source_box=f['component_box'].copy();outer=f['outer_box'].copy();path=Path(str(f['rgbd_path']))
 with h5py.File(path) as f:
  depth=f['agentview/depth_m'][:];rgb=f['agentview/rgb'][0]
  K=f['agentview/K'][:];T=f['agentview/camera_to_world'][:]
 first=depth[0];px=pixel[:,0];py=pixel[:,1];z=first[py,px]
 valid=(z>.05)&np.isfinite(z);px=px[valid];py=py[valid];z=z[valid];pixel=pixel[valid]
 if len(z)<64:raise ValueError('too few visual foreground points: '+id)
 xyz=world(px,py,z,K[0],T[0])
 cb=crop_box(outer.tolist(),*first.shape)
 cx1,cy1,cx2,cy2=int(math.floor(cb[0])),int(math.floor(cb[1])),int(math.ceil(cb[2])),int(math.ceil(cb[3]))
 with torch.inference_mode():
  patch=net.forward_features(preproc(rgb[None,cy1:cy2,cx1:cx2]))[:,1:,:].reshape(32,32,384).cpu().numpy()
 gx=np.clip(((px-cx1)*32/max(1,cx2-cx1)).astype(int),0,31)
 gy=np.clip(((py-cy1)*32/max(1,cy2-cy1)).astype(int),0,31)
 dino=patch[gy,gx]
 removed=(depth[:,py,px]>z[None,:]+.025).mean(1)
 onsets=[i for i in range(1,len(removed)-2) if removed[i-1]<=.2 and np.all(removed[i:i+3]>.2)]
 if not onsets:raise ValueError('no visual removal event: '+id)
 x1,y1,x2,y2=outer
 xlo=max(0,x1-30);ylo=max(0,y1-30);xhi=min(first.shape[1],x2+30);yhi=min(first.shape[0],y2+30)
 scores=[]
 for event in onsets:
  nearest=np.full(len(xyz),np.inf)
  for frame in range(max(0,event-8),event):
   now=depth[frame,ylo:yhi:2,xlo:xhi:2]
   old=first[ylo:yhi:2,xlo:xhi:2]
   sy,sx=np.nonzero((now<old-.02)&(now>.05))
   if not len(sx):continue
   pu=xlo+2*sx;pv=ylo+2*sy;zz=now[sy,sx]
   moving=world(pu,pv,zz,K[frame],T[frame])
   nearest=np.minimum(nearest,cKDTree(moving).query(xyz)[0])
  if np.isfinite(nearest).any():scores.append(nearest)
 if not scores:raise ValueError('no pre-event moving surface: '+id)
 distance=np.min(np.stack(scores),axis=0)
 target=np.exp(-np.minimum(distance,.2)/.02).astype(np.float32)
 dest=OUT/split/(id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
 tmp=dest.with_suffix('.partial.npz')
 if tmp.exists():raise RuntimeError('inspect partial file: '+str(tmp))
 np.savez_compressed(tmp,xyz=xyz.astype(np.float32),dino_feat=dino.astype(np.float32),rgb=rgb[py,px],pixel=pixel.astype(np.int16),target=target,depth=z.astype(np.float32),distance=distance.astype(np.float32),onsets=np.asarray(onsets,dtype=np.int32),source_box=source_box,outer_box=outer,first_frame_only=np.asarray(True))
 tmp.replace(dest)
 return dict(id=id,split=split,points=len(xyz),onsets=onsets,target_mean=float(target.mean()),target_max=float(target.max()),source=str(path))
def main():
 assert json.loads((COMP/'manifest.json').read_text())['status']=='uniform_448_train_val_grabcut_complete'
 torch.set_num_threads(4)
 net=None;rows=[]
 for split in ('train','val'):
  for id in S[split]:
   dest=OUT/split/(id+'.npz')
   if dest.exists():
    with np.load(dest) as f:assert all(k in f for k in ('xyz','dino_feat','target','onsets'))
    row=dict(id=id,split=split,status='reused');rows.append(row);print(json.dumps(row),flush=True);continue
   if net is None:net=model()
   row=build(split,id,net);rows.append(row);print(json.dumps(row),flush=True)
 manifest=OUT/'manifest.json'
 data=dict(status='visual_interaction_uniform448_complete',train=len(S['train']),val=len(S['val']),no_simulator_truth=True,no_names=True,no_robot_actions=True,rows=rows)
 if manifest.exists():assert json.loads(manifest.read_text())['status']==data['status']
 else:manifest.write_text(json.dumps(data,indent=2))
 print(json.dumps(dict(status='complete',train=len(S['train']),val=len(S['val']))),flush=True)
if __name__=='__main__':main()
