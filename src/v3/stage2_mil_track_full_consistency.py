"""Full-frame visual self-consistency of automatic object tracks; TRAIN/VAL only."""
from pathlib import Path
import json,time
import cv2,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
E={x['id']:x for x in (json.loads(line) for line in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines())}
COMP=R/'data/v3/proposals/foreground3d_components'
TRACK=R/'data/v3/tracking/mil_448_v1'
FEATURE=R/'data/v3/features/dinov2_all_frames'
OUT=R/'audits/v3/stage2_mil_track_full_consistency'
def hist(rgb,mask=None):
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
 a=hsv[mask] if mask is not None else hsv.reshape(-1,3)
 h=np.histogramdd(a,bins=(12,4,4),range=((0,180),(0,256),(0,256)))[0].ravel()+1e-5
 return h/h.sum()
def one(split,id):
 with np.load(COMP/split/(id+'.npz')) as z:pix=z['pixel'];rgbd=Path(str(z['rgbd_path']))
 with np.load(TRACK/split/(id+'.npz')) as z:boxes=z['boxes'];ok=z['ok']
 e=E[id];fp=FEATURE/e['asset_group']/(Path(e['rgbd_path']).stem+'.h5')
 target=OUT/split/(id+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  with np.load(target) as z:
   assert len(z['color_hellinger'])==len(boxes) and len(z['dino_cosine'])==len(boxes)
   assert str(z['rgbd_path'])==str(rgbd)
  return dict(split=split,id=id,status='reused',frames=len(boxes))
 start=time.time()
 with h5py.File(rgbd) as f,h5py.File(fp) as g:
  rgb=f['agentview/rgb'];depth=f['agentview/depth_m'];patch=g['agentview/patch'];n=len(boxes)
  assert len(rgb)==n and len(patch)==n
  base=rgb[0];mask=np.zeros(base.shape[:2],bool);mask[pix[:,1],pix[:,0]]=True
  refhist=hist(base,mask)
  grid=np.zeros((32,32),bool)
  grid[np.clip((pix[:,1]/base.shape[0]*32).astype(int),0,31),np.clip((pix[:,0]/base.shape[1]*32).astype(int),0,31)]=True
  ref=patch[0][grid].astype(np.float32).mean(0);ref/=np.linalg.norm(ref)+1e-9
  colors=np.zeros(n,np.float32);dinos=np.zeros(n,np.float32);depths=np.zeros(n,np.float32);visible=np.zeros(n,np.float32)
  for t in range(n):
   x,y,w,h=map(int,boxes[t]);x1=max(0,x);y1=max(0,y);x2=min(rgb.shape[2],x+w);y2=min(rgb.shape[1],y+h)
   if x2<=x1 or y2<=y1:continue
   visible[t]=(x2-x1)*(y2-y1)/max(1,w*h)
   image=rgb[t,y1:y2,x1:x2]
   colors[t]=float(np.sqrt(refhist*hist(image)).sum())
   gx1=max(0,int(np.floor(x1/rgb.shape[2]*32)));gy1=max(0,int(np.floor(y1/rgb.shape[1]*32)))
   gx2=min(32,int(np.ceil(x2/rgb.shape[2]*32)));gy2=min(32,int(np.ceil(y2/rgb.shape[1]*32)))
   feat=patch[t,gy1:gy2,gx1:gx2].astype(np.float32).mean((0,1))
   feat/=np.linalg.norm(feat)+1e-9
   dinos[t]=float(np.dot(ref,feat))
   d=depth[t,y1:y2,x1:x2];valid=d[(d>.05)&np.isfinite(d)]
   depths[t]=float(np.median(valid)) if len(valid) else np.nan
 tmp=target.with_suffix('.partial.npz')
 if tmp.exists():raise RuntimeError(str(tmp))
 np.savez_compressed(tmp,color_hellinger=colors,dino_cosine=dinos,depth_median_m=depths,
  bbox_visible_fraction=visible,tracker_ok=ok,rgbd_path=np.asarray(str(rgbd)),
  no_simulator_truth=np.asarray(True),no_actions=np.asarray(True),no_manual_labels=np.asarray(True))
 tmp.replace(target)
 return dict(split=split,id=id,status='complete',frames=n,seconds=round(time.time()-start,2),
  color_first=float(colors[0]),color_last=float(colors[-1]),dino_first=float(dinos[0]),dino_last=float(dinos[-1]))
def main():
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   row=one(split,id);rows.append(row);print(json.dumps(row),flush=True)
 manifest=OUT/'manifest.json'
 record=dict(status='visual_track_full_consistency_complete',rows=rows,
  source='full RGB-D sequence and frozen DINOv2 features; automatic MIL candidate track',
  caveat='appearance self-consistency only; no true identity, contact, or grasp metric',
  no_simulator_truth=True,no_actions=True,no_names=True,no_manual_labels=True)
 if manifest.exists():assert json.loads(manifest.read_text())['status']==record['status']
 else:manifest.write_text(json.dumps(record,indent=2))
 print(json.dumps(dict(status='complete',episodes=len(rows))),flush=True)
if __name__=='__main__':main()
