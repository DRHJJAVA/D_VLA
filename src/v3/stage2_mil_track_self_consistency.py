"""Visual-only appearance consistency audit for MIL track candidates; TRAIN/VAL only."""
from pathlib import Path
import json
import numpy as np,h5py,cv2
R=Path('/mnt/nvme1/drh/D_VLA')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
E={x['id']:x for x in (json.loads(line) for line in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines())}
COMP=R/'data/v3/proposals/foreground3d_components'
TRACK=R/'data/v3/tracking/mil_448_v1'
FEATURE=R/'data/v3/features/dinov2_all_frames'
OUT=R/'audits/v3/stage2_mil_track_self_consistency_5frame.json'
def color_hist(rgb,mask=None):
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
 pts=hsv[mask] if mask is not None else hsv.reshape(-1,3)
 hist=np.histogramdd(pts,bins=(12,4,4),range=((0,180),(0,256),(0,256)))[0]
 hist=(hist+1e-5).reshape(-1)
 return hist/hist.sum()
def similarity(a,b):
 return float(np.sqrt(a*b).sum())
rows=[]
for split in ('train','val'):
 for id in S[split]:
  e=E[id]
  with np.load(COMP/split/(id+'.npz')) as z:
   pix=z['pixel'];rgbd=Path(str(z['rgbd_path']))
  with np.load(TRACK/split/(id+'.npz')) as z:boxes=z['boxes'];ok=z['ok']
  fp=FEATURE/e['asset_group']/(Path(e['rgbd_path']).stem+'.h5')
  with h5py.File(rgbd) as f,h5py.File(fp) as g:
   frame=f['agentview/rgb'];patch=g['agentview/patch'];n=len(boxes)
   base=frame[0];mask=np.zeros(base.shape[:2],bool);mask[pix[:,1],pix[:,0]]=True
   hist0=color_hist(base,mask)
   gridmask=np.zeros((32,32),bool);gridmask[np.clip((pix[:,1]/base.shape[0]*32).astype(int),0,31),np.clip((pix[:,0]/base.shape[1]*32).astype(int),0,31)]=True
   ref=patch[0][gridmask].astype(np.float32).mean(0);ref/=np.linalg.norm(ref)+1e-9
   ts=np.unique(np.rint(np.linspace(0,n-1,5)).astype(int))
   for t in ts:
    x,y,w,h=map(int,boxes[t]);rgb=frame[int(t)]
    x1=max(0,x);y1=max(0,y);x2=min(rgb.shape[1],x+w);y2=min(rgb.shape[0],y+h)
    crop=rgb[y1:y2,x1:x2]
    hist=color_hist(crop);hs=similarity(hist0,hist)
    gx1=max(0,int(np.floor(x1/rgb.shape[1]*32)));gy1=max(0,int(np.floor(y1/rgb.shape[0]*32)))
    gx2=min(32,int(np.ceil(x2/rgb.shape[1]*32)));gy2=min(32,int(np.ceil(y2/rgb.shape[0]*32)))
    feat=patch[t,gy1:gy2,gx1:gx2].astype(np.float32).mean((0,1))
    feat/=np.linalg.norm(feat)+1e-9
    ds=float(np.dot(ref,feat))
    d=f['agentview/depth_m'][int(t),y1:y2,x1:x2]
    rows.append(dict(split=split,id=id,frame=int(t),nframes=n,ok=bool(ok[t]),tracked_box=[x,y,w,h],
     color_hellinger=hs,dino_cosine=ds,depth_median_m=float(np.median(d[d>.05])) if np.any(d>.05) else None))
  print(split,id,flush=True)
record=dict(rows=rows,status='visual_self_consistency_audit_complete',no_contact_truth=True,
 note='first-frame automatic object foreground color and DINO descriptors versus candidate box at five uniformly spaced frames; no identity labels')
if OUT.exists():assert json.loads(OUT.read_text())==record
else:OUT.write_text(json.dumps(record,indent=2))
print('complete',len(rows))
