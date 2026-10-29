"""Internal visual audit gallery of automatic RGB tracker trajectories, TRAIN/VAL only."""
from pathlib import Path
import json,h5py,cv2,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
ROOT=R/'data/v3/tracking/mil_448_v1'
assert json.loads((ROOT/'manifest.json').read_text())['status']=='mil_448_v1_complete'
TILE=192
gallery=[]
for split in ('train','val'):
 ids=S[split]
 for start in range(0,len(ids),5):
  batch=ids[start:start+5]
  canvas=np.zeros((len(batch)*TILE,5*TILE,3),np.uint8)
  for row,id in enumerate(batch):
   with np.load(ROOT/split/(id+'.npz')) as z:
    boxes=z['boxes'];ok=z['ok'];path=Path(str(z['rgbd_path']))
   n=len(boxes);frames=np.unique(np.rint(np.linspace(0,n-1,5)).astype(int))
   assert len(frames)==5
   with h5py.File(path) as f:
    for col,t in enumerate(frames):
     rgb=f['agentview/rgb'][int(t)]
     x,y,w,h=map(int,boxes[t])
     cx=x+w/2;cy=y+h/2
     span=max(w,h)*2.4
     x1=max(0,int(cx-span/2));y1=max(0,int(cy-span/2))
     x2=min(rgb.shape[1],int(cx+span/2));y2=min(rgb.shape[0],int(cy+span/2))
     im=cv2.cvtColor(rgb[y1:y2,x1:x2],cv2.COLOR_RGB2BGR)
     cv2.rectangle(im,(x-x1,y-y1),(x+w-x1,y+h-y1),(0,255,0),2)
     im=cv2.resize(im,(TILE,TILE),interpolation=cv2.INTER_AREA)
     cv2.rectangle(im,(0,0),(TILE,28),(0,0,0),-1)
     label=id.replace('_demo_','_')[:17]+' t'+str(t)
     cv2.putText(im,label,(3,19),cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,255),1,cv2.LINE_AA)
     if not bool(ok[t]):cv2.circle(im,(TILE-12,14),6,(0,0,255),-1)
     canvas[row*TILE:(row+1)*TILE,col*TILE:(col+1)*TILE]=im
  p=R/'docs/图示'/('第二步_MIL物体轨迹_'+split+'_'+str(start//5+1)+'_内部核对.jpg')
  if p.exists():raise RuntimeError(str(p))
  cv2.imwrite(str(p),canvas,[cv2.IMWRITE_JPEG_QUALITY,91])
  gallery.append(dict(split=split,ids=batch,path=str(p)))
  print(json.dumps(gallery[-1]),flush=True)
p=R/'audits/v3/stage2_mil_track_gallery_manifest.json'
if p.exists():raise RuntimeError(str(p))
p.write_text(json.dumps(dict(gallery=gallery,frames_each=5,source='automatic tracker output; not identity truth'),indent=2))
print('complete',len(gallery))
