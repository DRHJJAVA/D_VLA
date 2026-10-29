"""Visual-only color-support pilot inside automatic tracked object boxes, TRAIN only."""
from pathlib import Path
import json
import numpy as np,h5py,cv2
R=Path('/mnt/nvme1/drh/D_VLA')
COMP=R/'data/v3/proposals/foreground3d_components/train'
TRACK=R/'data/v3/tracking/mil_448_v1/train'
OUT=R/'audits/v3/stage2_tracked_object_color_support_pilot.json'
GALLERY=R/'docs/图示'
CASES={
 'tomato_sauce_demo_3':[0,45,75,110,150],
 'porcelain_mug_demo_4':[0,36,57,70,90],
 'white_yellow_mug_demo_2':[0,45,61,75,100],
}
def bins(rgb):
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
 return np.stack((np.minimum(hsv[:,:,0]//15,11),np.minimum(hsv[:,:,1]//64,3),np.minimum(hsv[:,:,2]//64,3)),axis=-1)
rows=[]
canvas=np.zeros((3*256,5*256,3),np.uint8)
for ri,(id,frames) in enumerate(CASES.items()):
 with np.load(COMP/(id+'.npz')) as z:pix=z['pixel'];path=Path(str(z['rgbd_path']));box=z['component_box'].astype(int)
 with np.load(TRACK/(id+'.npz')) as z:track=z['boxes']
 with h5py.File(path) as f:
  initial=f['agentview/rgb'][0]
  initmask=np.zeros(initial.shape[:2],bool);initmask[pix[:,1],pix[:,0]]=True
  x1,y1,x2,y2=box
  ring=np.zeros_like(initmask);pad=20
  ring[max(0,y1-pad):min(len(ring),y2+pad),max(0,x1-pad):min(ring.shape[1],x2+pad)]=True
  ring[initmask]=False
  b=bins(initial)
  obj=np.bincount(np.ravel_multi_index(b[initmask].T,(12,4,4)),minlength=192).astype(float)
  bg=np.bincount(np.ravel_multi_index(b[ring].T,(12,4,4)),minlength=192).astype(float)
  obj=(obj+0.5)/(obj.sum()+0.5*192);bg=(bg+0.5)/(bg.sum()+0.5*192)
  lut=np.log(obj/bg)
  for ci,t in enumerate(frames):
   rgb=f['agentview/rgb'][t];depth=f['agentview/depth_m'][t];x,y,w,h=map(int,track[t])
   xlo=max(0,x);ylo=max(0,y);xhi=min(rgb.shape[1],x+w);yhi=min(rgb.shape[0],y+h)
   sub=rgb[ylo:yhi,xlo:xhi];bd=bins(sub)
   score=lut[np.ravel_multi_index(bd.reshape(-1,3).T,(12,4,4))].reshape(sub.shape[:2])
   support=(score>0)&(depth[ylo:yhi,xlo:xhi]>.05)
   support=cv2.morphologyEx(support.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)
   overlay=cv2.cvtColor(sub,cv2.COLOR_RGB2BGR)
   yellow=np.zeros_like(overlay);yellow[:,:,0]=0;yellow[:,:,1]=255;yellow[:,:,2]=255
   overlay[support]=cv2.addWeighted(overlay,.55,yellow,.45,0)[support]
   tile=cv2.resize(overlay,(256,256),interpolation=cv2.INTER_AREA)
   cv2.rectangle(tile,(0,0),(256,32),(0,0,0),-1)
   cv2.putText(tile,id.replace('_demo_','_')[:17]+' t'+str(t),(3,21),cv2.FONT_HERSHEY_SIMPLEX,.41,(255,255,255),1,cv2.LINE_AA)
   canvas[ri*256:(ri+1)*256,ci*256:(ci+1)*256]=tile
   rows.append(dict(id=id,frame=t,support_fraction=float(support.mean()),support_points=int(support.sum()),box=[x,y,w,h]))
  print(id,flush=True)
p=GALLERY/'第二步_跟踪框物体颜色支持内部核对.jpg'
if p.exists():raise RuntimeError(str(p))
cv2.imwrite(str(p),canvas,[cv2.IMWRITE_JPEG_QUALITY,91])
record=dict(rows=rows,source='train full RGB-D, firstframe automatic foreground HSV vs local background, MIL tracked boxes',no_contact_truth=True,no_manual_labels=True,no_actions=True,caveat='color support is not object segmentation or grasp labels')
if OUT.exists():raise RuntimeError(str(OUT))
OUT.write_text(json.dumps(record,indent=2))
print('complete',len(rows),str(p))
