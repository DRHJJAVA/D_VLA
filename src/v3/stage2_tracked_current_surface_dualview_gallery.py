"""Visual audit of tracked current object candidate in the wrist RGB-D view (train only)."""
from pathlib import Path
import json,cv2,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
COMP=R/'data/v3/proposals/foreground3d_components/train'
TRACK=R/'data/v3/tracking/mil_448_v1/train'
OUT=R/'docs/图示/第二步_当前物体双视角几何内部核对.jpg'
CASES={'tomato_sauce_demo_3':[45,75,110],
       'porcelain_mug_demo_4':[36,57,70],
       'white_yellow_mug_demo_2':[45,61,75]}
def bins(rgb):
 h=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
 return np.stack((np.minimum(h[:,:,0]//15,11),np.minimum(h[:,:,1]//64,3),np.minimum(h[:,:,2]//64,3)),axis=-1)
def lut(rgb,mask,box):
 b=bins(rgb);x1,y1,x2,y2=box
 ring=np.zeros(mask.shape,bool);ring[max(0,y1-20):min(mask.shape[0],y2+20),max(0,x1-20):min(mask.shape[1],x2+20)]=True;ring[mask]=False
 a=np.bincount(np.ravel_multi_index(b[mask].T,(12,4,4)),minlength=192).astype(float)
 c=np.bincount(np.ravel_multi_index(b[ring].T,(12,4,4)),minlength=192).astype(float)
 return np.log((a+.5)/(a.sum()+96)/(c+.5)*(c.sum()+96))
def masknow(rgb,dep,bbox,ll):
 x,y,w,h=map(int,bbox);x1=max(0,x);y1=max(0,y);x2=min(rgb.shape[1],x+w);y2=min(rgb.shape[0],y+h)
 m=np.zeros(dep.shape,bool)
 if x2<=x1 or y2<=y1:return m
 b=bins(rgb[y1:y2,x1:x2]);s=ll[np.ravel_multi_index(b.reshape(-1,3).T,(12,4,4))].reshape(b.shape[:2])
 m[y1:y2,x1:x2]=(s>0)&(dep[y1:y2,x1:x2]>.05)&np.isfinite(dep[y1:y2,x1:x2])
 return m
canvas=np.zeros((3*360,3*360,3),np.uint8)
for ri,(id,frames) in enumerate(CASES.items()):
 with np.load(COMP/(id+'.npz')) as z:pix=z['pixel'];path=Path(str(z['rgbd_path']));box=z['component_box']
 with np.load(TRACK/(id+'.npz')) as z:boxes=z['boxes']
 with h5py.File(path) as f:
  first=f['agentview/rgb'][0];init=np.zeros(first.shape[:2],bool);init[pix[:,1],pix[:,0]]=True;ll=lut(first,init,box)
  for ci,t in enumerate(frames):
   a=f['agentview'];w=f['robot0_eye_in_hand']
   rgb=a['rgb'][t];dep=a['depth_m'][t];m=masknow(rgb,dep,boxes[t],ll)
   yy,xx=np.nonzero(m);z=dep[yy,xx];K=a['K'][t];T=a['camera_to_world'][t]
   cam=np.stack(((xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z),1)
   xyz=cam@T[:3,:3].T+T[:3,3]
   wk=w['K'][t];wt=w['camera_to_world'][t];wd=w['depth_m'][t]
   wc=(xyz-wt[:3,3])@wt[:3,:3];zz=wc[:,2];safe=np.where(zz>.01,zz,1)
   u=np.rint(wk[0,0]*wc[:,0]/safe+wk[0,2]).astype(int)
   v=np.rint(wk[1,1]*wc[:,1]/safe+wk[1,2]).astype(int)
   inside=(zz>.05)&(u>=0)&(u<448)&(v>=0)&(v<448)
   ids=np.flatnonzero(inside);delta=wd[v[ids],u[ids]]-zz[ids]
   match=ids[np.abs(delta)<.015];near=ids[delta<-.02]
   im=w['rgb'][t].copy()
   for inds,color in ((near,(255,60,60)),(match,(25,245,80))):
    if len(inds):
     im[v[inds],u[inds]]=np.asarray(color,np.uint8)
   tile=cv2.resize(im,(360,360),interpolation=cv2.INTER_AREA)
   cv2.rectangle(tile,(0,0),(360,30),(0,0,0),-1)
   txt=id.replace('_demo_','_')+' t'+str(t)+' green '+str(len(match))+' red '+str(len(near))
   cv2.putText(tile,txt[:42],(3,19),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
   canvas[ri*360:(ri+1)*360,ci*360:(ci+1)*360]=tile
if OUT.exists():raise RuntimeError(str(OUT))
cv2.imwrite(str(OUT),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,91])
print(str(OUT),flush=True)
