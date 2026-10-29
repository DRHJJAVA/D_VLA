"""Train-only RGB-D rigid surface correspondence pilot within visual object tracks.
No actions, names, simulator truth/contacts, manual masks, or reserved episodes."""
from pathlib import Path
import json
import cv2,h5py,numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
COMP=R/'data/v3/proposals/foreground3d_components/train'
TRACK=R/'data/v3/tracking/mil_448_v1/train'
OUT=R/'audits/v3/stage2_tracked_surface_rigid_pilot.json'
IMG=R/'docs/图示/第二步_跟踪物体表面三维对应内部核对.jpg'
CASES={'tomato_sauce_demo_3':[45,75,110,150],
       'porcelain_mug_demo_4':[36,57,70,90],
       'white_yellow_mug_demo_2':[45,61,75,100]}
def bins(rgb):
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
 return np.stack((np.minimum(hsv[:,:,0]//15,11),np.minimum(hsv[:,:,1]//64,3),np.minimum(hsv[:,:,2]//64,3)),axis=-1)
def make_lut(rgb,mask,box):
 b=bins(rgb);x1,y1,x2,y2=box
 ring=np.zeros(mask.shape,bool)
 ring[max(0,y1-20):min(mask.shape[0],y2+20),max(0,x1-20):min(mask.shape[1],x2+20)]=True
 ring[mask]=False
 a=np.bincount(np.ravel_multi_index(b[mask].T,(12,4,4)),minlength=192).astype(float)
 c=np.bincount(np.ravel_multi_index(b[ring].T,(12,4,4)),minlength=192).astype(float)
 a=(a+.5)/(a.sum()+96);c=(c+.5)/(c.sum()+96)
 return np.log(a/c)
def mask_now(rgb,dep,bbox,lut):
 x,y,w,h=map(int,bbox);x1=max(0,x);y1=max(0,y);x2=min(rgb.shape[1],x+w);y2=min(rgb.shape[0],y+h)
 m=np.zeros(dep.shape,bool)
 if x2<=x1 or y2<=y1:return m
 b=bins(rgb[y1:y2,x1:x2]);s=lut[np.ravel_multi_index(b.reshape(-1,3).T,(12,4,4))].reshape(b.shape[:2])
 q=(s>0)&(dep[y1:y2,x1:x2]>.05)&np.isfinite(dep[y1:y2,x1:x2])
 q=cv2.morphologyEx(q.astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 m[y1:y2,x1:x2]=q>0
 return m
def unproject(p,dep,K,T):
 x=np.rint(p[:,0]).astype(int);y=np.rint(p[:,1]).astype(int)
 valid=(x>=0)&(x<dep.shape[1])&(y>=0)&(y<dep.shape[0])
 z=np.full(len(p),np.nan)
 z[valid]=dep[y[valid],x[valid]]
 xyz=np.stack(((p[:,0]-K[0,2])*z/K[0,0],(p[:,1]-K[1,2])*z/K[1,1],z),1)
 xyz=xyz@T[:3,:3].T+T[:3,3]
 return xyz,np.isfinite(z)&(z>.05)
def fit(a,b):
 ac=a.mean(0);bc=b.mean(0)
 u,s,vt=np.linalg.svd((a-ac).T@(b-bc))
 d=np.eye(3);d[-1,-1]=np.linalg.det(vt.T@u.T)
 rot=vt.T@d@u.T
 trans=bc-rot@ac
 return rot,trans
def ransac(a,b,seed):
 if len(a)<3:return None
 rng=np.random.default_rng(seed);best=None
 for _ in range(700):
  ids=rng.choice(len(a),3,replace=False)
  if np.linalg.norm(np.cross(a[ids[1]]-a[ids[0]],a[ids[2]]-a[ids[0]]))<2e-5:continue
  rot,trans=fit(a[ids],b[ids])
  e=np.linalg.norm(a@rot.T+trans-b,axis=1)
  good=e<.008
  if best is None or good.sum()>best.sum():best=good
 if best is None or best.sum()<6:return None
 rot,trans=fit(a[best],b[best])
 e=np.linalg.norm(a@rot.T+trans-b,axis=1)
 good=e<.008
 if good.sum()<6:return None
 rot,trans=fit(a[good],b[good])
 return rot,trans,good,e
rows=[];tiles=[]
sift=cv2.SIFT_create(nfeatures=1200,contrastThreshold=.02)
bf=cv2.BFMatcher(cv2.NORM_L2)
for ri,(id,frames) in enumerate(CASES.items()):
 with np.load(COMP/(id+'.npz')) as z:pix=z['pixel'];path=Path(str(z['rgbd_path']));box=z['component_box'].astype(int)
 with np.load(TRACK/(id+'.npz')) as z:boxes=z['boxes']
 with h5py.File(path) as f:
  first=f['agentview/rgb'][0];d0=f['agentview/depth_m'][0];k0=f['agentview/K'][0];t0=f['agentview/camera_to_world'][0]
  init=np.zeros(first.shape[:2],np.uint8);init[pix[:,1],pix[:,0]]=255
  lut=make_lut(first,init>0,box)
  gray0=cv2.cvtColor(first,cv2.COLOR_RGB2GRAY)
  kp0,des0=sift.detectAndCompute(gray0,init)
  for ci,t in enumerate(frames):
   rgb=f['agentview/rgb'][t];dep=f['agentview/depth_m'][t];kt=f['agentview/K'][t];tt=f['agentview/camera_to_world'][t]
   m=mask_now(rgb,dep,boxes[t],lut)
   kp1,des1=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),m.astype(np.uint8)*255)
   candidates=[]
   if des0 is not None and des1 is not None and len(des1)>=2:
    for pair in bf.knnMatch(des0,des1,k=2):
     if len(pair)==2 and pair[0].distance<.78*pair[1].distance:candidates.append(pair[0])
   points0=np.asarray([kp0[q.queryIdx].pt for q in candidates],np.float32).reshape(-1,2)
   points1=np.asarray([kp1[q.trainIdx].pt for q in candidates],np.float32).reshape(-1,2)
   if len(candidates):
    a,va=unproject(points0,d0,k0,t0);b,vb=unproject(points1,dep,kt,tt)
    valid=va&vb
    candidates=[q for q,v in zip(candidates,valid) if v];a=a[valid];b=b[valid]
   else:a=b=np.empty((0,3))
   result=ransac(a,b,seed=t+ri*1000)
   row=dict(id=id,frame=t,first_sift=len(kp0),current_sift=len(kp1),
       ratio_matches=len(candidates),color_support_pixels=int(m.sum()),
       rigid_inliers=0,rigid_inlier_fraction=0.,median_residual_mm=None,
       translation_cm=None,rotation_deg=None,abstain=True)
   goodmatches=[]
   if result is not None:
    rot,trans,good,e=result
    ang=np.degrees(np.arccos(np.clip((np.trace(rot)-1)/2,-1,1)))
    row.update(rigid_inliers=int(good.sum()),rigid_inlier_fraction=float(good.mean()),
        median_residual_mm=float(np.median(e[good])*1000),
        translation_cm=float(np.linalg.norm(trans)*100),rotation_deg=float(ang),
        abstain=bool(good.sum()<10 or good.mean()<.25 or np.median(e[good])>.005))
    goodmatches=[q for q,g in zip(candidates,good) if g]
   aimg=cv2.cvtColor(first,cv2.COLOR_RGB2BGR);bimg=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
   draw=cv2.drawMatches(aimg,kp0,bimg,kp1,goodmatches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
   draw=cv2.resize(draw,(768,384),interpolation=cv2.INTER_AREA)
   cv2.rectangle(draw,(0,0),(768,32),(0,0,0),-1)
   title=id.replace('_demo_','_')+' t'+str(t)+' m'+str(len(candidates))+' in'+str(row['rigid_inliers'])+(' ABSTAIN' if row['abstain'] else ' candidate')
   cv2.putText(draw,title,(5,21),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
   tiles.append((ri,ci,draw));rows.append(row);print(json.dumps(row),flush=True)
canvas=np.zeros((3*384,4*768,3),np.uint8)
for ri,ci,draw in tiles:canvas[ri*384:(ri+1)*384,ci*768:(ci+1)*768]=draw
if IMG.exists() or OUT.exists():raise RuntimeError('output already exists')
cv2.imwrite(str(IMG),canvas,[cv2.IMWRITE_JPEG_QUALITY,88])
OUT.write_text(json.dumps(dict(rows=rows,method='SIFT on automatic firstframe object pixels and appearance-supported tracked box, RGB-D K/T unprojection, 8mm 3D rigid RANSAC',
    caveat='train-only pilot; inliers are geometry self-consistency, not true object identity, contact, grasp, or wrapped region',
    no_actions=True,no_names=True,no_manual_labels=True,no_simulator_truth=True),indent=2))
print('complete',len(rows),flush=True)
