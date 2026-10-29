"""TRAIN-only visual RGB-D rigid ICP pilot; fit quality is not surface identity."""
from pathlib import Path
import json,cv2,h5py,numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
COMP=R/'data/v3/proposals/foreground3d_components/train'
TRACK=R/'data/v3/tracking/mil_448_v1/train'
OUT=R/'audits/v3/stage2_tracked_surface_icp_pilot.json'
CASES={'tomato_sauce_demo_3':[45,75,110,150],
       'porcelain_mug_demo_4':[36,57,70,90],
       'white_yellow_mug_demo_2':[45,61,75,100]}
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
 b=bins(rgb[y1:y2,x1:x2])
 s=ll[np.ravel_multi_index(b.reshape(-1,3).T,(12,4,4))].reshape(b.shape[:2])
 m[y1:y2,x1:x2]=(s>0)&(dep[y1:y2,x1:x2]>.05)&np.isfinite(dep[y1:y2,x1:x2])
 return m
def cloud(pix,dep,K,T):
 x=pix[:,0];y=pix[:,1];z=dep[y,x]
 valid=(z>.05)&np.isfinite(z)
 x=x[valid];y=y[valid];z=z[valid]
 cam=np.stack(((x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z),1)
 xyz=cam@T[:3,:3].T+T[:3,3]
 return xyz
def down(xyz):
 ix=np.floor(xyz/.003).astype(np.int32)
 _,u=np.unique(ix,axis=0,return_index=True)
 return xyz[np.sort(u)]
def fit(a,b):
 ac=a.mean(0);bc=b.mean(0)
 u,s,v=np.linalg.svd((a-ac).T@(b-bc))
 d=np.eye(3);d[-1,-1]=np.linalg.det(v.T@u.T)
 rot=v.T@d@u.T
 return rot,bc-rot@ac
def icp(source,target):
 if len(source)<50 or len(target)<50:return None
 tree=cKDTree(target)
 rot=np.eye(3);trans=np.median(target,axis=0)-np.median(source,axis=0)
 for _ in range(40):
  moved=source@rot.T+trans;dist,ids=tree.query(moved)
  cutoff=min(.025,float(np.quantile(dist,.65)))
  keep=dist<=cutoff
  if keep.sum()<30:break
  r,t=fit(moved[keep],target[ids[keep]])
  rot=r@rot;trans=r@trans+t
  if np.linalg.norm(t)<.0001 and np.linalg.norm(r-np.eye(3))<.001:break
 moved=source@rot.T+trans
 fwd=tree.query(moved)[0]
 rev=cKDTree(moved).query(target)[0]
 ang=np.degrees(np.arccos(np.clip((np.trace(rot)-1)/2,-1,1)))
 return dict(forward_8mm=float(np.mean(fwd<.008)),reverse_8mm=float(np.mean(rev<.008)),
    median_forward_mm=float(np.median(fwd)*1000),median_reverse_mm=float(np.median(rev)*1000),
    translation_cm=float(np.linalg.norm(trans)*100),rotation_deg=float(ang))
rows=[]
for id,frames in CASES.items():
 with np.load(COMP/(id+'.npz')) as z:pix=z['pixel'];path=Path(str(z['rgbd_path']));box=z['component_box']
 with np.load(TRACK/(id+'.npz')) as z:boxes=z['boxes']
 with h5py.File(path) as f:
  rgb0=f['agentview/rgb'][0];dep0=f['agentview/depth_m'][0]
  mask=np.zeros(dep0.shape,bool);mask[pix[:,1],pix[:,0]]=True
  ll=lut(rgb0,mask,box);source=down(cloud(pix,dep0,f['agentview/K'][0],f['agentview/camera_to_world'][0]))
  for t in frames:
   rgb=f['agentview/rgb'][t];dep=f['agentview/depth_m'][t];m=masknow(rgb,dep,boxes[t],ll)
   target=down(cloud(np.argwhere(m)[:,::-1],dep,f['agentview/K'][t],f['agentview/camera_to_world'][t]))
   result=icp(source,target)
   row=dict(id=id,frame=t,source_points=len(source),current_color_points=len(target),icp=result)
   rows.append(row);print(json.dumps(row),flush=True)
if OUT.exists():raise RuntimeError(str(OUT))
OUT.write_text(json.dumps(dict(rows=rows,method='3mm voxel and trimmed point-to-point ICP on firstframe automatic object and current appearance-supported tracked box',
 caveat='geometric fit is necessary but not sufficient for correct object identity or unique surface correspondence; symmetric textureless cups remain ambiguous',
 no_actions=True,no_names=True,no_manual_labels=True,no_simulator_truth=True),indent=2))
print('complete',len(rows),flush=True)
