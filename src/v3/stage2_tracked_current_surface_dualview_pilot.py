"""TRAIN-only current tracked object surface cross-view RGB-D visibility pilot."""
from pathlib import Path
import json,cv2,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
COMP=R/'data/v3/proposals/foreground3d_components/train'
TRACK=R/'data/v3/tracking/mil_448_v1/train'
OUT=R/'audits/v3/stage2_tracked_current_surface_dualview_pilot.json'
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
 b=bins(rgb[y1:y2,x1:x2]);s=ll[np.ravel_multi_index(b.reshape(-1,3).T,(12,4,4))].reshape(b.shape[:2])
 m[y1:y2,x1:x2]=(s>0)&(dep[y1:y2,x1:x2]>.05)&np.isfinite(dep[y1:y2,x1:x2])
 return m
def cloud(pix,depth,K,T):
 x=pix[:,0];y=pix[:,1];z=depth[y,x]
 valid=(z>.05)&np.isfinite(z)
 x=x[valid];y=y[valid];z=z[valid]
 cam=np.stack(((x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z),1)
 return cam@T[:3,:3].T+T[:3,3]
def project(xyz,K,T,depth):
 cam=(xyz-T[:3,3])@T[:3,:3];z=cam[:,2]
 safe=np.where(z>.01,z,1)
 u=np.rint(K[0,0]*cam[:,0]/safe+K[0,2]).astype(int)
 v=np.rint(K[1,1]*cam[:,1]/safe+K[1,2]).astype(int)
 inside=(z>.05)&(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
 delta=np.full(len(z),np.nan)
 ii=np.flatnonzero(inside)
 delta[ii]=depth[v[ii],u[ii]]-z[ii]
 return inside,delta
rows=[]
for id,frames in CASES.items():
 with np.load(COMP/(id+'.npz')) as z:pix=z['pixel'];path=Path(str(z['rgbd_path']));box=z['component_box']
 with np.load(TRACK/(id+'.npz')) as z:boxes=z['boxes']
 with h5py.File(path) as f:
  first=f['agentview/rgb'][0];init=np.zeros(first.shape[:2],bool);init[pix[:,1],pix[:,0]]=True;ll=lut(first,init,box)
  for t in frames:
   a=f['agentview'];w=f['robot0_eye_in_hand']
   rgb=a['rgb'][t];depth=a['depth_m'][t];m=masknow(rgb,depth,boxes[t],ll)
   xyz=cloud(np.argwhere(m)[:,::-1],depth,a['K'][t],a['camera_to_world'][t])
   wd=w['depth_m'][t];wi,delta=project(xyz,w['K'][t],w['camera_to_world'][t],wd)
   row=dict(id=id,frame=t,main_appearance_points=len(xyz),
     wrist_in_frame=float(wi.mean()) if len(wi) else 0.,
     wrist_depth_matches_15mm=float((wi&(np.abs(delta)<.015)).mean()) if len(wi) else 0.,
     wrist_nearer_occluder=float((wi&(delta<-.02)).mean()) if len(wi) else 0.,
     wrist_behind_or_disoccluded=float((wi&(delta>.025)).mean()) if len(wi) else 0.)
   rows.append(row);print(json.dumps(row),flush=True)
if OUT.exists():raise RuntimeError(str(OUT))
OUT.write_text(json.dumps(dict(rows=rows,reference='current appearance-supported main-camera candidate object points projected to wrist at same time',
 caveat='main object mask is only a visual candidate; wrist overlap is not new-surface identity, contact, or grasp evidence',
 no_simulator_truth=True,no_actions=True,no_names=True,no_manual_labels=True),indent=2))
print('complete',len(rows),flush=True)
