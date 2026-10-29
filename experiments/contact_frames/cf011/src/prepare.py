import os
os.environ.setdefault('MUJOCO_GL','egl')
from common import *
import sys,numpy as np,h5py,cv2
from scipy.ndimage import maximum_filter,minimum_filter
sys.path.insert(0,str(ROOT/'src/v3'))
from stage2_robot_only_depth import RobotDepth
from stage2_probe_motion_fingerpads import cloud
def main():
 stage('preparing_inputs')
 paths=[BASE/'config.json']
 cases={}
 for ep in CFG['train_episodes']:
  p=ROOT/'experiments/contact_frames/cf005/data'/ep
  c=json.loads((p/'config.json').read_text());cases[ep]=(p,c)
  paths += [p/'config.json',p/'artifacts/reference.npz',p/'artifacts/tracks.npz',p/'artifacts/dataset.npz',ROOT/c['rgbd'],ROOT/c['motion']]
 paths += [ROOT/'src/v3/stage2_robot_only_depth.py',ROOT/'src/v3/stage2_probe_motion_fingerpads.py']
 save(BASE/'input_locked.json',lock(paths))
 rows=[]
 for ep,(p,c) in cases.items():
  stage('preparing_episode',episode=ep)
  with np.load(p/'artifacts/reference.npz') as z:center=z['surface_xyz'].mean(0).astype(np.float64)
  with np.load(p/'artifacts/tracks.npz') as z:uv=z['tracks'].copy();vis=z['visibility'].copy()
  with np.load(p/'artifacts/dataset.npz') as z:states=z['states'].copy();state_valid=z['state_valid'].copy()
  with np.load(ROOT/c['motion']) as z:H=z['hand_to_world'].copy();q=z['measured_finger_q'].copy()
  assert np.max(np.abs(states[1:,:3]-H[1:,:3,3]))<1e-6
  assert np.max(np.abs(states[1:,9:11]-q[1:]))<1e-6
  states[:,0:3]=(states[:,0:3]-center)/.1
  states=np.nan_to_num(states)
  T,N=uv.shape[:2];R=CFG['resolution'];F=[];D=[];RGB=[];Ks=[];Cs=[]
  xyz=np.zeros((T,N,3),np.float32);colors=np.zeros_like(xyz)
  good=np.zeros((T,N),bool);grid=np.zeros((T,N,2),np.float32);pixels=np.zeros((T,N,2),np.int64)
  max_reprojection=0.
  with h5py.File(ROOT/c['rgbd'],'r') as h:
   assert len(h['agentview/depth_m'])==T
   height,width=h['agentview/depth_m'].shape[1:]
   assert height==width
   robot=RobotDepth(height)
   for t in range(T):
    dep=h['agentview/depth_m'][t];rgb=h['agentview/rgb'][t].astype(np.float32)/255.
    K=h['agentview/K'][t].astype(np.float64);C=h['agentview/camera_to_world'][t].astype(np.float64)
    world=cloud(dep,K,C).reshape(height,width,3)
    valid=np.isfinite(dep)&(dep>0)&np.isfinite(world).all(-1)
    if t==0:valid[:]=False
    else:
     rd=robot.render(q[t],np.linalg.inv(H[t].astype(np.float64))@C,K)
     valid &= (rd>1)|(dep<rd-.002)
    # Inputs use raw current/past pixels only. No tracking/motion fit enters them.
    feats=np.concatenate([rgb,(world-center)/.1,valid[...,None]],-1).astype(np.float32)
    feats[~valid,:6]=0
    fr=np.concatenate([cv2.resize(feats[:,:,:3],(R,R),interpolation=cv2.INTER_AREA),cv2.resize(feats[:,:,3:],(R,R),interpolation=cv2.INTER_NEAREST)],-1)
    # Remove partially averaged RGB at an invalid nearest depth pixel.
    fr[fr[:,:,-1]<.5,:6]=0
    F.append(fr.transpose(2,0,1));D.append(cv2.resize(dep,(R,R),interpolation=cv2.INTER_NEAREST)[None])
    RGB.append(cv2.resize(rgb,(R,R),interpolation=cv2.INTER_AREA).transpose(2,0,1))
    K2=K.copy();K2[0,:]*=R/width;K2[1,:]*=R/height
    # OpenCV resizing pixel-centre convention.
    K2[0,2]+=(R/width-1)/2;K2[1,2]+=(R/height-1)/2
    Ks.append(K2);Cs.append(C)
    finite_uv=np.isfinite(uv[t]).all(-1)
    ij=np.rint(np.nan_to_num(uv[t])).astype(np.int64)
    bounds=finite_uv&(ij[:,0]>=0)&(ij[:,0]<width)&(ij[:,1]>=0)&(ij[:,1]<height)
    ij[:,0]=np.clip(ij[:,0],0,width-1);ij[:,1]=np.clip(ij[:,1],0,height-1)
    x,y=ij.T;pixels[t]=ij;xyz[t]=world[y,x];colors[t]=rgb[y,x]
    span=maximum_filter(dep,size=3)-minimum_filter(dep,size=3)
    good[t]=bounds&vis[t]&valid[y,x]&(span[y,x]<=CFG['depth_edge_limit_m'])
    grid[t,:,0]=2*(x+.5)/width-1;grid[t,:,1]=2*(y+.5)/height-1
    cam=(xyz[t].astype(np.float64)-C[:3,3])@np.linalg.inv(C[:3,:3]).T
    proj=cam@K.T;proj=proj[:,:2]/proj[:,2:]
    if good[t].any():max_reprojection=max(max_reprojection,float(np.max(np.abs(proj[good[t]]-ij[good[t]]))))
   robot.close()
  F=np.stack(F).astype(np.float32);D=np.stack(D).astype(np.float32);RGB=np.stack(RGB).astype(np.float32)
  target=np.zeros((T,3,N,3),np.float32);masks=np.zeros((T,3,N),bool)
  for j,k in enumerate(CFG['horizons']):
   target[:-k,j]=xyz[k:]-xyz[:-k];masks[:-k,j]=good[:-k]&good[k:]
  anchor=(np.arange(T)>=CFG['history']-1)&(np.arange(T)<T-max(CFG['horizons']))&state_valid
  eligible=np.flatnonzero(anchor&masks.any((1,2)))
  assert len(eligible)>0 and max_reprojection<.001
  out=BASE/'data'/f'{ep}.npz';out.parent.mkdir(exist_ok=True)
  np.savez_compressed(out,frames=F,depth=D,rgb=RGB,states=states,grid=grid,xyz=xyz,color=colors,point_valid=good,target=target,mask=masks,eligible=eligible,K=np.stack(Ks).astype(np.float32),C=np.stack(Cs).astype(np.float32),center=center,pixels=pixels)
  row=dict(episode=ep,frames=T,eligible_frames=len(eligible),valid_pairs_by_horizon=[int(masks[eligible,j].sum()) for j in range(3)],available_pairs_per_horizon=len(eligible)*N,reprojection_max_px=max_reprojection,input_shape=list(F.shape))
  rows.append(row);save(BASE/'preparation_progress.json',dict(rows=rows,time=now()))
 verify(json.loads((BASE/'input_locked.json').read_text()))
 save(BASE/'data_locked.json',lock((BASE/'data').glob('*.npz')))
 save(BASE/'audits/preparation.json',dict(time=now(),rows=rows,camera='agentview',filter='known finger geometry only; full arm not guaranteed removed',tracker='offline pseudo-correspondences used only after dense forward for supervision',frame_alignment='replay t = official observation t-1; t0 excluded'))
 stage('inputs_ready')
if __name__=='__main__':
 try:main()
 except Exception as e:failure(e);raise SystemExit(1)
