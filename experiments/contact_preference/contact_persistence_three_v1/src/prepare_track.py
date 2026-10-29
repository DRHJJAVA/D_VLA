import os,sys,json,hashlib,traceback,importlib.util
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ['MUJOCO_GL']='egl'
os.environ['OMP_NUM_THREADS']='4'
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/contact_preference/contact_persistence_three_v1'
sys.path[:0]=[str(R),str(R/'src/v3'),str(R/'third_party/co_tracker_010')]
import numpy as np,h5py,cv2,torch
from scipy.spatial import cKDTree
from stage2_probe_motion_fingerpads import tc
from stage2_robot_only_depth import RobotDepth
from cotracker.predictor import CoTrackerPredictor
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def back(dep,K,T,uv):
 u,v=uv.T;z=dep[v,u];P=np.c_[(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z];return P@T[:3,:3].T+T[:3,3]
def main():
 cases=[
 dict(id='mug',title='夹取瓷杯并搬运 · demo9',rgbd=str(R/'data/v3/trainval_rgbd_448/porcelain_mug/demo_9.h5'),motion=str(R/'experiments/region_pose/iter_003/evidence/motion/porcelain_mug_demo_9.npz'),episode='demo_9'),
 dict(id='switch',title='操作炉灶开关 · demo0',rgbd=str(R/'experiments/contact_frames/cf017/rgbd/stove_demo_0.h5'),motion=str(R/'experiments/contact_frames/cf017/cases/stove_demo_0/motion.npz'),episode='demo_0'),
 dict(id='push',title='推关抽屉 · demo0',rgbd=str(D/'rgbd/push_drawer_demo_0.h5'),motion=None,episode='demo_0')]
 cams=['agentview','robot0_eye_in_hand'];robot=RobotDepth(448);locks={};audits=[]
 for c in cases:
  cid=c['id'];out=D/'data'/cid;out.mkdir(exist_ok=True);im=D/'images'/cid;im.mkdir(exist_ok=True)
  h=h5py.File(c['rgbd'],'r');N=len(h['frame_indices']);H=h['robot0_eye_in_hand/camera_to_world'][:]@np.linalg.inv(tc())
  raw=str(h.attrs['source_hdf5'])
  with h5py.File(raw,'r') as rr:
   g=rr['data/'+c['episode']];q=np.full((N,2),np.nan);q[1:]=g['obs/gripper_states'][:N-1]
  if c['motion']:
   old=np.load(c['motion']);assert np.allclose(H[1:],old['hand_to_world'][1:],atol=1e-7)
   assert np.array_equal(q[1:],old['measured_finger_q'][1:])
  H[0]=np.nan
  np.savez_compressed(out/'motion.npz',hand_to_world=H,measured_finger_q=q)
  locks[c['rgbd']]=sha(c['rgbd']);locks[raw]=sha(raw)
  center=(H[1:,:3,:3]@np.array([0,0,.09]))+H[1:,:3,3];tree=cKDTree(center)
  sfs=[1,N//3,2*N//3];c.update(frames=N,source_frames=sfs,raw=raw,cameras=cams)
  for cam in cams:
   valid=[];rdepth=[]
   for t in range(N):
    rgb=h[cam+'/rgb'][t];cv2.imwrite(str(im/f'{cam}_{t:03d}.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90])
    if t==0:rd=np.full((448,448),np.nan)
    else:rd=robot.render(q[t],np.linalg.inv(H[t])@h[cam+'/camera_to_world'][t],h[cam+'/K'][t])
    rdepth.append(rd.astype(np.float32))
   np.savez_compressed(out/f'{cam}_robot_depth.npz',depth=np.array(rdepth))
   for sf in sfs:
    dep=h[cam+'/depth_m'][sf];K=h[cam+'/K'][sf];T=h[cam+'/camera_to_world'][sf];rd=rdepth[sf]
    v,u=np.where(np.isfinite(dep)&(dep>0)&(dep<rd-.002));uv=np.c_[u,v];P=back(dep,K,T,uv)
    keep=tree.query(P)[0]<.075;uv=uv[keep];P=P[keep]
    # 1mm cells retain real observations, never averaged/invented.
    _,ii=np.unique(np.floor(P/.001).astype(np.int64),axis=0,return_index=True);ii=np.sort(ii)
    if len(ii)>10000:ii=ii[np.linspace(0,len(ii)-1,10000).astype(int)]
    fulluv=uv.copy();fullP=P.copy();uv=uv[ii];P=P[ii]
    # spatial image grid followed by deterministic spread; not target-dependent sampling.
    _,j=np.unique(uv//2,axis=0,return_index=True);j=np.sort(j)
    if len(j)>600:j=j[np.linspace(0,len(j)-1,600).astype(int)]
    suv=uv[j];held=((suv[:,0]*73856093+suv[:,1]*19349663)%5)==0
    key=f'{cam}_{sf}'
    np.savez_compressed(out/(key+'_source.npz'),world=P,uv=uv,rgb=h[cam+'/rgb'][sf][uv[:,1],uv[:,0]],full_world=fullP,full_uv=fulluv,seeds_world=P[j],seeds_uv=suv,heldout=held,source_frame=sf,K=K,T=T)
    audits.append(dict(case=cid,camera=cam,source_frame=sf,points=len(P),full_points=len(fullP),seeds=len(j),heldout=int(held.sum())))
  h.close()
 robot.close();save(D/'cases.json',cases);save(D/'source_audit.json',audits);save(D/'input_locked.json',locks)
 weights=R/'experiments/region_pose/iter_010/weights/scaled_offline.pth';locks[str(weights)]=sha(weights);save(D/'input_locked.json',locks)
 save(D/'tracking_config.json',dict(seed=2026,source_frames='1,N//3,2N//3',queries_per_camera_source_max=600,backward_tracking=True,why='full-video supervision includes pre/post observation; output includes every frame; no online causality claim',training=False))
 torch.manual_seed(2026);torch.cuda.set_per_process_memory_fraction(.4)
 model=CoTrackerPredictor(checkpoint=str(weights),offline=True,v2=False,window_len=60).cuda().eval()
 for c in cases:
  with h5py.File(c['rgbd'],'r') as h:
   for cam in cams:
    video=torch.from_numpy(h[cam+'/rgb'][:]).permute(0,3,1,2)[None].float().cuda()
    for sf in c['source_frames']:
     key=f'{cam}_{sf}';p=D/'data'/c['id'];s=np.load(p/(key+'_source.npz'))
     save(D/'stage_state.json',dict(stage='tracking',case=c['id'],camera=cam,source_frame=sf,time=now()))
     if len(s['seeds_uv'])<8:continue
     queries=torch.from_numpy(np.c_[np.full(len(s['seeds_uv']),sf),s['seeds_uv']].astype(np.float32))[None].cuda()
     with torch.no_grad():tr,vis=model(video,queries=queries,backward_tracking=True)
     np.savez_compressed(p/(key+'_tracks.npz'),tracks=tr[0].cpu().numpy(),visible=vis[0].cpu().numpy())
     print(json.dumps(dict(case=c['id'],camera=cam,source=sf,stage='tracked')),flush=True)
    del video;torch.cuda.empty_cache()
 save(D/'stage_state.json',dict(stage='tracking_complete',time=now()))
if __name__=='__main__':
 try:main()
 except Exception as e:
  (D/'prepare_track_traceback.log').write_text(traceback.format_exc())
  save(D/'failure.json',dict(stage='prepare_track',type=type(e).__name__,message=str(e),stack=[dict(file=x.filename,line=x.lineno) for x in traceback.extract_tb(e.__traceback__)]))
  save(D/'stage_state.json',dict(stage='failed',type=type(e).__name__,time=now()));sys.exit(1)
