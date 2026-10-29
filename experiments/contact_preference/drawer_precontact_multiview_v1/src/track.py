import os,sys,json,hashlib,traceback,ast
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ['MUJOCO_GL']='egl'
import numpy as np,h5py,torch
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/contact_preference/drawer_precontact_multiview_v1';OLD=R/'experiments/contact_preference/drawer_hidden_surface_v1';C=R/'experiments/contact_frames/cf017/cases/drawer_demo_0'
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'third_party/co_tracker_010'))
from src.v3.stage2_robot_only_depth import RobotDepth
from cotracker.predictor import CoTrackerPredictor
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def back(dep,K,T,uv):
 u,v=uv.T;z=dep[v,u];p=np.c_[(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z];return p@T[:3,:3].T+T[:3,3]
def main():
 cfg=json.loads((D/'config.json').read_text());mot=np.load(C/'motion.npz');centers=np.load(OLD/'data/source_seed.npz')['centers']
 hp=R/'experiments/contact_frames/cf017/rgbd/drawer_demo_0.h5';wp=R/'experiments/region_pose/iter_010/weights/scaled_offline.pth'
 save(D/'input_locked.json',{str(p):sha(p) for p in [hp,wp,C/'motion.npz',OLD/'data/source_seed.npz',D/'config.json',Path(__file__)]})
 robot=RobotDepth(448);h=h5py.File(hp,'r');cases=[]
 for sf in cfg['source_frames']:
  for cam in cfg['cameras']:
   dep=h[cam+'/depth_m'][sf];K=h[cam+'/K'][sf];T=h[cam+'/camera_to_world'][sf];H=mot['hand_to_world'][sf];q=mot['measured_finger_q'][sf]
   rd=robot.render(q,np.linalg.inv(H)@T,K);v,u=np.indices(dep.shape);uv=np.c_[u.ravel(),v.ravel()];P=back(dep,K,T,uv)
   valid=(dep.ravel()>0)&np.isfinite(dep.ravel())&(dep.ravel()<rd.ravel()-.002)&(np.linalg.norm(P[:,None]-centers[None],axis=2).min(1)<=.08)
   uv=uv[valid];P=P[valid];rgb=h[cam+'/rgb'][sf][uv[:,1],uv[:,0]]
   _,ix=np.unique(uv//2,axis=0,return_index=True);ix=np.sort(ix)
   if len(ix)>600:ix=ix[np.linspace(0,len(ix)-1,600).astype(int)]
   su=uv[ix];sx=P[ix];held=(((su[:,0]*73856093)^(su[:,1]*19349663))%5)==0
   key=cam+'_'+str(sf)
   np.savez_compressed(D/'data'/(key+'_source.npz'),world=P,uv=uv,rgb=rgb,seeds_world=sx,seeds_uv=su,heldout=held,source_frame=sf,K=K,T=T)
   cases.append({'key':key,'source_frame':sf,'camera':cam,'surface_points':len(P),'seeds':len(ix),'heldout':int(held.sum())})
 robot.close();save(D/'cases.json',cases)
 save(D/'seeds_locked.json',{str(D/'data'/(r['key']+'_source.npz')):sha(D/'data'/(r['key']+'_source.npz')) for r in cases})
 torch.manual_seed(2026);torch.cuda.set_per_process_memory_fraction(.4)
 model=CoTrackerPredictor(checkpoint=str(wp),offline=True,v2=False,window_len=60).cuda().eval()
 for r in cases:
  sf=r['source_frame'];key=r['key'];cam=r['camera'];seed=np.load(D/'data'/(key+'_source.npz'))
  save(D/'stage_state.json',{'time':now(),'stage':'tracking','case':key})
  if r['seeds']<8:
   r['tracking']='insufficient';continue
  video=torch.from_numpy(h[cam+'/rgb'][sf:121]).permute(0,3,1,2)[None].float().cuda()
  qu=torch.from_numpy(np.c_[np.zeros(r['seeds']),seed['seeds_uv']].astype(np.float32))[None].cuda()
  with torch.no_grad():tr,vis=model(video,queries=qu,backward_tracking=False)
  np.savez_compressed(D/'data'/(key+'_tracks.npz'),tracks=tr[0].cpu().numpy(),visibility=vis[0].cpu().numpy(),first_frame=sf)
  del video,qu,tr,vis;torch.cuda.empty_cache();r['tracking']='complete';print(json.dumps(r),flush=True)
 h.close();save(D/'cases.json',cases);save(D/'stage_state.json',{'time':now(),'stage':'tracked','cases':len(cases)})
if __name__=='__main__':
 try:main()
 except Exception as e:
  (D/'failure.log').write_text(traceback.format_exc());save(D/'failure.json',{'time':now(),'type':type(e).__name__,'message':str(e),'stack':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]});sys.exit(1)
