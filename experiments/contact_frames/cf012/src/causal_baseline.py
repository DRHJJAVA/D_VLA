import os
os.environ.setdefault('MUJOCO_GL','egl')
from common12 import *
import numpy as np,cv2,h5py
from scipy.ndimage import maximum_filter,minimum_filter
sys.path.insert(0,str(ROOT/'src/v3'))
from stage2_robot_only_depth import RobotDepth
from stage2_probe_motion_fingerpads import cloud
cv2.setNumThreads(2)
def dense_pair(current_gray,previous_gray):
 a=CFG['flow_parameters']
 args=(None,a['pyr_scale'],a['levels'],a['winsize'],a['iterations'],a['poly_n'],a['poly_sigma'],a['flags'])
 return cv2.calcOpticalFlowFarneback(current_gray,previous_gray,*args),cv2.calcOpticalFlowFarneback(previous_gray,current_gray,*args)
def main():
 paths=[];cases={}
 for ep in CFG['episodes']:
  olddata=ROOT/'experiments/contact_frames/cf005/data'/ep
  c=json.loads((olddata/'config.json').read_text());cases[ep]=c
  paths += [olddata/'config.json',ROOT/c['rgbd'],ROOT/c['motion'],PREV/'data'/f'{ep}.npz']
 paths += [ROOT/'src/v3/stage2_robot_only_depth.py',ROOT/'src/v3/stage2_probe_motion_fingerpads.py',BASE/'config.json',BASE/'src/causal_baseline.py']
 save(BASE/'baseline_input_locked.json',lock(paths))
 rows=[]
 for ep,c in cases.items():
  with np.load(PREV/'data'/f'{ep}.npz') as z:
   pixels=z['pixels'];xyz=z['xyz'];eligible=z['eligible'];targetmask=z['mask'];target=z['target']
  with np.load(ROOT/c['motion']) as z:H=z['hand_to_world'];q=z['measured_finger_q']
  T,N=xyz.shape[:2]
  velocity=np.zeros_like(xyz);valid=np.zeros((T,N),bool);fb=np.full((T,N),np.nan,np.float32)
  with h5py.File(ROOT/c['rgbd'],'r') as h:
   height,width=h['agentview/depth_m'].shape[1:]
   robot=RobotDepth(height);previous=None
   for t in range(T):
    rgb=h['agentview/rgb'][t];dep=h['agentview/depth_m'][t]
    K=h['agentview/K'][t].astype(np.float64);C=h['agentview/camera_to_world'][t].astype(np.float64)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    world=cloud(dep,K,C).reshape(height,width,3)
    m=np.isfinite(dep)&(dep>0)&np.isfinite(world).all(-1)
    if t==0:m[:]=False
    else:
     rd=robot.render(q[t],np.linalg.inv(H[t].astype(np.float64))@C,K)
     m &= (rd>1)|(dep<rd-.002)
    m &= (maximum_filter(dep,size=3)-minimum_filter(dep,size=3))<=CFG['flow_parameters']['edge_depth_m']
    if previous is not None and t in eligible:
     pg,pw,pm=previous
     # Entire dense flows are computed before accessing any evaluation pixels.
     back,forward=dense_pair(gray,pg)
     x,y=pixels[t].T;v=back[y,x]
     past=pixels[t].astype(np.float32)+v
     px=past[:,0].astype(np.float32)[None];py=past[:,1].astype(np.float32)[None]
     rev=cv2.remap(forward,px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[0]
     fb[t]=np.linalg.norm(v+rev,axis=-1)
     ij=np.rint(past).astype(np.int64)
     inside=(ij[:,0]>=0)&(ij[:,0]<width)&(ij[:,1]>=0)&(ij[:,1]<height)&(past[:,0]>=0)&(past[:,0]<=width-1)&(past[:,1]>=0)&(past[:,1]<=height-1)
     ij[:,0]=np.clip(ij[:,0],0,width-1);ij[:,1]=np.clip(ij[:,1],0,height-1)
     previous_xyz=pw[ij[:,1],ij[:,0]]
     velocity[t]=xyz[t]-previous_xyz
     valid[t]=inside&m[y,x]&pm[ij[:,1],ij[:,0]]&(fb[t]<=CFG['flow_parameters']['fb_px'])&np.isfinite(velocity[t]).all(-1)
    previous=(gray,world,m)
   robot.close()
  pred=np.stack([velocity*k for k in CFG['horizons']],axis=1)
  out=BASE/'causal_baseline'/f'{ep}.npz';out.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(out,mean=pred,valid=valid,fb_error_px=fb,eligible=eligible)
  counts=[int((targetmask[eligible,j]&valid[eligible]).sum()) for j in range(3)]
  totals=[int(targetmask[eligible,j].sum()) for j in range(3)]
  rows.append(dict(episode=ep,common_points_by_horizon=counts,original_points_by_horizon=totals,coverage=[c/n if n else None for c,n in zip(counts,totals)]))
  save(BASE/'baseline_progress.json',dict(time=now(),rows=rows))
 verify(json.loads((BASE/'baseline_input_locked.json').read_text()))
 save(BASE/'baseline_locked.json',lock((BASE/'causal_baseline').glob('*.npz')))
 save(BASE/'baseline_complete.json',dict(time=now(),rows=rows,flow_input='raw RGB t-1,t only; query selection occurs after full dense flow',velocity_input='measured RGB-D at t-1,t, robot/camera calibration',no_contact_labels=True))
if __name__=='__main__':
 try:main()
 except Exception as e:
  (BASE/'baseline_failure_traceback.log').write_text(traceback.format_exc())
  save(BASE/'baseline_failure.json',dict(time=now(),type=type(e).__name__,frames=[dict(file=x.filename,line=x.lineno) for x in traceback.extract_tb(e.__traceback__)]))
  raise SystemExit(1)
