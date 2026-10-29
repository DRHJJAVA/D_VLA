from common13 import *
import cv2,h5py,importlib.util
sys.path.insert(0,str(ROOT/'src/v3'))
from stage2_robot_only_depth import RobotDepth
def pair(g,p):
 args=(None,.5,3,15,3,5,1.2,0)
 return cv2.calcOpticalFlowFarneback(g,p,*args),cv2.calcOpticalFlowFarneback(p,g,*args)
def world(dep,K,C):
 h,w=dep.shape;y,x=np.mgrid[:h,:w]
 a=np.stack([(x-K[0,2])*dep/K[0,0],(y-K[1,2])*dep/K[1,1],dep],-1)
 return a@C[:3,:3].T+C[:3,3]
def main():
 paths=[BASE/'config.json',PREV/'completed_locked.json',ROOT/'experiments/contact_frames/cf012/completed_locked.json',ROOT/'src/v3/stage2_robot_only_depth.py']
 for ep in CFG['train_episodes']:
  cp=ROOT/'experiments/contact_frames/cf005/data'/ep/'config.json'
  cc=json.loads(cp.read_text());paths += [cp,ROOT/cc['rgbd'],ROOT/cc['motion'],PREV/'data'/f'{ep}.npz',ROOT/'experiments/contact_frames/cf012/causal_baseline'/f'{ep}.npz']
 save(BASE/'input_locked.json',lock(paths))
 verify(json.loads((PREV/'completed_locked.json').read_text()))
 verify(json.loads((ROOT/'experiments/contact_frames/cf012/completed_locked.json').read_text()))
 rows=[]
 for ep in CFG['train_episodes']:
  stage('preparing_dense_causal_motion',episode=ep)
  c=json.loads((ROOT/'experiments/contact_frames/cf005/data'/ep/'config.json').read_text())
  data=np.load(PREV/'data'/f'{ep}.npz');mot=np.load(ROOT/c['motion']);prior=np.load(ROOT/'experiments/contact_frames/cf012/causal_baseline'/f'{ep}.npz')
  out=BASE/'data'/ep;out.mkdir(parents=True,exist_ok=True)
  with h5py.File(ROOT/c['rgbd'],'r') as f:
   dset=f['agentview'];T,H,W=dset['depth_m'].shape
   renderer=RobotDepth(H)
   field=np.lib.format.open_memmap(out/'dense_motion.npy',mode='w+',dtype=np.float32,shape=(T,4,H,W));field[:]=0
   small=np.zeros((T,4,96,96),np.float32);points=np.zeros((T,128,4),np.float32)
   previous=None; xyzerr=0.;nvalid=0
   for t in range(T):
    dep=np.asarray(dset['depth_m'][t],np.float32);rgb=np.asarray(dset['rgb'][t])
    K=np.asarray(dset['K'][t]);C=np.asarray(dset['camera_to_world'][t])
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);xyz=world(dep,K,C)
    valid=np.isfinite(dep)&(dep>0)
    edge=cv2.dilate(dep,np.ones((3,3),np.uint8))-cv2.erode(dep,np.ones((3,3),np.uint8))
    valid &= edge<=.01
    if t>0:
     rd=renderer.render(mot['measured_finger_q'][t],np.linalg.inv(mot['hand_to_world'][t])@C,K)
     valid &= (rd>1)|(dep<rd-.002)
    else:valid[:]=False
    if previous is not None:
     pg,pxyz,pvalid=previous
     back,forward=pair(gray,pg)
     yy,xx=np.mgrid[:H,:W];uv=np.stack([xx,yy],-1).astype(np.float32)+back
     reverse=cv2.remap(forward,uv[...,0],uv[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
     fb=np.linalg.norm(back+reverse,axis=-1)
     ip=np.rint(uv).astype(np.int64);inside=(ip[...,0]>=0)&(ip[...,0]<W)&(ip[...,1]>=0)&(ip[...,1]<H)
     sx=ip[...,0].clip(0,W-1);sy=ip[...,1].clip(0,H-1)
     good=valid&inside&pvalid[sy,sx]&(fb<=1.5)
     vel=xyz-pxyz[sy,sx];vel[~good]=0
     field[t,:3]=vel.transpose(2,0,1);field[t,3]=good
     small[t]=cv2.resize(np.asarray(field[t]).transpose(1,2,0),(96,96),interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
     nvalid+=int(good.sum())
    previous=(gray,xyz,valid)
    pix=data['pixels'][t];x=pix[:,0].clip(0,W-1);y=pix[:,1].clip(0,H-1)
    points[t]=np.asarray(field[t,:,y,x])
    if t in data['eligible']:
     use=data['point_valid'][t]
     if use.any():xyzerr=max(xyzerr,float(np.max(np.abs(xyz[y,x][use]-data['xyz'][t][use]))))
   renderer.close();field.flush()
   elig=data['eligible'];samevalid=np.array_equal(points[elig,:,3].astype(bool),prior['valid'][elig])
   joint=points[elig,:,3].astype(bool)&prior['valid'][elig]
   delta=float(np.max(np.abs(points[elig,:,:3][joint]-prior['mean'][elig,0][joint]))) if joint.any() else 0
   # Both inputs use only past/current actual RGBD; no correspondence identity is used to construct the field.
   assert xyzerr<1e-6,('actual_XYZ_mismatch',ep,xyzerr)
   assert samevalid and delta<1e-6,('causal_baseline_mismatch',ep,samevalid,delta)
   np.savez_compressed(out/'motion_features.npz',motion96=small,point_motion=points,eligible=elig)
   rows.append(dict(episode=ep,frames=T,shape=[H,W],dense_valid_points=nvalid,old_causal_valid_exact=samevalid,old_causal_max_diff_m=delta,actual_xyz_max_diff_m=xyzerr))
   print(json.dumps(rows[-1]),flush=True)
 verify(json.loads((BASE/'input_locked.json').read_text()))
 save(BASE/'audits/motion_preparation.json',dict(time=now(),rows=rows,source='dense full-resolution current-previous flow before any teacher point sampling',future_inputs=False))
 save(BASE/'motion_locked.json',lock(sorted((BASE/'data').rglob('*.np*'))))
 stage('motion_ready')
if __name__=='__main__':main()
