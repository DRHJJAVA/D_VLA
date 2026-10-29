from common14 import *
import numpy as np,cv2,h5py
sys.path.insert(0,str(ROOT/'src/v3'))
from stage2_robot_only_depth import RobotDepth
def xyz(dep,K,C):
 y,x=np.mgrid[:dep.shape[0],:dep.shape[1]]
 a=np.stack([(x-K[0,2])*dep/K[0,0],(y-K[1,2])*dep/K[1,1],dep],-1)
 return a@C[:3,:3].T+C[:3,3]
def descriptor(points,vel,good,current_valid,H,Hp):
 local=(points-H[:3,3])@H[:3,:3]
 w=np.exp(-.5*np.sum((local-np.array([0,0,.0934]))**2,axis=-1)/.06**2)*current_valid
 total=float(w.sum());wg=w*good;mass=float(wg.sum())
 if mass<=1e-12:return np.zeros(24,np.float32)
 x=points[good];v=vel[good];ww=wg[good]/mass
 past=x-v;transport=H@np.linalg.inv(Hp);expected=past@transport[:3,:3].T+transport[:3,3]-past
 vh=v@H[:3,:3]/.01;eh=expected@H[:3,:3]/.01;rh=vh-eh
 def avg(a):return np.sum(a*ww[:,None],axis=0)
 speed=np.linalg.norm(vh,axis=1);es=np.linalg.norm(eh,axis=1)
 cosine=np.sum(vh*eh,axis=1)/(np.maximum(speed*es,1e-12))
 vals=np.concatenate([avg(vh),np.sqrt(avg(vh**2)),avg(rh),np.sqrt(avg(rh**2)),avg(eh),[np.sum(ww*speed),np.sqrt(np.sum(ww*speed**2)),np.sum(ww*np.linalg.norm(rh,axis=1)),np.sum(ww*es),np.sum(ww*cosine),mass/max(total,1e-12),np.log1p(mass),np.log1p(total),1.]])
 assert vals.shape==(24,) and np.isfinite(vals).all()
 return vals.astype(np.float32)
def main(phase):
 eps=CFG['train_episodes'] if phase=='train' else CFG['review_episodes']
 if phase=='review':assert (BASE/'weights_frozen.json').exists()
 paths=[BASE/'config.json',BASE/'src/prepare_features.py',ROOT/'src/v3/stage2_robot_only_depth.py']
 for ep in eps:
  cp=episode_config(ep);c=json.loads(cp.read_text());paths += [cp,ROOT/c['rgbd'],ROOT/c['motion'],dataset(ep)]
  if phase=='train':paths.append(ROOT/'experiments/contact_frames/cf013/data'/ep/'dense_motion.npy')
 save(BASE/f'{phase}_feature_inputs_locked.json',lock(paths));summary=[]
 for ep in eps:
  stage('preparing_'+phase+'_motion_features',episode=ep)
  c=json.loads(episode_config(ep).read_text());m=np.load(ROOT/c['motion'])
  prior=None
  if phase=='train':prior=np.load(ROOT/'experiments/contact_frames/cf013/data'/ep/'dense_motion.npy',mmap_mode='r')
  statevalid=np.load(dataset(ep))['state_valid'].astype(bool)
  with h5py.File(ROOT/c['rgbd'],'r') as f:
   g=f['agentview'];T,H,W=g['depth_m'].shape;assert T==len(statevalid)
   features=np.zeros((T,24),np.float32);counts=np.zeros(T,np.int64);renderer=RobotDepth(H);previous=None;prior_diff=0.;valid_diff=0
   for t in range(T):
    dep=g['depth_m'][t];K=g['K'][t];C=g['camera_to_world'][t];points=xyz(dep,K,C);gray=cv2.cvtColor(g['rgb'][t],cv2.COLOR_RGB2GRAY)
    valid=np.isfinite(dep)&(dep>0)
    valid &= cv2.dilate(dep,np.ones((3,3),np.uint8))-cv2.erode(dep,np.ones((3,3),np.uint8))<=.01
    if t>0:
     rd=renderer.render(m['measured_finger_q'][t],np.linalg.inv(m['hand_to_world'][t])@C,K);valid &= (rd>1)|(dep<rd-.002)
    else:valid[:]=False
    if t>1 and statevalid[t] and statevalid[t-1]:
     if prior is not None:
      vel=np.asarray(prior[t,:3]).transpose(1,2,0);good=np.asarray(prior[t,3]).astype(bool)
     else:
      pg,px,pvalid=previous
      back=cv2.calcOpticalFlowFarneback(gray,pg,None,.5,3,15,3,5,1.2,0);fw=cv2.calcOpticalFlowFarneback(pg,gray,None,.5,3,15,3,5,1.2,0)
      yy,xx=np.mgrid[:H,:W];uv=np.stack([xx,yy],-1).astype(np.float32)+back
      reverse=cv2.remap(fw,uv[...,0],uv[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
      ip=np.rint(uv).astype(np.int64);inside=(ip[...,0]>=0)&(ip[...,0]<W)&(ip[...,1]>=0)&(ip[...,1]<H)
      ix=ip[...,0].clip(0,W-1);iy=ip[...,1].clip(0,H-1)
      good=valid&inside&pvalid[iy,ix]&(np.linalg.norm(back+reverse,axis=-1)<=1.5)
      vel=points-px[iy,ix];vel[~good]=0
     features[t]=descriptor(points,vel,good,valid,m['hand_to_world'][t],m['hand_to_world'][t-1]);counts[t]=int(good.sum())
    previous=(gray,points,valid)
   renderer.close()
  out=BASE/'features'/f'{ep}.npz';assert not out.exists();np.savez_compressed(out,features=features,valid=features[:,-1]>0,point_counts=counts)
  row=dict(episode=ep,frames=T,valid_frames=int((features[:,-1]>0).sum()),reuse_cf013_dense=phase=='train');summary.append(row);print(json.dumps(row),flush=True)
 if phase=='train':
  allf=np.concatenate([np.load(BASE/'features'/f'{ep}.npz')['features'] for ep in eps]);ok=allf[:,-1]>0;mu=allf[ok].mean(0);sd=allf[ok].std(0);sd=np.maximum(sd,1e-6);mu[-1]=0;sd[-1]=1
  save(BASE/'motion_normalization.json',dict(mean=mu.tolist(),std=sd.tolist(),episodes=eps,valid_frames=int(ok.sum())))
 verify(json.loads((BASE/f'{phase}_feature_inputs_locked.json').read_text()))
 save(BASE/f'{phase}_features_summary.json',summary);save(BASE/f'{phase}_features_locked.json',lock([BASE/'features'/f'{ep}.npz' for ep in eps]))
 stage(phase+'_features_ready')
if __name__=='__main__':main(sys.argv[1])
