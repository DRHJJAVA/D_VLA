from pathlib import Path
import json,functools
import numpy as np,h5py,torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from geometry import finger_geometry,graph,rigid
from sampling import fps
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/contact_joint_object'
CAMS=['agentview','robot0_eye_in_hand']
def features(frame,obj):
 uv=obj['uv'][:];cam=obj['camera'][:];out=np.zeros((len(uv),384),'f4')
 for ci in [0,1]:
  sel=np.flatnonzero(cam==ci)
  if not len(sel):continue
  ft=frame['features_'+str(ci)][:].astype('f4');xy=np.clip((uv[sel]+.5)/256*32-.5,0,31);lo=np.floor(xy).astype(int);hi=np.minimum(lo+1,31);a=xy-lo
  out[sel]=ft[lo[:,1],lo[:,0]]*((1-a[:,0])*(1-a[:,1]))[:,None]+ft[lo[:,1],hi[:,0]]*(a[:,0]*(1-a[:,1]))[:,None]+ft[hi[:,1],lo[:,0]]*((1-a[:,0])*a[:,1])[:,None]+ft[hi[:,1],hi[:,0]]*(a[:,0]*a[:,1])[:,None]
 return out
def mask(obj,ci):
 k='mask_'+str(ci)
 return np.unpackbits(obj[k][:])[:256*256].reshape(256,256).astype(bool) if k in obj else np.zeros((256,256),bool)
def plane_from_depth(h,t,omasks):
 points=[]
 for ci,cam in enumerate(CAMS):
  dep=h[cam+'/depth_m'][t];K=h[cam+'/K'][t];C=h[cam+'/camera_to_world'][t]
  valid=(h[cam+'/robot_mask'][t]==0)&~omasks[ci];uv=np.argwhere(valid)[::32,::-1]
  z=dep[uv[:,1],uv[:,0]];a=np.column_stack(((uv[:,0]-K[0,2])*z/K[0,0],(uv[:,1]-K[1,2])*z/K[1,1],z))
  points.append(a@C[:3,:3].T+C[:3,3])
 x=np.concatenate(points);rng=np.random.default_rng(2026)
 if len(x)<32:return np.zeros(4,'f4'),False
 best=None;count=0
 for _ in range(32):
  a,b,c=x[rng.choice(len(x),3,replace=False)];n=np.cross(b-a,c-a);norm=np.linalg.norm(n)
  if norm<1e-8:continue
  n/=norm;d=-a@n;support=abs(x@n+d)<.003;s=int(support.sum())
  if s>count:best=(n,d);count=s
 if best is None or count<len(x)*.1:return np.zeros(4,'f4'),False
 n,d=best
 # Plane orientation points toward the robot workspace camera side, not object class.
 if n[2]<0:n=-n;d=-d
 return np.r_[n,d].astype('f4'),True
def to_device(b,device='cuda'):
 out={}
 for k,v in b.items():
  if isinstance(v,dict):out[k]=to_device(v,device)
  elif isinstance(v,np.ndarray):out[k]=torch.from_numpy(v.astype('f4') if v.dtype.kind=='f' else v).to(device)
  else:out[k]=v
 return out
class Dataset:
 def __init__(self,split,allow_heldout=False):
  self.rows=[r for r in json.loads((B/'manifests/episodes.json').read_text())['episodes'] if r['split']==split]
  assert split in ['train','dev'] or allow_heldout and (B/'runs/v001/frozen_for_test.json').exists()
  self.split=split;self.sigma=.003
  q=B/'manifests/train_measurement_scale.json'
  if q.exists():self.sigma=json.loads(q.read_text())['sigma_m']
  self.hand_index=None
 def paths(self,r):return B/'cache/observations'/(r['id']+'.h5'),B/'cache/rgbd'/(r['id']+'.h5')
 def frame_objects(self,r,t):
  path,_=self.paths(r)
  with h5py.File(path,'r') as h:return [int(x) for x in h['frames/'+str(t)].attrs['object_ids']]
 def get(self,r,t,oid,ci=0,point_limit=2048):
  op,rp=self.paths(r)
  with h5py.File(op,'r') as obs,h5py.File(rp,'r') as h:
   f=obs['frames/'+str(t)];obj=f['object_'+str(oid)];xyz=obj['xyz'][:];normal=obj['normal'][:];uv=obj['uv'][:];cams=obj['camera'][:]
   feat=features(f,obj);selected=fps(xyz.astype(float),min(point_limit,len(xyz)));xyz=xyz[selected];normal=normal[selected];uv=uv[selected];cams=cams[selected]
   b={'xyz':xyz,'normal':normal,'features':feat[selected],'graph':graph(xyz),'id':r['id'],'frame':t,'object_id':oid,'uv':uv,'camera_ids':cams,'source_index':selected}
   b.update(finger_geometry(h,t,xyz));omasks=[mask(obj,k) for k in [0,1]];meta=json.loads(h.attrs['robot_metadata']);hand=meta['body_names'].index('robot0_right_hand')
   ctx=[];ctxvalid=[]
   for tt in range(t-15,t+1):
    if tt<0:ctx.append(np.zeros(402));ctxvalid.append(False);continue
    ff=obs['frames/'+str(tt)];name='object_'+str(oid)
    if name not in ff:ctx.append(np.zeros(402));ctxvalid.append(False);continue
    ob=ff[name];ct=np.r_[ob['desc'][:],ob['xyz'][:].mean(0),h['robot/body_position'][tt,hand],h['robot/body_rotation'][tt,hand].reshape(-1),h['robot/gripper_qpos'][tt],1]
    assert ct.shape==(402,);ctx.append(ct);ctxvalid.append(True)
   b['context']=np.asarray(ctx,'f4');b['context_valid']=np.asarray(ctxvalid,bool)
   b['measurement_sigma']=np.asarray(self.sigma,'f4')
   tr=obs['tracks/'+str(ci)];source=np.empty((0,3),'f4');target=source.copy();tu=np.empty((0,2),'f4');td=np.empty(0,'f4');hist=np.zeros(6,'f4');history_error=np.nan
   if 'world' in tr and t+1<r['frames'] and t>0:
    pix=np.rint(tr['uv'][t]).astype(int);valid=tr['valid'][t]&tr['valid'][t+1]
    valid&=(pix[:,0]>=0)&(pix[:,0]<256)&(pix[:,1]>=0)&(pix[:,1]<256)
    ids=np.flatnonzero(valid);ids=ids[omasks[ci][pix[ids,1],pix[ids,0]]]
    if len(ids):
     _,unique=np.unique(pix[ids],axis=0,return_index=True);ids=ids[np.sort(unique)]
     source=tr['world'][t,ids];target=tr['world'][t+1,ids];tu=np.rint(tr['uv'][t+1,ids]).astype('f4')
     dep=h[CAMS[ci]+'/depth_m'][t+1];td=dep[tu[:,1].astype(int),tu[:,0].astype(int)]
     prev=tr['valid'][t-1,ids];pi=ids[prev]
     if len(pi)>=3:
      A=tr['world'][t-1,pi];C=tr['world'][t,pi];R,v,history_error=rigid(A,C)
      if np.isfinite(history_error):
       center=xyz.mean(0);hist[:3]=(center@R.T+v-center);hist[3:]=Rotation.from_matrix(R).as_rotvec()
   b.update(motion_source=source,motion_target=target,motion_target_uv=tu,motion_target_depth=td,history_twist=hist,history_fit_error=history_error,target_K=h[CAMS[ci]+'/K'][min(t+1,r['frames']-1)],target_camera=h[CAMS[ci]+'/camera_to_world'][min(t+1,r['frames']-1)])
   plane,valid=plane_from_depth(h,t,omasks);b['environment_plane']=plane;b['environment_plane_valid']=valid
   b['point_area']=np.square(cKDTree(xyz).query(xyz,k=min(4,len(xyz)))[0][:,-1]).astype('f4');b['point_area']/=max(float(b['point_area'].sum()),1e-12)
   return b
