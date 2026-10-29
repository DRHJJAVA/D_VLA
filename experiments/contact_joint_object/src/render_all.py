from pathlib import Path
import os,sys,json,hashlib,subprocess,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,h5py
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/contact_joint_object'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
rows=[r for r in json.loads((B/'manifests/episodes.json').read_text())['episodes'] if r['split'] in ['train','dev']]
assert len(rows)==450 and all(int(r['episode'][5:]) not in range(40,45) for r in rows)
env=os.environ.copy();env.update(PYTHONPATH=str(P/'src'),MUJOCO_GL='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
def one(r):
 q=B/'cache/rgbd'/(r['id']+'.h5');audit=B/'cache/rgbd'/(r['id']+'.audit.json')
 if q.exists():
  with h5py.File(q,'r') as h:compatible='finger_surface' in h and 'robot_mask' in h['agentview']
  if not compatible:
   old=q.with_suffix('.initial_smoke.h5');assert not old.exists();q.rename(old)
 if not q.exists():
  log=B/'logs'/('render_'+r['id']+'.log')
  with log.open('w') as f:
   run=subprocess.run([sys.executable,str(B/'src/render_episode.py'),'--input',str(P/r['raw']),'--output',str(q),'--libero-root','/home/drh/project/ECV-VLA/vendor/LIBERO','--episode',r['episode'],'--size','256'],env=env,stdout=f,stderr=f)
  if run.returncode:raise RuntimeError('render subprocess failed: '+r['id'])
 with h5py.File(q,'r') as h,h5py.File(P/r['raw'],'r') as raw:
  g=raw['data'][r['episode']];n=r['frames'];assert len(h['frame_indices'])==n
  checks={};hashes={}
  for k,v in [('ee_position','ee_pos'),('gripper_qpos','gripper_states'),('joint_qpos','joint_states')]:
   x=h['robot/'+k][:];y=g['obs/'+v][:]
   assert np.isfinite(x).all() and np.isfinite(y).all()
   checks[k]={'previous_max':float(np.max(abs(x[1:]-y[:-1]))),'same_max':float(np.max(abs(x-y)))}
   hashes['obs/'+v]=hashlib.sha256(y.tobytes()).hexdigest()
  assert checks['gripper_qpos']['previous_max']<1e-5 and checks['joint_qpos']['previous_max']<1e-5
  for name in ['actions','states','obs/ee_ori']:
   hashes[name]=hashlib.sha256(g[name][:].tobytes()).hexdigest()
  checks['rgbd']={}
  for cam in ['agentview','robot0_eye_in_hand']:
   dep=h[cam+'/depth_m'][:];K=h[cam+'/K'][:];C=h[cam+'/camera_to_world'][:]
   assert dep.shape==(n,256,256) and np.isfinite(dep).all() and (dep>0).all()
   assert np.max(abs(np.linalg.det(C[:,:3,:3])-1))<1e-4
   rr=np.random.default_rng(2026);uv=rr.integers(0,256,(64,2));tt=np.arange(64)%n
   rays=np.stack([(uv[:,0]-K[tt,0,2])/K[tt,0,0],(uv[:,1]-K[tt,1,2])/K[tt,1,1],np.ones(64)],1)
   xyz=rays*dep[tt,uv[:,1],uv[:,0],None];w=np.einsum('nij,nj->ni',C[tt,:3,:3],xyz)+C[tt,:3,3]
   back=np.linalg.solve(C[tt,:3,:3],(w-C[tt,:3,3])[...,None])[...,0]
   image=np.stack([back[:,0]/back[:,2]*K[tt,0,0]+K[tt,0,2],back[:,1]/back[:,2]*K[tt,1,1]+K[tt,1,2]],1)
   err=float(np.max(abs(image-uv)));assert err<1e-4
   checks['rgbd'][cam]={'projection_max_px':err,'robot_pixels_min':int(h[cam+'/robot_mask'][:].sum((1,2)).min())}
   hashes[cam+'/rgbd']=hashlib.sha256(h[cam+'/rgb'][:].tobytes()+dep.tobytes()+K.tobytes()+C.tobytes()).hexdigest()
  hashes['raw_allowed_episode']=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
 save(audit,{'time':now(),'id':r['id'],'split':r['split'],'checks':checks,'hashes':hashes,'frame0_robot_obs_alignment':'unknown','model_never_reads_states':True,'contact_labels_generated':False})
 return {'id':r['id'],'frames':r['frames'],'split':r['split']}
done=[];start=time.time()
save(B/'stage_state.json',{'stage':'rendering_all_train_dev','time':now(),'count':0,'total':450})
try:
 with ThreadPoolExecutor(max_workers=2) as pool:
  for fut in as_completed([pool.submit(one,r) for r in rows]):
   done.append(fut.result())
   save(B/'manifests/render_progress.json',{'time':now(),'completed':len(done),'total':len(rows),'frames':sum(r['frames'] for r in done),'elapsed_seconds':time.time()-start,'episodes':done})
 save(B/'manifests/render_complete.json',{'time':now(),'completed':len(done),'frames':sum(r['frames'] for r in done),'episodes':done})
 save(B/'stage_state.json',{'stage':'render_complete_awaiting_feature_pipeline','time':now(),'count':len(done)})
except Exception as e:
 (B/'logs/render_all_traceback.log').write_text(traceback.format_exc())
 save(B/'failure.json',{'time':now(),'stage':'render_all','exception':type(e).__name__,'detail':str(e),'completed':len(done)})
 raise
