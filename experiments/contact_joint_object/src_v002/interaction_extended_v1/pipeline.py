import os,json,pathlib,subprocess,time,hashlib,traceback,concurrent.futures
from datetime import datetime
from zoneinfo import ZoneInfo
import h5py,numpy as np
BASE=pathlib.Path('/mnt/nvme1/drh/D_VLA');EXP=BASE/'experiments/contact_joint_object'
RUN=EXP/'runs/v002/preflight/P1_interaction_extended_v1';SRC=EXP/'src_v002/interaction_extended_v1'
PY='/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
def call(name,args):
 with (RUN/'logs'/name).open('ab') as log:
  p=subprocess.run([PY]+args,env=env,stdout=log,stderr=log)
 if p.returncode:raise RuntimeError('SUBPROCESS_'+name+'_EXIT_'+str(p.returncode))
try:
 protocol=json.loads((RUN/'protocol.json').read_text())
 if isinstance(protocol['episodes'][0],dict):
  protocol['episode_details']=protocol['episodes'];protocol['episodes']=[x['id'] for x in protocol['episode_details']]
  save(RUN/'protocol.json',protocol)
 for d in ['logs','rgbd']:(RUN/d).mkdir(exist_ok=True)
 env=os.environ.copy();env.update(MUJOCO_GL='egl',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(BASE/'src')+':'+str(EXP/'src')+':/home/drh/project/ECV-VLA/vendor/LIBERO')
 def render(row):
  id=row['id'];out=RUN/'rgbd'/(id+'.h5')
  call('render_'+id+'.log',[str(SRC/'render_episode.py'),'--input',str(RUN/'raw_demo0'/(id+'.hdf5')),'--output',str(out),'--libero-root','/home/drh/project/ECV-VLA/vendor/LIBERO','--bddl-subdir',row['suite'],'--episode','demo_0','--size','256'])
  checks={}
  with h5py.File(out,'r') as h,h5py.File(RUN/'raw_demo0'/(id+'.hdf5'),'r') as raw:
   n=len(h['frame_indices']);assert n==len(raw['data/demo_0/states'])
   for cam in ['agentview','robot0_eye_in_hand']:
    z=h[cam+'/depth_m'][:];assert z.shape==(n,256,256) and np.isfinite(z).all() and (z>0).all()
    checks[cam]={'frames':n,'robot_pixels_min':int(h[cam+'/robot_mask'][:].sum((1,2)).min()),'depth_finite_positive':True}
   for a,b in [('gripper_qpos','gripper_states'),('joint_qpos','joint_states')]:
    x=h['robot/'+a][:];y=raw['data/demo_0/obs/'+b][:]
    d=float(np.max(np.abs(x[1:]-y[:-1])));checks[a+'_previous_obs_error']=d;assert d<1e-5
  audit={'id':id,'frames':n,'checks':checks,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
  save(RUN/(id+'_render_check.json'),audit);return audit
 save(RUN/'pipeline_state.json',{'time':now(),'stage':'rendering'})
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rendered=list(pool.map(render,protocol['episode_details']))
 save(RUN/'render_complete.json',{'time':now(),'episodes':rendered})
 save(RUN/'pipeline_state.json',{'time':now(),'stage':'segmentation'})
 call('segmentation.log',[str(SRC/'segment_track.py')])
 for ep in protocol['episodes']:
  save(RUN/'pipeline_state.json',{'time':now(),'stage':'association','episode':ep})
  call('associate_'+ep+'.log',[str(SRC/'associate_render.py'),'--episode',ep])
 save(RUN/'association_all_complete.json',{'time':now(),'episodes':protocol['episodes']})
 save(RUN/'pipeline_state.json',{'time':now(),'stage':'videos'})
 call('videos.log',[str(SRC/'build_videos.py')])
 call('page.log',[str(SRC/'build_page.py')])
 save(RUN/'pipeline_complete.json',{'time':now(),'episodes':protocol['episodes'],'training_updates':0})
 save(RUN/'pipeline_state.json',{'time':now(),'stage':'complete'})
except Exception as e:
 save(RUN/'pipeline_failure.json',{'time':now(),'type':type(e).__name__,'detail':str(e)[:150],'frames':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
