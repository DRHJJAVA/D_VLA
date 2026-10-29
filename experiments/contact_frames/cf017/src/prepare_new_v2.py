import os,sys,json,pathlib,subprocess,datetime,traceback,hashlib,time,ast
os.environ.setdefault('MUJOCO_GL','egl')
R=pathlib.Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/contact_frames/cf017'
sys.path.insert(0,str(R/'src/v3'))
import numpy as np,h5py,cv2
from scipy.spatial import cKDTree
from stage2_probe_motion_fingerpads import tc,cloud
from stage2_robot_only_depth import RobotDepth
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def main():
 cfg=json.loads((B/'config.json').read_text());manifest=[]
 # Each case fixed before reading outputs. Process smaller tasks while two-object file downloads.
 cases=sorted(cfg['cases'],key=lambda c:c['id']=='two_objects_demo_0')
 for c in cases:
  ep=c['id'];root=B/'cases'/ep;root.mkdir(exist_ok=True);(root/'artifacts').mkdir(exist_ok=True);(root/'src').mkdir(exist_ok=True)
  raw=B/'raw'/c['name'];rgbd=B/'rgbd'/f'{ep}.h5'
  save(B/'stage_state.json',dict(time=now(),stage='waiting_for_official_download',episode=ep))
  deadline=time.time()+1200
  while not raw.exists():
   if time.time()>deadline:raise TimeoutError('raw_download_not_ready_'+ep)
   time.sleep(3)
  save(B/'stage_state.json',dict(time=now(),stage='rendering_official_rgbd',episode=ep))
  cmd=[sys.executable,str(B/'src/render_rgbd.py'),'--input',str(raw),'--output',str(rgbd),'--libero-root','/home/drh/project/ECV-VLA/vendor/LIBERO','--bddl-subdir',c['suite'],'--episode','demo_0','--size','448']
  env=dict(os.environ,PYTHONPATH=str(R/'src'),MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
  while rgbd.with_suffix('.part').exists() and not rgbd.exists():
   time.sleep(3)
  if not rgbd.exists():
   with open(root/'render.log','w') as log:rc=subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
   if rc:raise RuntimeError('render_exit_'+str(rc)+'_'+ep)
  with h5py.File(raw,'r') as f:
   e=f['data/demo_0'];qraw=e['obs/gripper_states'][:];N=len(e['states'])
  with h5py.File(rgbd,'r') as f:
   g=f['agentview'];D=g['depth_m'][0];K=g['K'][0];C=g['camera_to_world'][0]
   H=f['robot0_eye_in_hand/camera_to_world'][:]@np.linalg.inv(tc())
   q=np.empty((N,2),np.float32);q[0]=np.nan;q[1:]=qraw[:N-1]
   X=cloud(D,K,C);centers=(H[:,:3,:3]@np.array([0,0,.0934]))+H[:,:3,3]
   # Valid original measured surface only. Spatial ROI selects context, not contact.
   flat=X.reshape(-1,3);dist=cKDTree(centers[1:]).query(flat)[0].reshape(D.shape)
   robot=RobotDepth(D.shape[0]);rd=robot.render(qraw[0],np.linalg.inv(H[0])@C,K);robot.close()
   mask=np.isfinite(D)&(D>.01)&(D<2)&(dist<=.12)&((rd>1)|(D<rd-.002))
   yy,xx=np.where(mask);assert len(xx)>=8
   pixel=np.stack([xx,yy],1).astype(np.int64)
  motion=root/'motion.npz';component=root/'component.npz'
  np.savez_compressed(motion,hand_to_world=H,measured_finger_q=q)
  np.savez_compressed(component,pixel=pixel,query_frame=np.array(0),no_simulator_truth=True)
  conf=json.loads((R/'experiments/contact_frames/cf001/config.json').read_text())
  conf.update(episode=ep,rgbd=str(rgbd),motion=str(motion),component=str(component))
  save(root/'config.json',conf)
  manifest.append(dict(id=ep,source_path=str(raw),source_episode='demo_0',timestamps_seconds=(np.arange(N)/20.).tolist()))
  mf=B/'original_sequences.jsonl';mf.write_text(''.join(json.dumps(x)+'\n' for x in manifest))
  src=(R/'experiments/contact_frames/cf001/src/prepare.py').read_text().replace('data/v3/manifests/original_sequences.jsonl',str(mf))
  (root/'src/prepare.py').write_text(src)
  cm=(R/'experiments/contact_frames/cf001/src/common.py').read_text().replace("experiments/contact_frames/cf001",str(root))
  (root/'src/common.py').write_text(cm)
  save(root/'adapter_audit.json',dict(time=now(),frames=N,initial_real_surface_pixels=len(pixel),roi_radius_m=.12,uses_full_actual_hand_trajectory_for_offline_context=True,query_frame=0,raw_observation_alignment='t>=1 from official obs t-1; t0 unknown',labels_used=False))
  save(B/'stage_state.json',dict(time=now(),stage='preparing_frozen_input',episode=ep))
  with open(root/'prepare.log','w') as log:rc=subprocess.run([sys.executable,'-u',str(root/'src/prepare.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  if rc:raise RuntimeError('prepare_exit_'+str(rc)+'_'+ep)
 save(B/'stage_state.json',dict(time=now(),stage='inputs_prepared'))
 with open(B/'features.log','w') as log:rc=subprocess.run([sys.executable,'-u',str(B/'src/prepare_features.py'),'review'],cwd=B,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 if rc:raise RuntimeError('motion_features_exit_'+str(rc))
 save(B/'stage_state.json',dict(time=now(),stage='ready_for_prediction'))
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(B/'failure_prepare.json',dict(time=now(),type=type(e).__name__,message=str(e),stack=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(e.__traceback__)]));save(B/'stage_state.json',dict(time=now(),stage='prepare_failed'));raise SystemExit(1)
