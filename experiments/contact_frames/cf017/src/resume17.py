import pathlib,json,time,subprocess,sys,os,traceback,datetime
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf017')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def main():
 deadline=time.time()+900;case=B/'cases/two_objects_demo_0'
 while time.time()<deadline:
  state=json.loads((case/'stage_state.json').read_text())['stage']
  if state=='prepared':break
  if state=='failed':raise RuntimeError('long_video_retry_failed')
  time.sleep(3)
 else:raise TimeoutError('long_video_retry_timeout')
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MUJOCO_GL='egl')
 with open(B/'features.log','w') as log:
  rc=subprocess.run([sys.executable,'-u',str(B/'src/prepare_features.py'),'review'],cwd=B,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 if rc:raise RuntimeError('features_failed')
 with open(B/'predict.log','w') as log:
  rc=subprocess.run([sys.executable,'-u',str(B/'src/predict17.py')],cwd=B,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 if rc:raise RuntimeError('predict_failed')
if __name__=='__main__':
 try:main()
 except Exception as e:
  (B/'failure_resume.json').write_text(json.dumps(dict(time=now(),type=type(e).__name__,message=str(e),stack=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(e.__traceback__)]),indent=2));raise SystemExit(1)
