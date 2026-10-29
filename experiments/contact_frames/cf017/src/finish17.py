import pathlib,json,time,subprocess,sys,os,traceback,datetime
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf017')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def main():
 deadline=time.time()+1200
 while time.time()<deadline:
  s=json.loads((B/'stage_state.json').read_text()).get('stage','')
  if s=='ready_for_prediction':break
  if s=='prepare_failed':raise RuntimeError('input_preparation_failed')
  time.sleep(3)
 else:raise TimeoutError('input_preparation_not_ready')
 with open(B/'predict.log','w') as log:
  p=subprocess.run([sys.executable,'-u',str(B/'src/predict17.py')],cwd=B,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError('prediction_child_failed')
if __name__=='__main__':
 try:main()
 except Exception as e:
  (B/'failure_finish.json').write_text(json.dumps(dict(time=now(),type=type(e).__name__,message=str(e),stack=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(e.__traceback__)]),indent=2));raise SystemExit(1)
