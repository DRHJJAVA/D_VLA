from pathlib import Path
import subprocess,sys,json,time,hashlib,fcntl
P=Path('/mnt/nvme1/drh/D_VLA');O=Path(__file__).resolve().parents[1]
def state(stage,**kw):(O/'stage_state.json').write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2))
def verify():
 for name in ['source_locked.json','input_locked.json','control_locked.json']:
  for p,h in json.loads((O/name).read_text()).items():assert hashlib.sha256((P/p).read_bytes()).hexdigest()==h,p
def main():
 lock=open(O/'pipeline.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (O/'stage_state.json').exists() and json.loads((O/'stage_state.json').read_text())['stage']=='trained':return
 try:
  verify()
  for stage,script in [('preflight','preflight.py'),('training','train.py')]:
   state(stage);subprocess.run([sys.executable,'-u',str(O/'src'/script)],cwd=P,check=True)
  verify();state('trained');print('TRAINED_AWAITING_COMPLETION_AUDIT',flush=True)
 except BaseException as ex:
  state('failed',error=str(ex));raise
if __name__=='__main__':main()
