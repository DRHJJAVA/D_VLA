from common import *
import subprocess,time
def main():
 for _ in range(480):
  s=json.loads((BASE/'stage_state.json').read_text())['stage']
  if s=='training_complete_pending_audit':
   verify(json.loads((BASE/'completion_source_locked.json').read_text()))
   with open(BASE/'finalize.log','ab') as log:
    result=subprocess.run(['/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python',str(BASE/'src/finalize.py')],cwd=ROOT,stdout=log,stderr=log)
   return result.returncode
  if s in ['failed','complete_dynamics_control']:return 0
  time.sleep(15)
 save(BASE/'completion_watch_timeout.json',dict(time=now(),message='watch timed out; inspect stage before continuing'))
 return 1
if __name__=='__main__':raise SystemExit(main())
