from common12 import *
import time,subprocess
def main():
 for _ in range(480):
  if (BASE/'failure.json').exists() or (BASE/'baseline_failure.json').exists():return 1
  if (BASE/'completed_locked.json').exists():return 0
  if (BASE/'training_complete.json').exists() and (BASE/'baseline_complete.json').exists():
   verify(json.loads((BASE/'completion_source_locked.json').read_text()))
   with open(BASE/'finalize.log','ab') as log:
    p=subprocess.run(['/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python',str(BASE/'src/finalize.py')],cwd=ROOT,stdout=log,stderr=log)
   return p.returncode
  time.sleep(15)
 save(BASE/'completion_watch_timeout.json',dict(time=now()))
 return 1
if __name__=='__main__':raise SystemExit(main())
