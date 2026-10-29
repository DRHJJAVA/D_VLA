from common import *
import subprocess,time
def main():
 for _ in range(360):
  s=json.loads((BASE/'stage_state.json').read_text())['stage']
  if s=='inputs_ready':break
  if s=='failed':raise RuntimeError('preparation_failed')
  time.sleep(5)
 else:raise TimeoutError('preparation_not_ready')
 verify(json.loads((BASE/'source_locked.json').read_text()))
 log=open(BASE/'training.log','ab')
 p=subprocess.run(['/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python',str(BASE/'src/train.py')],cwd=ROOT,stdout=log,stderr=log)
 if p.returncode:raise RuntimeError('training_process_failed')
if __name__=='__main__':
 try:main()
 except Exception as e:
  if not (BASE/'failure.json').exists():failure(e)
  raise SystemExit(1)
