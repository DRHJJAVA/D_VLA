from pathlib import Path
import subprocess,sys,json,os,time,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
B=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object')
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(name,obj):
 p=B/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)
def main():
 for n in ['network_smoke.json','online_smoke.json']:
  assert json.loads((B/'manifests'/n).read_text())['ok'],n+' did not pass'
 assert not (B/'runs/v001/initial.pt').exists(),'Formal run exists; do not restart blindly'
 for source in (B/'src').glob('*.py'):compile(source.read_text(),str(source),'exec')
 env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
 with (B/'logs/train.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,str(B/'src/train.py')],stdout=log,stderr=log,env=env)
  save('pipeline_state.json',{'time':now(),'stage':'training_worker_launched_waiting_for_full_observations','pipeline_pid':os.getpid(),'training_pid':proc.pid})
  rc=proc.wait()
 if rc or (B/'training_failure.json').exists():raise RuntimeError('Training worker failed; inspect structured failure, no automatic repeated run')
 assert (B/'runs/v001/frozen_for_test.json').exists()
 save('pipeline_state.json',{'time':now(),'stage':'frozen_test_and_review','pipeline_pid':os.getpid()})
 with (B/'logs/evaluate_frozen.log').open('w') as log:
  proc=subprocess.Popen([sys.executable,str(B/'src/evaluate_frozen.py')],stdout=log,stderr=log,env=env);rc=proc.wait()
 if rc or (B/'test_failure.json').exists():raise RuntimeError('Frozen inference/review failed; inspect before resume')
 assert (B/'experiment_complete.json').exists()
 save('pipeline_state.json',{'time':now(),'stage':'complete_pending_user_review','pipeline_pid':os.getpid(),'model_prediction_generated':True,'human_acceptance':'pending'})
except_placeholder=0
if __name__=='__main__':
 try:main()
 except Exception as e:
  (B/'logs/pipeline_traceback.log').write_text(traceback.format_exc());save('pipeline_failure.json',{'time':now(),'exception':type(e).__name__,'detail':str(e)[:300]})
