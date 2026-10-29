from pathlib import Path
import json,hashlib,subprocess,sys,time,fcntl,traceback
from audit_source_support import audit,dump,sha
R=Path('/mnt/nvme1/drh/D_VLA');B=Path(__file__).resolve().parents[1];P=B.parent
def verify(file,base):
 for path,h in json.loads(file.read_text()).items():assert sha(base/path)==h,path
def main():
 verify(B/'source_locked.json',R);verify(B/'input_locked.json',R)
 freeze=B/'pre_validation_freeze.json'
 if not freeze.exists():
  train=audit('train')
  dump(freeze,{'time':time.time(),'source_lock_sha256':sha(B/'source_locked.json'),'input_lock_sha256':sha(B/'input_locked.json'),'train_audit_sha256':sha(B/'audits/train_source_support.json'),'rules':'014 unchanged; >=3 unique positive reference points and >=16 unique negative reference points','train_eligible_queries':sum(x['eligible'] for x in train['rows'])})
 else:
  f=json.loads(freeze.read_text());assert sha(B/'source_locked.json')==f['source_lock_sha256'];assert sha(B/'audits/train_source_support.json')==f['train_audit_sha256']
 dump(B/'stage_state.json',{'stage':'validation','time':time.time()})
 if not (B/'val/locked.json').exists():
  subprocess.run([sys.executable,str(B/'src/diagnose_validation.py')],check=True,cwd=R)
 verify(B/'val/locked.json',B/'val')
 val=audit('val')
 verify(B/'source_locked.json',R);verify(B/'input_locked.json',R)
 dump(B/'summary.json',{'train':json.loads((B/'audits/train_source_support.json').read_text())['offsets'],'val':val['offsets'],'no_new_training':True})
 dump(B/'stage_state.json',{'stage':'complete','time':time.time()})
 print('complete',flush=True)
if __name__=='__main__':
 with (B/'pipeline.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:main()
  except Exception as e:dump(B/'blocked.json',{'error':str(e),'traceback':traceback.format_exc()});raise
