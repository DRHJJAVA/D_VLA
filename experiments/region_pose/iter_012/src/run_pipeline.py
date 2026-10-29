from pathlib import Path
import subprocess,sys,json,time,hashlib,fcntl,traceback
B=Path(__file__).resolve().parents[1];R=B.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2))
with (B/'pipeline.lock').open('w') as lk:
 fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  for split in ['train','val']:
   dump(B/'stage_state.json',{'stage':split,'time':time.time()})
   if split=='val' and not (B/'pre_validation_freeze.json').exists():
    dump(B/'pre_validation_freeze.json',{'time':time.time(),'train_summary_sha256':sha(B/'train/summary.json'),'train_lock_sha256':sha(B/'train/locked.json'),'source_lock_sha256':sha(B/'source_locked.json'),'input_lock_sha256':sha(B/'input_locked.json')})
   with (B/(split+'.log')).open('a') as log:
    subprocess.run([sys.executable,str(B/'src/audit_regions.py'),split],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=R)
  dump(B/'stage_state.json',{'stage':'complete','time':time.time(),'training_performed':False})
 except Exception as e:
  dump(B/'blocked.json',{'error':str(e),'traceback':traceback.format_exc(),'time':time.time()});raise
