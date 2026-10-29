from pathlib import Path
import subprocess,sys,json,time,hashlib,fcntl,traceback
B=Path(__file__).resolve().parents[1];R=B.parents[2]
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
with (B/'pipeline.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  for fn in ['source_locked.json','input_locked.json']:
   for path,h in json.loads((B/fn).read_text()).items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==h,path
  for stage,script in [('preflight','preflight.py'),('training','train.py')]:
   dump(B/'stage_state.json',{'stage':stage,'time':time.time()})
   with (B/'logs'/(stage+'.log')).open('wb') as f:
    p=subprocess.run([sys.executable,'-u',str(B/'src'/script)],cwd=R,stdout=f,stderr=f)
   if p.returncode:raise RuntimeError(stage+' failed; inspect remote sanitized error')
  dump(B/'stage_state.json',{'stage':'training_complete_pending_audit','time':time.time()})
 except Exception as e:
  (B/'logs/pipeline_traceback_remote_only.txt').write_text(traceback.format_exc())
  dump(B/'blocked.json',{'error_type':type(e).__name__,'message':str(e),'time':time.time()})
  raise SystemExit(1)
