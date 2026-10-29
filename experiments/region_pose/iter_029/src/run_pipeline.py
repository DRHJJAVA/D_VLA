from pathlib import Path
import subprocess,sys,json,time,hashlib,fcntl,traceback
B=Path(__file__).resolve().parents[1];R=B.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2))
def stage(name,script):
 dump(B/'stage_state.json',{'stage':name,'time':time.time()})
 with (B/'logs'/(name+'.log')).open('a') as log:
  subprocess.run([sys.executable,'-u',str(B/'src'/script)],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
with (B/'pipeline.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  for fn in ['source_locked.json','input_locked.json']:
   for path,h in json.loads((B/fn).read_text()).items():assert sha(Path(path) if Path(path).is_absolute() else R/path)==h,path
  if not (B/'manifest.json').exists():stage('samples','prepare_samples.py')
  stage('preflight','check_inputs.py')
  if not json.loads((B/'manifest.json').read_text())['training_gate_passed']:
   dump(B/'stage_state.json',{'stage':'insufficient_data_no_training','time':time.time()});sys.exit(0)
  stage('training','train.py')
  dump(B/'stage_state.json',{'stage':'training_complete_pending_audit','time':time.time()})
 except Exception as e:
  dump(B/'blocked.json',{'error':str(e),'traceback':traceback.format_exc(),'time':time.time()});raise
