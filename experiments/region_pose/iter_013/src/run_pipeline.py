from pathlib import Path
import subprocess,sys,json,time,hashlib,fcntl,traceback
B=Path(__file__).resolve().parents[1];R=B.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def stage(name,args):
 dump(B/'stage_state.json',{'stage':name,'time':time.time()})
 with (B/'logs'/(name+'.log')).open('a') as log:
  subprocess.run([sys.executable]+[str(a) for a in args],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=R)
with (B/'pipeline.lock').open('w') as lk:
 fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  stage('targets_train',[B/'src/build_targets.py','train'])
  if not (B/'pre_validation_freeze.json').exists():dump(B/'pre_validation_freeze.json',{'time':time.time(),'train_summary':sha(B/'targets/train/summary.json'),'train_lock':sha(B/'targets/train/locked.json'),'source_lock':sha(B/'source_locked.json'),'input_lock':sha(B/'input_locked.json')})
  stage('targets_val',[B/'src/build_targets.py','val'])
  stage('samples',[B/'src/prepare_samples.py'])
  stage('preflight',[B/'src/check_inputs.py'])
  if not json.loads((B/'manifest.json').read_text())['training_gate_passed']:
   dump(B/'stage_state.json',{'stage':'insufficient_data_no_training','time':time.time()});sys.exit(0)
  stage('training',[B/'src/train.py'])
  dump(B/'stage_state.json',{'stage':'complete','time':time.time()})
 except Exception as e:
  dump(B/'blocked.json',{'error':str(e),'traceback':traceback.format_exc(),'time':time.time()});raise
