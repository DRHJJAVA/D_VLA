from pathlib import Path
import subprocess,sys,os,json,time,hashlib,fcntl
I=Path(__file__).resolve().parents[1];R=Path('/mnt/nvme1/drh/D_VLA');PY='/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python'
def main():
 lock=(I/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env.update(PYTHONPATH=str(I/'src')+':'+str(R/'src/v3')+':'+str(R/'src'),MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
 src={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (I/'src').glob('*.py')}
 if (I/'source_locked.json').exists():assert src==json.loads((I/'source_locked.json').read_text())
 else:(I/'source_locked.json').write_text(json.dumps(src,indent=2))
 for name,script,marker in [('features','prepare_dense_rgb.py','manifest.json'),('training','train.py','summary.json')]:
  if (I/marker).exists():print('REUSE '+name,flush=True);continue
  state=dict(stage=name,status='running',pid=os.getpid(),started_unix=time.time());(I/'stage_state.json').write_text(json.dumps(state,indent=2))
  log=I/'logs'/(name+'_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.log')
  print('START '+name,flush=True)
  with log.open('x') as f:r=subprocess.run([PY,'-u',str(I/'src'/script)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode or not (I/marker).exists():
   state.update(status='failed',returncode=r.returncode,log=str(log));(I/'stage_state.json').write_text(json.dumps(state,indent=2));(I/'blocked.json').write_text(json.dumps(state,indent=2));raise RuntimeError(name+' failed; retain all existing outputs')
  print('DONE '+name,flush=True)
 (I/'stage_state.json').write_text(json.dumps(dict(status='complete',finished_unix=time.time()),indent=2))
 print('ITERATION004_COMPLETE',flush=True)
if __name__=='__main__':main()
