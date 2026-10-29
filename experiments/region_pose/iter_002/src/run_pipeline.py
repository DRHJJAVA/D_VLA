from pathlib import Path
import subprocess,sys,os,json,time,hashlib,fcntl
I=Path(__file__).resolve().parents[1];R=Path('/mnt/nvme1/drh/D_VLA');PY='/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python'
def lockfusion():
 O=I/'evidence/fusion';files=[p for p in O.iterdir() if p.is_file()]+[I/'src/build_fusion.py']
 if not (O/'locked.json').exists():(O/'locked.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
def main():
 lockfile=(I/'pipeline.lock').open('a');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env['PYTHONPATH']=str(I/'src')+':'+str(R/'src/v3')+':'+str(R/'src');env['OPENBLAS_NUM_THREADS']='4';env['OMP_NUM_THREADS']='4';env['MUJOCO_GL']='egl'
 stages=[('motion','build_motion.py',I/'evidence/motion/summary.json'),('anchors','select_anchors.py',I/'evidence/motion/review_frames.json'),('reference','build_reference.py',I/'evidence/reference_cloud/locked.json'),('region','build_region.py',I/'evidence/region/locked.json'),('fusion','build_fusion.py',I/'evidence/fusion/summary.json'),('samples','prepare_samples.py',I/'manifest.json'),('training','train.py',I/'summary.json')]
 src={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (I/'src').glob('*.py')}
 old=I/'source_locked.json'
 if old.exists():assert json.loads(old.read_text())==src,'source changed since pipeline launch: review version'
 else:old.write_text(json.dumps(src,indent=2))
 for name,script,marker in stages:
  if name=='samples':lockfusion()
  if marker.exists():
   print('REUSE COMPLETE '+name,flush=True);continue
  state=dict(stage=name,status='running',pid=os.getpid(),started_unix=time.time(),script=script)
  (I/'stage_state.json').write_text(json.dumps(state,indent=2))
  log=I/'logs'/(name+'_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.log')
  print('START '+name+' '+str(log),flush=True)
  with log.open('x') as f:result=subprocess.run([PY,'-u',str(I/'src'/script)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode or not marker.exists():
   state.update(status='failed',returncode=result.returncode,log=str(log));(I/'stage_state.json').write_text(json.dumps(state,indent=2));(I/'blocked.json').write_text(json.dumps(state,indent=2));raise RuntimeError('stage failed, preserves outputs: '+name)
  print('DONE '+name,flush=True)
 (I/'stage_state.json').write_text(json.dumps(dict(status='complete',finished_unix=time.time()),indent=2));print('ITERATION002_COMPLETE',flush=True)
if __name__=='__main__':main()
