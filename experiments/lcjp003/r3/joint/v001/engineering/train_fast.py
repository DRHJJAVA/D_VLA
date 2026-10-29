import os,sys,pathlib,json,hashlib,time
N=pathlib.Path(__file__).resolve().parents[1]
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(N/'src'))
import geometry,train,numpy as np
from fps_fast import fps
assert json.loads((N/'engineering/fps_validation.json').read_text())['passed']
lock=json.loads((N/'engineering/source_lock.json').read_text())
for name,h in lock['files'].items():assert hashlib.sha256((N/name).read_bytes()).hexdigest()==h,name
fps(np.zeros((32,3),np.float32),8)
geometry.fps=fps
(N/'engineering/resumed.json').write_text(json.dumps({'pid':os.getpid(),'from_step':json.loads((N/'engineering/handoff.json').read_text())['step'],'optimizer_rng_restore':'original train.main checkpoint resume path','at':time.time()}))
train.main()
