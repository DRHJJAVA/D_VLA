import os,sys,pathlib,json,time,traceback
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,torch
N=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(N/'src'))
import geometry as g,train
from fps_fast import fps
torch.set_num_threads(2)
try:
 old=g.fps;cases=[]
 es=[e for e in json.loads((N/'episodes.json').read_text()) if e['episode']=='demo_0' and e['v2_split']=='train'][:4]
 for e in es:
  for frame in [13,38]:
   g.fps=old;t=time.time();a,y=train.load_window(e,frame);t0=time.time()-t
   g.fps=fps;t=time.time();b,z=train.load_window(e,frame);t1=time.time()-t
   for k in a:assert torch.equal(a[k],b[k]),k
   for k in y:
    if isinstance(y[k],torch.Tensor):assert torch.equal(y[k],z[k]),k
   cases.append(dict(episode=e['id'],frame=frame,old_seconds=t0,new_seconds=t1,input_exact=True,target_exact=True))
 r=np.random.default_rng(8)
 for x in [np.zeros((32,3),np.float32),np.ones((128,3),np.float32),r.normal(size=(4096,3)).astype(np.float32)]:
  for k in [8,64,2048]:assert np.array_equal(old(x,k),fps(x,k))
 (N/'engineering/fps_validation.json').write_text(json.dumps(dict(passed=True,cases=cases,ties_and_padding_exact=True),indent=2))
except Exception as ex:
 (N/'engineering/fps_validation_failure.json').write_text(json.dumps(dict(error_type=type(ex).__name__,message=str(ex)[:100],frames=[dict(file=f.filename,line=f.lineno) for f in traceback.extract_tb(ex.__traceback__)])))
