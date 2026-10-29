import os,sys,pathlib,json,time,traceback
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,torch,numba
N=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(N/'src'))
import geometry as g,train
torch.set_num_threads(2)
try:
 original=g.fps;from fps_fast import fps as compiled
 x=np.random.default_rng(20261028).normal(size=(4096,3)).astype(np.float32)
 t=time.time();a=original(x,2048);b=compiled(x,2048);assert np.array_equal(a,b)
 episodes=json.loads((N/'episodes.json').read_text());e=episodes[0]
 start=time.time();d,y=train.load_window(e,67);old=time.time()-start
 g.fps=compiled
 start=time.time();dd,yy=train.load_window(e,67);new=time.time()-start
 for k in d: assert torch.equal(d[k],dd[k]),k
 for k in y:
  if isinstance(y[k],torch.Tensor):assert torch.equal(y[k],yy[k]),k
 out=dict(passed=True,baseline_seconds=old,compiled_seconds=new,speedup=old/new,inputs_exact=True,targets_exact=True,numba_version=numba.__version__)
 (N/'engineering/fps_check.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
except Exception as ex:
 out=dict(error_type=type(ex).__name__,message=str(ex).splitlines()[0][:120],frames=[dict(file=f.filename,line=f.lineno) for f in traceback.extract_tb(ex.__traceback__)])
 (N/'engineering/fps_failure.json').write_text(json.dumps(out));print(json.dumps(out))
