
import os
os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("PYOPENGL_PLATFORM","egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import sys,pathlib,json,ast,types,copy,time,hashlib,traceback,fcntl
import numpy as np,torch,h5py
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA')
RUN=ROOT/'experiments/lcjp003/r3/joint/v001'
OUT=RUN/'closed_loop_object'
OLD2=ROOT/'experiments/lcjp003/r2/v2/v001'
VENDOR=pathlib.Path('/home/drh/project/ECV-VLA/vendor/LIBERO/libero/libero')
sys.path[:0]=[str(RUN/'src'),str(RUN/'engineering'),str(OLD2/'src')]
import common
sys.path[:0]=[str(RUN/'src'),str(RUN/'engineering')]
import geometry
from fps_fast import fps
geometry.fps=fps
from network import JointNetwork
def save(n,d):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2));q.replace(p)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def legacy_adapter():
 p=ROOT/'experiments/lcjp003/p1/v001/closed_loop/adapters.py'
 t=ast.parse(p.read_text());body=[]
 for n in t.body:
  if isinstance(n,ast.ImportFrom) and n.module in ['common','features','model']:continue
  if isinstance(n,ast.ClassDef) and n.name=='LiveInput':continue
  body.append(n)
 m=types.ModuleType('r3_render_reuse');m.__file__=str(p)
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(p),'exec'),m.__dict__)
 return m
def input_builder(work,words):
 p=OLD2/'src/inputs.py';t=ast.parse(p.read_text())
 for n in ast.walk(t):
  if isinstance(n,ast.FunctionDef) and n.name=='__init__':
   n.body=[ast.Pass()]
  if isinstance(n,ast.FunctionDef) and n.name=='_cache_frame':
   n.body=[x for x in n.body if x.lineno<=95 and not 88<=x.lineno<=90]
   # No priors or future supervision: retain only the identical causal history builder.
   n.body+=ast.parse("return {'h':torch.from_numpy(h).float(),'words':self.fixed_words,'target':torch.zeros(16,6),'grip':torch.zeros(16,dtype=torch.long)}").body
 m=types.ModuleType('r3_live_inputs');m.__file__=str(p)
 exec(compile(ast.fix_missing_locations(t),str(p),'exec'),m.__dict__)
 m.RUN=work/'r2';m.RUN.mkdir(parents=True,exist_ok=True)
 (m.RUN/'cache').mkdir(parents=True,exist_ok=True)
 m.atomic_torch=lambda *args:None
 original=m.legacy()['make_data']
 source=pathlib.Path(original.__code__.co_filename)
 tree=ast.parse(source.read_text())
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_data')
 class LiveHistory(ast.NodeTransformer):
  def visit_Assign(self,n):
   if any(isinstance(x,ast.Name) and x.id=='base' for x in n.targets):
    return ast.copy_location(ast.parse("base=LIVE_OLD/'cache'").body[0],n)
   return self.generic_visit(n)
  def visit_With(self,n):
   if any(isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store) and x.id=='act' for x in ast.walk(n)):
    return ast.copy_location(ast.parse("act=np.zeros((16,7),np.float32)").body[0],n)
   return self.generic_visit(n)
 fn=LiveHistory().visit(fn)
 ns=dict(original.__globals__);ns['LIVE_OLD']=work;ns['save']=lambda *args,**kwargs:None
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),ns)
 m.legacy=lambda:{'make_data':ns['make_data']}
 obj=m.Inputs(None);obj.old=work;obj.fixed_words=words;obj.sam=None;obj.prior=None
 norm=json.loads((OLD2/'normalization.json').read_text())
 obj.mu=np.asarray(norm['mean'],np.float32);obj.sd=np.asarray(norm['std'],np.float32)
 return obj,m
def live_geometry(work):
 p=RUN/'src/geometry.py';t=ast.parse(p.read_text())
 class Redirect(ast.NodeTransformer):
  def visit_Constant(self,n):
   if isinstance(n.value,str) and n.value=='experiments/contact_joint_object/cache/rgbd':
    return ast.copy_location(ast.Constant(str(work/'cache/rgbd')),n)
   return n
 t=Redirect().visit(t);m=types.ModuleType('r3_live_geometry');m.__file__=str(p)
 exec(compile(ast.fix_missing_locations(t),str(p),'exec'),m.__dict__)
 m.OLD=work/'r2';m.fps=fps
 return m
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'executor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ck=RUN/'checkpoints/step3000.pt'
 assert sha(ck)=='9a84bf5b21c6437f567c1208e3f876e61a32d73e53c5475a155b712ece7b8446'
 adapter=legacy_adapter()
 rows=json.loads((ROOT/'experiments/contact_joint_object/manifests/episodes.json').read_text())['episodes']
 taskfiles=sorted((VENDOR/'init_files/libero_object').glob('*.pruned_init'))
 task=taskfiles[0].stem
 row=next(e for e in rows if e['id']==task+'__demo_0')
 work=OUT/('preflight_'+str(time.time_ns()))
 save('state.json',{'status':'building_live_adapter','task':task})
 render=adapter.LiveRender(row['id'],work,capacity=505)
 # Engineering gate starts from the already-authorized training initial state.
 words=torch.load(OLD2/'cache'/(row['task']+'_words.pt'),map_location='cpu',weights_only=False)
 builder,mod=input_builder(work,words)
 builder.actions={row['id']:np.zeros((505,7),np.float32)}
 row=dict(row);row['frames']=505
 live=live_geometry(work)
 torch.backends.cudnn.allow_tf32=True
 proposals=adapter.LiveProposals('cuda')
 # Legacy proposal masks are unused: the R2 32x32 generator below supplies all candidates.
 save('state.json',{'status':'generating_initial_observation','task':task})
 proposals.frame(render,0)
 r3vis=sys.modules.get('visibility')
 sys.modules['visibility']=adapter.module_from(ROOT/'experiments/lcjp003/src/visibility.py','visibility')
 sys.modules['data']=adapter.module_from(ROOT/'experiments/lcjp003/src/data.py','data')
 sys.modules['visibility']=r3vis
 d0=builder._cache_frame(row,0)
 cache=work/'r2/cache'/row['id'];cache.mkdir(parents=True,exist_ok=True)
 torch.save(d0,cache/'0.pt')
 d,y,objects,meta=live.policy_input(row,0)
 save('live_input_schema.json',{'fields':{k:list(v.shape) for k,v in d.items()},'objects':len(objects),'history':list(d['h'].shape),'grid':32,'teacher_used':False})
 save('state.json',{'status':'initial_live_input_built','objects':len(objects)})
 render.close()
if __name__=='__main__':
 try:main()
 except Exception as e:
  save('failure.json',{'type':type(e).__name__,'summary':str(e)[:500],'frames':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
