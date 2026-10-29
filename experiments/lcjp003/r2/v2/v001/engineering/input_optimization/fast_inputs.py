import sys,os,ast,copy,threading,hashlib,json,collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,"/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r2/v2/v001/src")
from common import *
import inputs
WORK=RUN/"engineering/input_optimization"
class ProcessorProxy:
 def __init__(self,processor):
  self.original=processor;self.local=threading.local();self.cache=collections.OrderedDict();self.lock=threading.Lock()
 def get(self):
  if not hasattr(self.local,"value"):
   p=copy.deepcopy(self.original);resize=p.image_processor.resize
   def memo(image,*args,**kwargs):
    a=np.asarray(image);key=(a.shape,str(a.dtype),hashlib.sha256(a.tobytes()).digest(),repr(args),repr(sorted(kwargs.items())))
    with self.lock:
     out=self.cache.get(key)
     if out is not None:self.cache.move_to_end(key);return out.copy()
    out=resize(image,*args,**kwargs)
    if isinstance(out,np.ndarray):
     with self.lock:
      self.cache[key]=out.copy()
      while len(self.cache)>128:self.cache.popitem(last=False)
    return out
   p.image_processor.resize=memo;self.local.value=p
  return self.local.value
 def __call__(self,*a,**kw):return self.get()(*a,**kw)
 def __getattr__(self,key):return getattr(self.get(),key)
def batched_query(original,pool):
 tree=ast.parse((V1/"v1_check.py").read_text())
 q=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="query")
 loop=next(n for n in q.body if isinstance(n,ast.For))
 cut=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=="inp" for v in n.targets))
 body=copy.deepcopy(loop.body[:cut+1]);assign=body[-1]
 assert isinstance(assign.value,ast.Call) and isinstance(assign.value.func,ast.Attribute) and assign.value.func.attr=="to"
 move=copy.deepcopy(assign.value);move.func.value=ast.Name(id="inp",ctx=ast.Load())
 assign.value=assign.value.func.value
 def definition(name,args,body):
  return ast.FunctionDef(name=name,args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=a) for a in args],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=body,decorator_list=[])
 prepare=definition("prepare_batch",["entries","start"],body+[ast.Return(ast.Name(id="inp",ctx=ast.Load()))])
 gpu=definition("execute_batch",["inp"],[ast.Assign(targets=[ast.Name(id="inp",ctx=ast.Store())],value=move)]+copy.deepcopy(loop.body[cut+1:-1])+[ast.Return(ast.Name(id="probabilities",ctx=ast.Load()))])
 g=original.__globals__.copy();g["processor"]=ProcessorProxy(g["processor"])
 module=ast.fix_missing_locations(ast.Module(body=[prepare,gpu],type_ignores=[]))
 exec(compile(module,"<same_batches_parallel_preprocess>","exec"),g)
 def query(entries):
  # Preserve every original model batch of eight, padding and forward order.
  futures=[pool.submit(g["prepare_batch"],entries,start) for start in range(0,len(entries),8)]
  result=[]
  for f in futures:result.extend(g["execute_batch"](f.result()))
  return result
 return query
class FastInputs(inputs.Inputs):
 def __init__(self,expert,rootdir=RUN):
  super().__init__(expert);self.rootdir=Path(rootdir);(self.rootdir/"cache").mkdir(parents=True,exist_ok=True)
  self.pool=ThreadPoolExecutor(max_workers=4,thread_name_prefix="vlm_cpu")
 def load_prior(self):
  if self.prior is not None:return
  super().load_prior();self.prior=batched_query(self.prior,self.pool)
 def put_bundle(self,name,d):
  p=self.rootdir/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(".tmp."+str(os.getpid()))
  torch.save(d,tmp);tmp.replace(p)
# Reuse the input builder unchanged except output locations and atomic writer.
tree=ast.parse((RUN/"src/inputs.py").read_text())
f=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="_cache_frame"))
class Redirect(ast.NodeTransformer):
 def visit_Name(self,node):
  if node.id=="RUN":return ast.copy_location(ast.Attribute(value=ast.Name(id="self",ctx=ast.Load()),attr="rootdir",ctx=node.ctx),node)
  if node.id=="atomic_torch":return ast.copy_location(ast.Attribute(value=ast.Name(id="self",ctx=ast.Load()),attr="put_bundle",ctx=node.ctx),node)
  return node
 def visit_Constant(self,node):
  if node.value=="observations.h5":
   return ast.copy_location(ast.BinOp(left=ast.BinOp(left=ast.Constant("observations_"),op=ast.Add(),right=ast.Call(func=ast.Name(id="str",ctx=ast.Load()),args=[ast.Name(id="t",ctx=ast.Load())],keywords=[])),op=ast.Add(),right=ast.Constant(".h5")),node)
  return node
f=Redirect().visit(f);g=inputs.__dict__.copy()
exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),"<same_input_builder_isolated_writes>","exec"),g)
FastInputs._cache_frame=g["_cache_frame"]
