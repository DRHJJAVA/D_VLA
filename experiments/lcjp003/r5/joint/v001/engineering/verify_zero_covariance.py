import sys,pathlib,json,time,traceback
RUN=pathlib.Path("/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r5/joint/v001")
sys.path.insert(0,str(RUN/"src"))
import torch
from io_runtime import *
from train import previous_encoding
from losses import objective,response_targets
import r5_model
original_summary=r5_model.cloud_summary
def checked_summary(d,prob):
 out=original_summary(d,prob)
 if not torch.isfinite(out).all():
  grid=r5_model.world_grid(d)
  uv=torch.stack(torch.meshgrid(torch.arange(32,device=grid.device),torch.arange(32,device=grid.device),indexing="ij"),-1)[...,[1,0]]*8+3
  depth=d["depth"][:,uv[...,1],uv[...,0]]
  valid=(prob[:,uv[...,1],uv[...,0]]>.5)&(~d["robot"][:,uv[...,1],uv[...,0]])&torch.isfinite(depth)&(depth>0)
  x=grid[valid];center=x.mean(0);cov=(x-center).T@(x-center)/max(len(x),1)
  atomic("engineering/nonfinite_geometry_root.json",{"bad_indices":(~torch.isfinite(out)).nonzero().flatten().tolist(),"npoints":len(x),"xyz_finite":bool(torch.isfinite(x).all()),"cov":cov.tolist(),"cpu64_eigenvalues":torch.linalg.eigvalsh(cov.cpu().double()).tolist(),"gpu_repeat":torch.linalg.eigvalsh(cov).tolist(),"time":time.time()})
 return out
# The fixed helper is exercised by the unchanged exact training loop.
def main():
 import train,ast
 train.RUN=RUN/"engineering/exact_run"
 original_atomic=train.atomic
 def diagnostic_atomic(name,value):original_atomic("engineering/exact_probe/"+name,value)
 train.atomic=diagnostic_atomic
 train.checkpoint=lambda *args,**kwargs:None
 source=ast.parse((RUN/"src/train.py").read_text())
 fun=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=="main")
 class Bounded(ast.NodeTransformer):
  def visit_Constant(self,node):
   if node.value==3001:return ast.copy_location(ast.Constant(75),node)
   return node
 fun=Bounded().visit(fun)
 module=ast.fix_missing_locations(ast.Module(body=[fun],type_ignores=[]))
 exec(compile(module,str(RUN/"src/train.py"),"exec"),train.__dict__)
 train.main()
 atomic("engineering/exact_probe_complete.json",{"completed_through":74,"formal_checkpoint_changed":False,"time":time.time()})
if __name__=="__main__":
 try:main()
 except BaseException as ex:
  atomic("engineering/nonfinite_replay_failure.json",{"type":type(ex).__name__,"message":str(ex)[:200],"frames":[{"file":f.filename,"line":f.lineno} for f in traceback.extract_tb(ex.__traceback__)],"time":time.time()})
