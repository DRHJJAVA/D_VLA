import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS","2")
import sys,pathlib,json,random,hashlib,importlib.util,time
import numpy as np
import torch
ROOT=pathlib.Path("/mnt/nvme1/drh/D_VLA")
RUN=ROOT/"experiments/lcjp003/r5/joint/v001"
OLD=ROOT/"experiments/lcjp003/r4/joint/v001"
CFG=json.loads((RUN/"config.json").read_text())
def atomic(name,x):
 p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(".tmp")
 t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def setup():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def build(device="cuda"):
 spec=importlib.util.spec_from_file_location("r4_scaffold_runtime",OLD/"src/runtime.py")
 m=importlib.util.module_from_spec(spec);oldpath=list(sys.path);sys.path.insert(0,str(OLD/"src"))
 try:spec.loader.exec_module(m);scaffold=m.build(device)
 finally:sys.path[:]=oldpath
 from r5_model import R5Network
 model=R5Network(scaffold).float().to(device)
 del scaffold
 return model
def move(x,device="cuda"):
 if torch.is_tensor(x):return x.to(device,non_blocking=True)
 if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
 if isinstance(x,list):return [move(v,device) for v in x]
 return x
def rng_state():return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}
def restore_rng(s):
 random.setstate(s["python"]);np.random.set_state(s["numpy"]);torch.set_rng_state(s["torch"].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in s["cuda"]])
def optimizer(m):return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
def checkpoint(model,opt,step,extra=None):
 d={"model":model.state_dict(),"optimizer":opt.state_dict(),"step":step,"rng":rng_state(),"source_lock":digest(RUN/"source_lock.json"),"config_hash":digest(RUN/"config.json"),"extra":extra or {}}
 p=RUN/"checkpoints/last.pt";p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(".tmp");torch.save(d,tmp);tmp.replace(p)
 if step in (0,20,50) or step%250==0:
  import shutil
  shutil.copy2(p,p.parent/("step%04d.pt"%step))
def load_record(rec):
 eid=rec["episode"];t=int(rec["frame"])
 q=torch.load(OLD/"cache"/eid/(str(t)+".pt"),map_location="cpu",weights_only=False)
 y=torch.load(RUN/"teacher"/eid/(str(t)+".pt"),map_location="cpu",weights_only=False)
 assert y["teacher_version"]=="r5_response_mean_v1" and not y["future_in_inputs"]
 y["target"]=q["labels"]["target"];y["grip"]=q["labels"]["grip"]
 history=[];hv=[]
 for ix in y["history_indices"]:
  ix=int(ix);p=OLD/"cache"/eid/(str(ix)+".pt")
  if p.exists():history.append(torch.load(p,map_location="cpu",weights_only=False)["inputs"])
  else:history.append(None)
  hv.append(ix>=0)
 assert len(history)==16 and int(y["history_indices"][-1])==t
 history[-1]=q["inputs"]
 return {"d":q["inputs"],"y":y,"history":history,"history_valid":torch.tensor(hv,dtype=torch.bool)}
