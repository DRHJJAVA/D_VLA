import os,sys,json,ast,hashlib,random,datetime,traceback,copy
from pathlib import Path
os.environ.setdefault("USE_TF","0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
ROOT=Path("/mnt/nvme1/drh/D_VLA")
RUN=ROOT/"experiments/lcjp003/r2/v2/v001"
V1=ROOT/"experiments/lcjp003/r2/v0_v1/v001"
CONFIG=json.loads((RUN/"config.json").read_text())
ROWS=json.loads((RUN/"episodes.json").read_text())
os.chdir(ROOT)
sys.path[:0]=[str(V1),str(V1/"fusion_interface"),str(ROOT/"experiments/lcjp003/src")]
import numpy as np
import torch
def now():
 return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(name,data):
 p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def fail(stage,e):
 save("failure.json",{"time":now(),"stage":stage,"type":type(e).__name__,"message":str(e)[:400],"frames":[{"file":f.filename,"line":f.lineno} for f in traceback.extract_tb(e.__traceback__)]})
def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.set_num_threads(4)
def load_defs(path):
 t=ast.parse(Path(path).read_text())
 body=[]
 for n in t.body:
  if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef,ast.ClassDef)):body.append(n)
  elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=="insert":body.append(n)
 g={"__file__":str(path),"__name__":"r2_reused_library"}
 exec(compile(ast.Module(body=body,type_ignores=[]),str(path),"exec"),g)
 return g
_LEGACY=None
def legacy():
 global _LEGACY
 if _LEGACY is None:
  _LEGACY=load_defs(V1/"v0_corrected_check.py")
  _LEGACY["save"]=lambda *a,**k:None
 return _LEGACY
def build_expert(device="cuda"):
 from safetensors.torch import load_file
 from smolvlm_with_expert_official import SmolVLMWithExpertModel
 cfg=json.loads((ROOT/"experiments/lcjp003/weights/smolvla_base/config.json").read_text())
 keys=["attention_mode","num_expert_layers","num_vlm_layers","self_attn_every_n_layers","expert_width_multiplier"]
 model=SmolVLMWithExpertModel(model_id=str(ROOT/"experiments/lcjp003/weights/smolvlm2"),load_vlm_weights=False,train_expert_only=True,freeze_vision_encoder=True,device="cpu",**{k:cfg[k] for k in keys})
 raw=load_file(str(ROOT/"experiments/lcjp003/weights/smolvla_base/model.safetensors"))
 prefix="model.vlm_with_expert."
 sub={k[len(prefix):]:v for k,v in raw.items() if k.startswith(prefix)}
 result=model.load_state_dict(sub,strict=True);assert len(sub)==490
 del raw,sub
 model.float().eval()
 for p in model.parameters():p.requires_grad_(False)
 return model.to(device)
def text_words(expert,instruction):
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(ROOT/"experiments/lcjp003/weights/smolvlm2"),local_files_only=True)
 ti=tok(instruction,return_tensors="pt")["input_ids"].to(next(expert.parameters()).device)
 with torch.no_grad():w=expert.embed_language_tokens(ti)
 return w[0].detach().float().cpu()
def setup_normalization():
 import h5py
 rows=[e for e in ROWS if e["v2_split"]=="train"];a=[]
 for e in rows:
  with h5py.File(ROOT/e["raw"],"r") as f:a.append(f["data/"+e["episode"]+"/actions"][:])
 a=np.concatenate(a);assert set(np.unique(a[:,6]).tolist())=={-1.,1.}
 mu=a[:,:6].mean(0);sd=a[:,:6].std(0).clip(1e-6)
 d={"mean":mu.tolist(),"std":sd.tolist(),"episodes":80,"tasks":10,"frames":len(a),"gripper_values":[-1,1],"development_used":False}
 p=RUN/"normalization.json"
 if p.exists():assert json.loads(p.read_text())==d
 else:save("normalization.json",d)
 return mu,sd
def rng_state():
 return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}
def restore_rng(s):
 random.setstate(s["python"]);np.random.set_state(s["numpy"]);torch.set_rng_state(s["torch"]);torch.cuda.set_rng_state_all(s["cuda"])
def atomic_torch(name,data):
 p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(".tmp");torch.save(data,tmp);tmp.replace(p)
def append_doc(text):
 p=ROOT/"docs/LCJP003_状态条件三维部位评分与DriftingVLA设计方案.md"
 s=p.read_text();m="## 历史版本与实验记录（原文保留）";a,b=s.split(m,1)
 assert hashlib.sha256((m+b).encode()).hexdigest()=="bbc591fa881a9046d663676a1a5ce8a5804605e1d20cd2b6aea85d677ef0aa53"
 a=a.replace("### R2.12","\n### V2 执行记录 "+now()+"\n\n"+text+"\n\n### R2.12",1)
 p.write_text(a+m+b)
