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
r5_model.cloud_summary=checked_summary
def main():
 setup();seed(20261029);m=build();o=optimizer(m)
 ck=torch.load(RUN/"checkpoints/last.pt",map_location="cuda",weights_only=False)
 assert ck["step"]==70
 m.load_state_dict(ck["model"]);o.load_state_dict(ck["optimizer"]);restore_rng(ck["rng"]);m.train()
 schedule=json.loads((RUN/"sampling_schedule.json").read_text());rows=[]
 for step in range(71,75):
  o.zero_grad(set_to_none=True)
  for i,rec in enumerate(schedule[step-1]):
   b=move(load_record(rec));prev=previous_encoding(m,rec);noise=torch.randn(8,16,6,device="cuda")
   loss,metrics,e,out=objective(m,b,noise,prev)
   row={"step":step,"microbatch":i,"record":rec,"loss":float(loss.detach()),"components":metrics}
   if not torch.isfinite(loss):
    row["nonfinite_outputs"]={k:int((~torch.isfinite(v)).sum()) for k,v in e.items() if torch.is_tensor(v) and torch.is_floating_point(v) and not torch.isfinite(v).all()}
    row["point_valid"]=int(e["points"]["valid"].sum());row["part_valid"]=int(e["part_valid"].sum())
    row["previous_point_valid"]=int(prev["points"]["valid"].sum()) if prev else None
    row["motion_finite"]=bool(torch.isfinite(out[0]).all());row["grip_finite"]=bool(torch.isfinite(out[1]).all())
    rows.append(row);atomic("engineering/nonfinite_replay.json",{"reproduced":True,"rows":rows,"time":time.time()});return
   (loss/8).backward();rows.append(row)
  for group in ["front","parts.score","parts.gate","action"]:
   ps=[p for n,p in m.named_parameters() if n.startswith(group) and p.grad is not None]
   assert all(torch.isfinite(p.grad).all() for p in ps)
   unused=sum(float(p.grad.abs().sum()) for p in ps)
  torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
  if step==73:torch.save({"model":m.state_dict(),"rng":rng_state()},RUN/"engineering/replayed73.pt")
  atomic("engineering/nonfinite_replay_state.json",{"replayed_step":step,"time":time.time()})
 atomic("engineering/nonfinite_replay.json",{"reproduced":False,"rows":rows,"time":time.time()})
if __name__=="__main__":
 try:main()
 except BaseException as ex:
  atomic("engineering/nonfinite_replay_failure.json",{"type":type(ex).__name__,"message":str(ex)[:200],"frames":[{"file":f.filename,"line":f.lineno} for f in traceback.extract_tb(ex.__traceback__)],"time":time.time()})
