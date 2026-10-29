import sys,pathlib,json,time,traceback
RUN=pathlib.Path("/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r5/joint/v001")
sys.path.insert(0,str(RUN/"src"))
import torch
from io_runtime import *
from train import previous_encoding
from losses import objective,response_targets
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
  torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
  atomic("engineering/nonfinite_replay_state.json",{"replayed_step":step,"time":time.time()})
 atomic("engineering/nonfinite_replay.json",{"reproduced":False,"rows":rows,"time":time.time()})
if __name__=="__main__":
 try:main()
 except BaseException as ex:
  atomic("engineering/nonfinite_replay_failure.json",{"type":type(ex).__name__,"message":str(ex)[:200],"frames":[{"file":f.filename,"line":f.lineno} for f in traceback.extract_tb(ex.__traceback__)],"time":time.time()})
