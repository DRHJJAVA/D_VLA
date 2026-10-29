import os,sys,time,json,pathlib,traceback,fcntl,concurrent.futures
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import torch,numpy as np
from io_runtime import *
from losses import objective
from evaluation import evaluate
def state(status,step,**kw):
 atomic("state.json",{"status":status,"completed_updates":step,"total_updates":3000,"pid":os.getpid(),"time":time.time(),**kw})
def ready_record(rec):
 p=RUN/"teacher"/rec["episode"]/(str(rec["frame"])+".pt")
 while not p.exists():
  if (RUN/"STOP").exists():raise InterruptedError("user_stop")
  if (RUN/"prepare_failure.json").exists():raise RuntimeError("teacher_preparation_failed")
  time.sleep(1)
 return load_record(rec)
def previous_encoding(model,rec):
 t=int(rec["frame"])-1
 base=OLD/"cache"/rec["episode"]
 if t<0 or not (base/(str(t)+".pt")).exists():return None
 d=torch.load(base/(str(t)+".pt"),map_location="cpu",weights_only=False)["inputs"]
 history=[]
 for ix in range(t-15,t+1):
  p=base/(str(max(0,ix))+".pt")
  history.append(torch.load(p,map_location="cpu",weights_only=False)["inputs"] if p.exists() else None)
 history[-1]=d
 b=move({"d":d,"history":history})
 with torch.no_grad():return model.encode(b["d"],b["history"],torch.ones(16,device="cuda",dtype=torch.bool))
def eta(step,times,eval_seconds):
 recent=times[-min(len(times),20):];median=float(np.median(recent))
 remaining=(3000-step)*median+((3000-step)//250+1)*eval_seconds
 result={"step":step,"median_update_seconds":median,"recent_mean_update_seconds":float(np.mean(recent)),"estimated_remaining_seconds":remaining,"estimated_end_unix":time.time()+remaining,"evaluation_seconds":eval_seconds,"measured_updates_retained":True,"closed_loop_time_included":False,"shared_gpu_load_can_change":True,"time":time.time()}
 atomic("eta%04d.json"%step,result)
def main():
 lock=open(RUN/"train.lock","w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 src=json.loads((RUN/"source_lock.json").read_text())
 for path,value in src["files"].items():assert digest(ROOT/path)==value,"locked_source_changed"
 check=json.loads((RUN/"checks/integration.json").read_text());assert check["passed"]
 setup();seed(20261029);model=build();opt=optimizer(model);start=0
 last=RUN/"checkpoints/last.pt"
 if last.exists():
  ck=torch.load(last,map_location="cuda",weights_only=False)
  assert ck["config_hash"]==digest(RUN/"config.json")
  if ck["source_lock"]!=digest(RUN/"source_lock.json"):
   inheritance=json.loads((RUN/"engineering/history_fix_inheritance.json").read_text())
   assert ck["source_lock"]==inheritance["previous_source_lock"]
   assert digest(last)==inheritance["checkpoint_sha256"] and int(ck["step"])==inheritance["completed_updates"]
   assert inheritance["only_change"]=="historical_observation_loading_and_registered_resume"

  model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"]);restore_rng(ck["rng"]);start=int(ck["step"])
 else:checkpoint(model,opt,0)
 if (RUN/"eval0000.json").exists():ev=json.loads((RUN/"eval0000.json").read_text())
 else:
  state("initialization_evaluation",0);ev=evaluate(model,0)
 schedule=json.loads((RUN/"sampling_schedule.json").read_text());times=[];model.train()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for step in range(start+1,3001):
   if (RUN/"STOP").exists():
    checkpoint(model,opt,step-1);state("stopped",step-1);return
   state("training",step-1,next_update=step)
   records=schedule[step-1];futures=[pool.submit(ready_record,r) for r in records]
   opt.zero_grad(set_to_none=True);begin=time.perf_counter();stats=[];io=back=prior=0.
   for rec,future in zip(records,futures):
    t=time.perf_counter();batch=move(future.result());torch.cuda.synchronize();io+=time.perf_counter()-t
    t=time.perf_counter();prev=previous_encoding(model,rec);torch.cuda.synchronize();prior+=time.perf_counter()-t
    noise=torch.randn(8,16,6,device="cuda")
    loss,metric,e,out=objective(model,batch,noise,prev)
    if not torch.isfinite(loss):raise FloatingPointError("nonfinite_objective")
    t=time.perf_counter();(loss/8).backward();torch.cuda.synchronize();back+=time.perf_counter()-t
    stats.append(metric);del batch,prev,e,out,loss
   grads={};optbegin=time.perf_counter()
   for group in ["front","parts.score","parts.gate","action"]:
    ps=[p for n,p in model.named_parameters() if n.startswith(group) and p.grad is not None]
    if not all(torch.isfinite(p.grad).all() for p in ps):raise FloatingPointError("nonfinite_"+group+"_gradient")
    grads[group]=sum(float(p.grad.abs().sum()) for p in ps)
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.cuda.synchronize()
   elapsed=time.perf_counter()-begin;times.append(elapsed)
   row={"step":step,"seconds":elapsed,"data_h2d":io,"previous_causal_encode":prior,"encode":sum(x["encode_seconds"] for x in stats),"action_forward":sum(x["action_seconds"] for x in stats),"supervision":sum(x["supervision_seconds"] for x in stats),"backward":back,"optimizer_check":time.perf_counter()-optbegin,"gradients":grads,"losses":{k:float(np.mean([x[k] for x in stats])) for k in stats[0] if not k.endswith("_seconds")},"peak_gpu_gib":torch.cuda.max_memory_allocated()/1024**3,"time":time.time()}
   with (RUN/"runtime_timings.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
   if step%10==0 or step in (20,50):checkpoint(model,opt,step)
   state("training",step,last_seconds=elapsed,score_gradient_l1=grads["parts.score"])
   if step in (20,50):eta(step,times,ev["seconds"])
   if step%250==0:
    state("evaluating",step);ev=evaluate(model,step)
    from datetime import datetime
    from zoneinfo import ZoneInfo
    stamp=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    with (ROOT/"docs/LCJP003_状态条件三维部位评分与DriftingVLA设计方案.md").open("a") as f:
     f.write("\n\n### "+stamp+" R5 第"+str(step)+"次更新\n\n固定80窗口评估："+json.dumps(ev["summary"],ensure_ascii=False)+"。以上为离线误差，闭环成功率待独立测试。\n")
 checkpoint(model,opt,3000);state("training_complete_pending_closed_loop",3000)
if __name__=="__main__":
 try:main()
 except BaseException as ex:
  atomic("failure.json",{"type":type(ex).__name__,"message":str(ex)[:200],"frames":[{"file":f.filename,"line":f.lineno} for f in traceback.extract_tb(ex.__traceback__)],"time":time.time()})
  raise SystemExit(1)
