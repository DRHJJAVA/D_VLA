import os
os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import sys,pathlib,json,time,traceback,fcntl
RUN=pathlib.Path("/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r5/joint/v001")
OUT=RUN/"closed_loop_object50"
sys.path.insert(0,str(RUN/"src"))
def write(name,data):
 p=OUT/name;t=p.with_suffix(".tmp");t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(p)
def main():
 lock=open(OUT/"executor.lock","w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write("state.json",{"status":"waiting_for_training","pid":os.getpid(),"video":False,"time":time.time()})
 while True:
  if (OUT/"STOP").exists() or (RUN/"STOP").exists():return
  state=json.loads((RUN/"state.json").read_text()) if (RUN/"state.json").exists() else {}
  if state.get("status")=="training_complete_pending_closed_loop" and state.get("completed_updates")==3000:break
  time.sleep(10)
 import torch,numpy as np
 from io_runtime import ROOT,setup,seed,build,move,digest
 from adapter import Front,render_module
 from r5_model import cloud_summary
 setup();seed(20261029)
 locked=json.loads((RUN/"source_lock.json").read_text())
 for p,h in locked["files"].items():assert digest(ROOT/p)==h,"source_hash_mismatch"
 buildlock=json.loads((OUT/"build_lock.json").read_text())
 for p,h in buildlock["files"].items():assert digest(ROOT/p)==h,"test_build_hash_mismatch"
 ckpath=RUN/"checkpoints/step3000.pt";ck=torch.load(ckpath,map_location="cuda",weights_only=False)
 assert ck["step"]==3000 and ck["source_lock"]==digest(RUN/"source_lock.json")
 steps=[int(s["step"]) for s in ck["optimizer"]["state"].values() if "step" in s]
 assert max(steps)==3000 and all(0<x<=3000 for x in steps),"optimizer_steps"
 assert json.loads((OUT/"ready.json").read_text())["passed"],"online_adapter_gate"
 model=build();model.load_state_dict(ck["model"]);model.eval();del ck
 front=Front();renderer=render_module()
 norm=json.loads((RUN/"normalization.json").read_text());mean=np.array(norm["mean"]);std=np.array(norm["std"])
 es=json.loads((RUN/"episodes.json").read_text());tasks=sorted(set(e["task"] for e in es))
 protocol={"suite":"libero_object","tasks":10,"trials":50,"initial_state_indices":[0,1,2,3,4],"max_control_steps":600,"settle_steps":5,"checkpoint_sha256":digest(ckpath),"sam_online":False,"generate_video":False,"replan":"execute first predicted action, observe again; no previous-plan input","noise_seed_base":20261029,"read_heldout_demonstrations":False,"time":time.time()}
 write("protocol.json",protocol)
 rows=json.loads((OUT/"results.json").read_text()).get("trials",[]) if (OUT/"results.json").exists() else []
 for trial in range(len(rows),50):
  if (OUT/"STOP").exists() or (RUN/"STOP").exists():break
  task=tasks[trial//5];initial=trial%5
  e=next(x for x in es if x["task"]==task and x["episode"]=="demo_0")
  work=OUT/("trial_%03d"%trial);work.mkdir(exist_ok=True)
  render=renderer.LiveRender(e["id"],work,capacity=601)
  initname=task.removesuffix("_demo")+".init"
  initpath=pathlib.Path("/home/drh/project/ECV-VLA/vendor/LIBERO/libero/libero/init_files/libero_object")/initname
  initial_states=torch.load(initpath,map_location="cpu",weights_only=False)
  render.env.set_init_state(initial_states[initial])
  for _ in range(5):render.env.step(np.zeros(7,dtype=np.float32))
  render.capture(0)
  words=torch.load(OUT/"task_words.pt",weights_only=False,map_location="cpu")[task]
  gen=torch.Generator(device="cpu").manual_seed(20261029+trial)
  acts=[];summaries=[];success=False;done=False;times=[];begin=time.perf_counter()
  for step in range(600):
   if (OUT/"STOP").exists() or (RUN/"STOP").exists():break
   tick=time.perf_counter();d=move(front.inputs(render.out,step,words,acts))
   with torch.no_grad():
    f=model.front_frame(d);mask,_=model.chosen(f);now=cloud_summary(d,mask.sigmoid())
    if summaries:past=([summaries[0]]*max(0,15-len(summaries))+summaries[-15:])
    else:past=[now]*15
    enc=model.encode(d,front_override=f,causal_summaries=torch.stack(past))
    noise=torch.randn(1,16,6,generator=gen).cuda();motion,grip=model.generate(enc,noise)
    raw=motion[0,0].cpu().numpy()*std+mean
    action=np.r_[raw,float(grip[0,0].argmax())*2-1].astype(np.float32)
   assert np.isfinite(action).all(),"nonfinite_action"
   executed=np.clip(action,-1,1);acts.append(executed);summaries.append(now.detach());summaries=summaries[-16:]
   _,done=render.step(executed,step+1);success=bool(render.env.check_success())
   dt=time.perf_counter()-tick;times.append(dt)
   with (work/"steps.jsonl").open("a") as log:log.write(json.dumps({"step":step,"action":action.tolist(),"executed":executed.tolist(),"seconds":dt,"target_index":int(enc["target_index"]),"score_mean":float(enc["score"].mean()),"lambda_mean":float(enc["lambda"].mean()),"success":success})+"\n")
   write("state.json",{"status":"testing","completed_trials":len(rows),"successes":sum(x["success"] for x in rows),"trial":trial,"step":step+1,"pid":os.getpid(),"time":time.time()})
   if success or done:break
  render.close()
  interrupted=(OUT/"STOP").exists() or (RUN/"STOP").exists()
  if interrupted:
   write("state.json",{"status":"paused","completed_trials":len(rows),"interrupted_trial":trial,"time":time.time()});break
  row={"trial":trial,"task":task,"initial_state":initial,"success":success,"environment_done":bool(done),"steps":step+1,"seconds":time.perf_counter()-begin,"median_step_seconds":float(np.median(times))}
  rows.append(row);(work/"result.json").write_text(json.dumps(row,indent=2))
  result={"completed":len(rows),"planned":50,"successes":sum(x["success"] for x in rows),"success_rate":sum(x["success"] for x in rows)/len(rows),"trials":rows,"video_generated":False,"time":time.time()}
  write("results.json",result)
 if len(rows)==50:
  write("complete.json",result)
  from datetime import datetime
  from zoneinfo import ZoneInfo
  stamp=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
  with (ROOT/"docs/LCJP003_状态条件三维部位评分与DriftingVLA设计方案.md").open("a") as f:f.write("\n\n### "+stamp+" R5 Object50 无视频闭环结果\n\n成功 "+str(result["successes"])+"/50，成功率 "+str(round(result["success_rate"]*100,2))+"%。所有失败计入分母，未生成视频。\n")
if __name__=="__main__":
 try:main()
 except BaseException as ex:
  write("failure.json",{"type":type(ex).__name__,"message":str(ex)[:200],"frames":[{"file":f.filename,"line":f.lineno} for f in traceback.extract_tb(ex.__traceback__)],"time":time.time()})
  raise SystemExit(1)
