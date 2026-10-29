from common import *
from network import *
from inputs import Inputs
from objective import *
import time,fcntl,statistics
def lora_state(expert):
 return {n:p.detach().clone() for n,p in expert.named_parameters() if p.requires_grad}
def checkpoint(front,expert,ema,optimizer,step):
 return {"front":front.state_dict(),"lora":lora_state(expert),"ema":ema.state_dict(),"optimizer":optimizer.state_dict(),"rng":rng_state(),"completed_updates":step,"sampler":"python_random_uniform_task_demo_window","config_sha256":sha(RUN/"config.json"),"source_lock_sha256":sha(RUN/"source_lock.json")}
def gradients(front,expert):
 def norm(m):return sum(float(p.grad.square().sum()) for p in m.parameters() if p.grad is not None)**.5
 return {"role":norm(front.role_out),"score":norm(front.score),"lambda":norm(front.gate),"explanation":norm(front.explain),"lora":norm(expert.lm_expert)}
def check_sources():
 d=json.loads((RUN/"source_lock.json").read_text())
 for p,h in d["files"].items():assert sha(ROOT/p)==h,("locked_source_changed",p)
 assert sha(RUN/"config.json")==d["config_sha256"]
def fixed_windows():
 rows=[]
 for e in ROWS:
  fractions=[.1,.3,.55,.8] if e["v2_split"]=="train" and e["episode"]=="demo_0" else ([.25,.65] if e["v2_split"]=="dev" else [])
  for j,f in enumerate(fractions):rows.append((e,int((e["frames"]-17)*f),20261027+len(rows)))
 assert len(rows)==80
 return rows
@torch.no_grad()
def evaluate(front,expert,ema,data,step):
 front.eval();rng=rng_state();start=time.perf_counter();records=[];episode_utility={}
 for e,t,ns in fixed_windows():
  d=to_device(data.cache_frame(e,t))
  noise=torch.randn(1,16,6,generator=torch.Generator().manual_seed(ns)).cuda()
  pred,grip,enc=infer(front,expert,d,noise=noise)
  diff=(pred-d["target"]).abs();raw=diff*torch.tensor(data.sd,device="cuda")
  records.append({"episode":e["id"],"split":e["v2_split"],"frame":t,"normalized_motion_mae":float(diff.mean()),"translation_action_mae":float(raw[:,:3].mean()),"rotation_action_mae":float(raw[:,3:].mean()),"gripper_error":float((grip.argmax(-1)!=d["grip"]).float().mean()),"lambda_mean":float(enc["lambda"].mean()),"score_mean":float(enc["scores"][enc["patch_valid"]].mean()),"score_std":float(enc["scores"][enc["patch_valid"]].std(unbiased=False)),"null_probability":float(enc["pi"][-1])})
  if e["v2_split"]=="dev":
   loc=enc["patches"];ct=enc["context"];a,fut=explain_error(ema,loc,ct,d);na,nf=explain_error(ema,ema.null[None,None].expand_as(loc),ct,d)
   v=enc["patch_valid"];fv=d["future_valid"]&v[...,None]
   delta=(na-a)
   fd=(((nf-d["future"]).square().mean(-1)-(fut-d["future"]).square().mean(-1))*fv).sum(-1)/fv.sum(-1).clamp_min(1)
   advantage=float(((delta+fd)*v).sum()/v.sum().clamp_min(1))
   episode_utility.setdefault(e["id"],[]).append(advantage)
 summary={}
 for split in ["train","dev"]:
  r=[x for x in records if x["split"]==split]
  summary[split]={k:float(np.mean([x[k] for x in r])) for k in r[0] if isinstance(r[0][k],float)}
 values=np.array([np.mean(v) for v in episode_utility.values()])
 rng_boot=np.random.default_rng(20261027)
 boot=np.mean(rng_boot.choice(values,(2000,len(values)),replace=True),axis=1)
 low,high=np.quantile(boot,[.025,.975]).tolist()
 gate={"episodes":len(values),"mean_local_over_null":float(values.mean()),"ci95":[low,high],"passed":bool(len(values)==20 and low>0),"unit":"heldout development episode; paired common observed dimensions"}
 report={"step":step,"time":now(),"seconds":time.perf_counter()-start,"windows":80,"fixed_noise":True,"summary":summary,"teacher_gate":gate,"records":records,"semantic_accuracy":"requires independent target annotation; not inferred from model confidence","score_causality":"not yet V4 action ablation proof"}
 save("eval"+str(step).zfill(4)+".json",report)
 front.train();restore_rng(rng)
 return report
def run_evaluation(front,expert,ema,data,optimizer,step):
 start=time.perf_counter()
 if not (RUN/"eval0000.json").exists():
  current={"front":copy.deepcopy(front.state_dict()),"lora":lora_state(expert),"ema":copy.deepcopy(ema.state_dict())}
  # Preserve current optimizer and RNG while evaluating the saved initialization.
  initial=torch.load(RUN/"checkpoints/step0000.pt",map_location="cuda",weights_only=False)
  front.load_state_dict(initial["front"]);expert.load_state_dict(initial["lora"],strict=False);ema.load_state_dict(initial["ema"])
  evaluate(front,expert,ema,data,0)
  front.load_state_dict(current["front"]);expert.load_state_dict(current["lora"],strict=False);ema.load_state_dict(current["ema"])
 report=evaluate(front,expert,ema,data,step)
 front.utility_enabled.fill_(report["teacher_gate"]["passed"])
 base=json.loads((RUN/"eval0000.json").read_text())
 report["relative_motion_mae_reduction"]={s:1-report["summary"][s]["normalized_motion_mae"]/base["summary"][s]["normalized_motion_mae"] for s in ["train","dev"]}
 report["evaluation_and_baseline_seconds"]=time.perf_counter()-start
 save("eval"+str(step).zfill(4)+".json",report)
 append_doc("V2完成"+str(step)+"次真实更新；固定80窗口/固定噪声动作评估："+json.dumps(report["summary"],ensure_ascii=False)+"。局部解释头跨20开发示范配对资格："+json.dumps(report["teacher_gate"],ensure_ascii=False)+"。该检查不代替目标识别准确率或评分作用对照。")
 return report
def main():
 lock=open(RUN/"train.lock","w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 check_sources()
 validation=json.loads((RUN/"engineering/validation.json").read_text());assert validation["passed"]
 seed(CONFIG["seed"]);expert=build_expert();install_lora(expert);front=R2Front().cuda()
 ema=copy.deepcopy(front.explain).eval()
 for p in ema.parameters():p.requires_grad_(False)
 params=[p for p in list(front.parameters())+list(expert.parameters()) if p.requires_grad]
 opt=torch.optim.AdamW(params,**{k:tuple(v) if k=="betas" else v for k,v in CONFIG["optimizer"].items() if k!="grad_clip"})
 data=Inputs(expert);completed=0
 last=RUN/"checkpoints/last.pt"
 if last.exists():
  c=torch.load(last,map_location="cuda",weights_only=False)
  assert c["config_sha256"]==sha(RUN/"config.json") and c["source_lock_sha256"]==sha(RUN/"source_lock.json")
  front.load_state_dict(c["front"]);expert.load_state_dict(c["lora"],strict=False);ema.load_state_dict(c["ema"]);opt.load_state_dict(c["optimizer"])
  c["rng"]["torch"]=c["rng"]["torch"].cpu();c["rng"]["cuda"]=[x.cpu() for x in c["rng"]["cuda"]];restore_rng(c["rng"])
  completed=c["completed_updates"]
 else:
  assert not (RUN/"checkpoints/step0000.pt").exists(),"Existing initialization requires explicit recovery"
  atomic_torch("checkpoints/step0000.pt",checkpoint(front,expert,ema,opt,0))
  atomic_torch("checkpoints/last.pt",checkpoint(front,expert,ema,opt,0))
 tasks=sorted(set(e["task"] for e in ROWS));train={t:[e for e in ROWS if e["task"]==t and e["v2_split"]=="train"] for t in tasks}
 recent=[];start=time.perf_counter()
 append_doc("R2 V2联合训练启动，工程验证通过；从"+str(completed)+"更新接续，正式初始化与工程优化器练习隔离。登记3000更新，80训练示范；不恢复旧P1。")
 for step in range(completed+1,CONFIG["updates"]+1):
  check_sources();front.train();front.updates.fill_(step-1);opt.zero_grad(set_to_none=True)
  times={"data_h2d":0.,"forward":0.,"backward":0.,"optimizer":0.};metrics=[];windows=[];st=time.perf_counter();cachehits=0
  save("state.json",{"status":"training","completed_updates":step-1,"current_update":step,"microbatch":0,"time":now(),"pid":os.getpid()})
  for micro in range(CONFIG["effective_batch"]):
   e=random.choice(train[random.choice(tasks)]);t=random.randint(0,e["frames"]-17)
   cachehits+=int((RUN/"cache"/e["id"]/(str(t)+".pt")).exists())
   a=time.perf_counter();d=to_device(data.cache_frame(e,t));torch.cuda.synchronize();times["data_h2d"]+=time.perf_counter()-a
   plan=None
   if t>0 and random.random()<.5*min((step-1)/500,1)*.8:
    a=time.perf_counter();past=to_device(data.cache_frame(e,t-1))
    with torch.no_grad():
     old,gr,_=infer(front,expert,past)
     gr=(gr.argmax(-1).float()*2-1)[:,None]
     full=torch.cat([old,gr],-1);plan=torch.cat([full[1:],full[-1:]],0).detach()
    torch.cuda.synchronize();times["data_h2d"]+=time.perf_counter()-a
   a=time.perf_counter();loss,met,_=objective(front,expert,ema,d,step,plan);torch.cuda.synchronize();times["forward"]+=time.perf_counter()-a
   assert torch.isfinite(loss),"nonfinite_loss"
   a=time.perf_counter();(loss/CONFIG["effective_batch"]).backward();torch.cuda.synchronize();times["backward"]+=time.perf_counter()-a
   metrics.append(met);windows.append({"episode":e["id"],"frame":t})
   save("state.json",{"status":"training","completed_updates":step-1,"current_update":step,"microbatch":micro+1,"time":now(),"pid":os.getpid()})
  a=time.perf_counter()
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params),"nonfinite_gradient"
  grads=gradients(front,expert)
  assert all(np.isfinite(v) and v>0 for v in grads.values()),("missing_gradient",grads)
  torch.nn.utils.clip_grad_norm_(params,CONFIG["optimizer"]["grad_clip"])
  opt.step()
  with torch.no_grad():
   for dst,src in zip(ema.parameters(),front.explain.parameters()):dst.lerp_(src,1-CONFIG["EMA"])
  front.updates.fill_(step);torch.cuda.synchronize();times["optimizer"]=time.perf_counter()-a
  seconds=time.perf_counter()-st
  record={"update":step,"time":now(),"seconds":seconds,"timings":times,"cache_hits":cachehits,"gradient_norms":grads,"metrics":{k:float(np.mean([v[k] for v in metrics])) for k in metrics[0]},"windows":windows,"lambda_cap":front.cap(),"utility_enabled":bool(front.utility_enabled)}
  with (RUN/"runtime_timings.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
  recent.append(record);recent=recent[-50:]
  if step%CONFIG["save_every"]==0 or step in [20,50,250]:
   c=checkpoint(front,expert,ema,opt,step);atomic_torch("checkpoints/last.pt",c)
   if step in [20,50] or step%250==0:atomic_torch("checkpoints/step"+str(step).zfill(4)+".pt",c)
  save("state.json",{"status":"training","completed_updates":step,"time":now(),"pid":os.getpid(),"last_update_seconds":seconds,"metrics":record["metrics"]})
  if step in [20,50]:
   stable=recent[-min(20,len(recent)):];secs=[v["seconds"] for v in stable];mean=float(np.mean(secs))
   report={"time":now(),"completed_updates":step,"mean_update_seconds":mean,"median_update_seconds":statistics.median(secs),"observed_cache_hit_fraction":sum(v["cache_hits"] for v in stable)/(8*len(stable)),"remaining_training_hours_at_observed_rate":(CONFIG["updates"]-step)*mean/3600,"range_hours_at_observed_p10_p90":[(CONFIG["updates"]-step)*float(np.quantile(secs,q))/3600 for q in [.1,.9]],"baseline_and_periodic_evaluation_not_included":True,"cold_cache_preparation_included":True,"shared_GPU_load_may_change":True,"mean_stage_seconds":{k:float(np.mean([v["timings"][k] for v in stable])) for k in times}}
   save("eta"+str(step).zfill(4)+".json",report)
   append_doc("前"+str(step)+"次正式更新已保存，实测耗时："+json.dumps(report,ensure_ascii=False)+"。")
  if step%CONFIG["eval_every"]==0:
   save("state.json",{"status":"evaluating","completed_updates":step,"time":now(),"pid":os.getpid()})
   report=run_evaluation(front,expert,ema,data,opt,step)
   atomic_torch("checkpoints/last.pt",checkpoint(front,expert,ema,opt,step))
 save("state.json",{"status":"V2_training_complete_pending_stage_review","completed_updates":CONFIG["updates"],"time":now(),"automatic_closed_loop":False})
 save("complete.json",{"time":now(),"completed_updates":CONFIG["updates"],"checkpoint_sha256":sha(RUN/"checkpoints/last.pt"),"V2_passed":False,"reason":"Training completed; semantic selection and utility acceptance still require registered evaluation"})
if __name__=="__main__":
 try:main()
 except Exception as e:fail("formal_training",e);raise SystemExit(1)
