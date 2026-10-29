from common import *
from inputs import Inputs
from network import *
from objective import *
import time,fcntl
def gradnorm(module):
 return sum(float(p.grad.float().square().sum()) for p in module.parameters() if p.grad is not None)**.5
def main():
 lock=open(RUN/"engineering/validation.lock","w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 seed(CONFIG["seed"]);save("state.json",{"status":"validating_R2_implementation","optimizer_updates":0,"time":now()})
 expert=build_expert();paths=install_lora(expert);front=R2Front().cuda()
 data=Inputs(expert)
 row=next(e for e in ROWS if "bbq_sauce" in e["task"] and e["episode"]=="demo_0")
 started=time.perf_counter();sample=data.cache_frame(row,20);d=to_device(sample)
 save("engineering/prepared_first_input.json",{"time":now(),"episode":row["id"],"frame":20,"shapes":{k:list(v.shape) for k,v in sample.items() if torch.is_tensor(v)},"seconds":time.perf_counter()-started})
 ema=copy.deepcopy(front.explain).eval()
 for p in ema.parameters():p.requires_grad_(False)
 # Verify the inherited official expert interpolation orientation numerically.
 seen=[]
 hook=front.motion_in.register_forward_pre_hook(lambda m,a:seen.append(a[0].detach().clone()))
 enc=front.encode(d["x"],d["valid"],d["h"],d["words"],d["priors"])
 count=enc["prefix"].shape[0]
 with torch.no_grad():expert_forward(front,expert,enc,torch.zeros(count,16,6,device="cuda"),.2,torch.ones(16,6,device="cuda"))
 hook.remove();interpolation=float(seen[0].mean());assert abs(interpolation-.8)<1e-6,("flow_orientation",interpolation)
 with torch.no_grad():
  e0=front.encode(d["x"],d["valid"],d["h"],d["words"],d["priors"],lambda_override=0)
  e1=front.encode(d["x"],d["valid"],d["h"],d["words"],d["priors"],lambda_override=1)
  assert torch.equal(e0["mixed"],e0["base"]);assert torch.equal(e1["mixed"],e1["part"])
  pad=d["x"].clone();pad[~d["valid"]]=4321
  epad=front.encode(pad,d["valid"],d["h"],d["words"],d["priors"])
  assert torch.equal(enc["prefix"],epad["prefix"]),"padding_dependency"
  ds=front.state_dict();front.updates.fill_(37);state=copy.deepcopy(front.state_dict());front.updates.zero_();front.load_state_dict(state);assert int(front.updates)==37
  front.updates.zero_()
  missing=front.encode(d["x"],torch.zeros_like(d["valid"]),d["h"],d["words"],d["priors"])
  assert torch.isfinite(missing["prefix"]).all() and (missing["lambda"]==0).all()
  assert (missing["scores"]==0).all()
  assert max(sample["history_indices"])==sample["frame"]
 # Formal progress is not advanced by the engineering optimizer exercise.
 params=[p for p in list(front.parameters())+list(expert.parameters()) if p.requires_grad]
 optimizer=torch.optim.AdamW(params,**{k:(tuple(v) if k=="betas" else v) for k,v in CONFIG["optimizer"].items() if k!="grad_clip"})
 loss,metrics,enc=objective(front,expert,ema,d,0);loss.backward()
 grads={"role":gradnorm(front.role_out),"score":gradnorm(front.score),"lambda":gradnorm(front.gate),"point":gradnorm(front.point),"explanation":gradnorm(front.explain),"lora":sum(float(p.grad.square().sum()) for p in expert.parameters() if p.requires_grad and p.grad is not None)**.5}
 assert all(np.isfinite(x) and x>0 for x in grads.values()),grads
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
 optimizer.step();optimizer.zero_grad(set_to_none=True)
 lo={n:v.detach().cpu().clone() for n,v in expert.state_dict().items() if ".a.weight" in n or ".b.weight" in n}
 checkpoint={"front":copy.deepcopy(front.state_dict()),"lora":lo,"optimizer":copy.deepcopy(optimizer.state_dict()),"rng":rng_state(),"step":1}
 atomic_torch("engineering/resume_probe.pt",checkpoint)
 def do_step():
  loss,_,_=objective(front,expert,ema,d,1);loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
 do_step();expected=[p.detach().cpu().clone() for p in params]
 c=torch.load(RUN/"engineering/resume_probe.pt",map_location="cuda",weights_only=False)
 front.load_state_dict(c["front"]);expert.load_state_dict(c["lora"],strict=False);optimizer.load_state_dict(c["optimizer"])
 # RNG tensors must be restored on CPU.
 c["rng"]["torch"]=c["rng"]["torch"].cpu();c["rng"]["cuda"]=[v.cpu() for v in c["rng"]["cuda"]];restore_rng(c["rng"])
 do_step();diff=max(float((p.detach().cpu()-a).abs().max()) for p,a in zip(params,expected))
 assert diff==0,("resume_mismatch",diff)
 report={"passed":True,"time":now(),"formal_optimizer_updates":0,"engineering_optimizer_exercises":3,"lora_modules":len(paths),"flow_interpolation_at_t_point2":interpolation,"gradient_norms":grads,"padding_max":0.,"lambda_endpoints_exact":True,"resume_parameter_max":diff,"history_indices":sample["history_indices"],"future_targets_not_in_encode":True,"all_parts_missing_lambda_zero":True,"frozen_expert_gradient_absent":all(p.grad is None for p in expert.parameters() if not p.requires_grad),"first_input_seconds":sample["cache_seconds"],"trainable_parameters":sum(p.numel() for p in params),"initial_metrics":metrics,"peak_gpu_gib":torch.cuda.max_memory_allocated()/2**30}
 save("engineering/validation.json",report)
 save("state.json",{"status":"implementation_validated_pending_training_launch","optimizer_updates":0,"time":now()})
if __name__=="__main__":
 try:main()
 except Exception as e:fail("implementation_validation",e);raise SystemExit(1)
