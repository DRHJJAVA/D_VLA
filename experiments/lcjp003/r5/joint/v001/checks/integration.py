import sys,time,traceback,json,os
from pathlib import Path
RUN=Path("/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r5/joint/v001")
sys.path.insert(0,str(RUN/"src"))
import torch
from io_runtime import setup,seed,build,load_record,move,atomic,optimizer,rng_state,restore_rng,digest
from losses import objective
def main():
 setup();seed(20261029);start=time.perf_counter()
 atomic("checks/integration_state.json",{"phase":"building_official_expert","pid":os.getpid(),"time":time.time()})
 model=build();opt=optimizer(model)
 es=json.loads((RUN/"episodes.json").read_text());rec={"episode":es[0]["id"],"frame":13,"split":"train"}
 batch=move(load_record(rec));noise=torch.randn(8,16,6,device="cuda")
 atomic("checks/integration_state.json",{"phase":"forward_backward","pid":os.getpid(),"time":time.time()})
 torch.cuda.synchronize();t=time.perf_counter()
 loss,metrics,e,out=objective(model,batch,noise)
 shapes={k:list(v.shape) for k,v in e.items() if torch.is_tensor(v)}
 assert torch.isfinite(loss),"nonfinite_loss"
 ag=torch.autograd.grad(out[0].square().mean()+out[1].square().mean(),e["score"],retain_graph=True)[0]
 atomic("checks/initial_geometry_diagnostic.json",{"target_index":int(e["target_index"]),"part_valid":int(e["part_valid"].sum()),"point_valid":int(e["points"]["valid"].sum()),"mask_min":float(e["target_logits"].min().detach()),"mask_max":float(e["target_logits"].max().detach()),"score_gradient":float(ag.abs().sum()),"components":metrics})
 assert torch.isfinite(ag).all(),"finite_initial_score_gradient"
 actual_initial_gradient=float(ag.abs().sum())
 loss.backward()
 # This explicit teacher geometry is ONLY an engineering connectivity probe.
 # Formal objective always uses the student-predicted mask and role.
 f=model.front_frame(batch["d"])
 f["role"]=f["role"].detach().clone();f["role"][0,0].fill_(-8);f["role"][0,0,0]=8
 f["masks"]=f["masks"].detach().clone()
 obj=int(batch["y"]["role_distribution"][0,:16].argmax())
 f["masks"][0,:,0]=batch["y"]["instance_masks"][:,obj].float()*16-8
 probe=model.encode(batch["d"],batch["history"],batch["history_valid"],front_override=f)
 pm,pg=model.generate(probe,noise)
 ag=torch.autograd.grad(pm.square().mean()+pg.square().mean(),probe["score"],retain_graph=True)[0]
 assert int(probe["part_valid"].sum())>0 and ag.abs().sum()>0,"nonempty_score_action_path"
 (pm.square().mean()+pg.square().mean()).backward()
 torch.cuda.synchronize()
 grad={}
 for key in ["front","parts.score","parts.gate","action"]:
  ps=[p for n,p in model.named_parameters() if n.startswith(key) and p.requires_grad and p.grad is not None]
  assert ps and all(torch.isfinite(p.grad).all() for p in ps),key+"_finite_gradient"
  grad[key]=sum(float(p.grad.abs().sum()) for p in ps)
  assert grad[key]>0,key+"_nonzero_gradient"
 forward_backward=time.perf_counter()-t
 assert max(batch["y"]["history_indices"])==rec["frame"],"future_history"
 assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
 state={"model":model.state_dict(),"optimizer":opt.state_dict(),"rng":rng_state(),"step":1}
 p=RUN/"checks/engineering_restore.pt";torch.save(state,p)
 oldsum={n:float(v.double().sum()) for n,v in model.state_dict().items() if torch.is_floating_point(v)}
 z=torch.load(p,map_location="cuda",weights_only=False)
 model.load_state_dict(z["model"]);opt.load_state_dict(z["optimizer"]);restore_rng(z["rng"])
 assert all(torch.equal(model.state_dict()[n],v) for n,v in z["model"].items()),"model_exact_restore"
 assert all(float(v.double().sum())==oldsum[n] for n,v in model.state_dict().items() if n in oldsum)
 assert all(int(s["step"])==1 for s in opt.state.values() if "step" in s),"optimizer_restore"
 result={"passed":True,"engineering_updates":1,"formal_updates":0,"official_expert":True,"actual_initial_score_gradient":actual_initial_gradient,"nonempty_probe_teacher_geometry_engineering_only":True,"history_causal":True,"loss":float(loss.detach()),"components":metrics,"gradients":grad,"score_action_gradient_l1":float(ag.abs().sum()),"shapes":shapes,"forward_backward_seconds":forward_backward,"peak_gpu_gib":torch.cuda.max_memory_allocated()/1024**3,"total_seconds":time.perf_counter()-start,"source_files":{str(p.relative_to(RUN)):digest(p) for p in (RUN/"src").glob("*.py")},"time":time.time()}
 atomic("checks/integration.json",result)
 atomic("checks/integration_state.json",{"phase":"passed","time":time.time()})
if __name__=="__main__":
 try:main()
 except BaseException as ex:
  atomic("checks/integration_failure.json",{"type":type(ex).__name__,"message":str(ex)[:240],"frames":[{"file":f.filename,"line":f.lineno} for f in traceback.extract_tb(ex.__traceback__)],"time":time.time()})
  raise SystemExit(1)
