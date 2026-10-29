from fast_inputs import *
from prefetch import forecast
from network import R2Front,install_lora
from objective import objective,to_device
import traceback,time,pickle
def main():
 seed(CONFIG["seed"]);expert=build_expert();install_lora(expert);front=R2Front().cuda()
 c=torch.load(RUN/"checkpoints/last.pt",map_location="cuda",weights_only=False)
 ema=copy.deepcopy(front.explain).eval()
 for p in ema.parameters():p.requires_grad_(False)
 params=[p for p in list(front.parameters())+list(expert.parameters()) if p.requires_grad]
 opt=torch.optim.AdamW(params,**{k:tuple(v) if k=="betas" else v for k,v in CONFIG["optimizer"].items() if k!="grad_clip"})
 cases=json.loads((WORK/"input_equivalence.json").read_text())["cases"];e=cases[-2]["episode"];t=cases[-2]["frame"]
 original=to_device(torch.load(WORK/"paired_inputs/original/cache"/e/(str(t)+".pt"),map_location="cpu",weights_only=False))
 fast=to_device(torch.load(WORK/"paired_inputs/optimized/cache"/e/(str(t)+".pt"),map_location="cpu",weights_only=False))
 def restore():
  front.load_state_dict(c["front"]);expert.load_state_dict(c["lora"],strict=False);ema.load_state_dict(c["ema"]);opt.load_state_dict(copy.deepcopy(c["optimizer"]));opt.zero_grad(set_to_none=True)
  s=copy.deepcopy(c["rng"]);s["torch"]=s["torch"].cpu();s["cuda"]=[v.cpu() for v in s["cuda"]];restore_rng(s)
 restore();loss,_,enc=objective(front,expert,ema,original,c["completed_updates"])
 loss.backward();loss1=float(loss);prefix=enc["prefix"].detach().cpu();grads=[p.grad.detach().cpu().clone() if p.grad is not None else None for p in params]
 torch.nn.utils.clip_grad_norm_(params,CONFIG["optimizer"]["grad_clip"]);opt.step();after=[p.detach().cpu().clone() for p in params]
 restore();loss,_,enc=objective(front,expert,ema,fast,c["completed_updates"]);loss.backward()
 gd=max(float((p.grad.detach().cpu()-g).abs().max()) for p,g in zip(params,grads) if g is not None)
 pd=float((prefix-enc["prefix"].detach().cpu()).abs().max());ld=abs(float(loss)-loss1)
 torch.nn.utils.clip_grad_norm_(params,CONFIG["optimizer"]["grad_clip"]);opt.step()
 ud=max(float((p.detach().cpu()-x).abs().max()) for p,x in zip(params,after))
 assert max(gd,pd,ld,ud)==0
 # Verify forecast against the literal original sampler and plan-condition AST.
 tree=ast.parse((RUN/"src/train.py").read_text());mainfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
 loop=next(n for n in ast.walk(mainfn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="micro")
 assigns=[n for n in loop.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ["e","t"] for x in n.targets)]
 cond=next(n.test for n in loop.body if isinstance(n,ast.If) and any(isinstance(v,ast.Name) and v.id=="random" for v in ast.walk(n.test)))
 rr=random.Random();rr.setstate(c["rng"]["python"]);tasks=sorted(set(e["task"] for e in ROWS))
 trains={t:[e for e in ROWS if e["task"]==t and e["v2_split"]=="train"] for t in tasks}
 env={"random":rr,"train":trains,"tasks":tasks,"min":min};ref=[]
 for step in range(c["completed_updates"]+1,c["completed_updates"]+51):
  env["step"]=step
  for micro in range(8):
   exec(compile(ast.Module(body=assigns,type_ignores=[]),"<original_sampler>","exec"),env)
   e=env["e"];t=env["t"];ref.append((e["id"],t,step,micro,"current"))
   if eval(compile(ast.Expression(cond),"<original_plan_condition>","eval"),env):ref.append((e["id"],t-1,step,micro,"previous_plan"))
 generated=list(forecast(c["rng"]["python"],c["completed_updates"]))[:len(ref)]
 assert ref==generated
 save("engineering/input_optimization/training_equivalence.json",{"passed":True,"time":now(),"checkpoint_update":c["completed_updates"],"prefix_max_difference":pd,"loss_difference":ld,"gradient_max_difference":gd,"adamw_update_max_difference":ud,"forecast_requests_compared":len(ref),"forecast_exact":True,"formal_training_not_modified":True})
if __name__=="__main__":
 try:main()
 except Exception as e:save("engineering/input_optimization/failure.json",{"time":now(),"stage":"training_equivalence","type":type(e).__name__,"message":str(e)[:500],"frames":[{"file":v.filename,"line":v.lineno} for v in traceback.extract_tb(e.__traceback__)]})
