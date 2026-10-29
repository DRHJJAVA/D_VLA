from common import *
from network import *
def to_device(d,device="cuda"):
 return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in d.items()}
def branch_error(m,g,target,grip,noise):
 # Conditional flow: x_t=t noise+(1-t) target, velocity noise-target.
 motion=(m-(noise-target[None])).square().mean((1,2))
 grip_loss=F.cross_entropy(g.reshape(-1,2),grip[None].expand(g.shape[0],-1).reshape(-1),reduction="none").reshape(g.shape[0],16).mean(1)
 return motion+.1*grip_loss
def explain_error(head,local,context,d):
 m,g,f=head(local,context)
 shape=local.shape[:-1]
 target=d["target"].expand(*shape,16,6);grip=d["grip"].expand(*shape,16)
 a=(m-target).square().mean((-2,-1))+.1*F.cross_entropy(g.reshape(-1,2),grip.reshape(-1),reduction="none").reshape(*shape,16).mean(-1)
 return a,f
def objective(front,expert,ema,d,update,plan=None,noise=None,time_value=None):
 device=d["x"].device
 enc=front.encode(d["x"],d["valid"],d["h"],d["words"],d["priors"],plan)
 C=d["x"].shape[0];B=C+1
 if noise is None:noise=torch.randn(1,16,6,device=device).expand(B,-1,-1)
 if time_value is None:time_value=float(torch.rand((),device=device).clamp(.001,.999))
 m,g=expert_forward(front,expert,enc,noise,time_value,d["target"])
 mixed_err=branch_error(m,g,d["target"],d["grip"],noise)
 part_prefix=front.adapter(torch.cat([enc["common"][None].expand(C,-1,-1),enc["part"]],1))
 penc={"prefix":torch.cat([part_prefix,enc["prefix"][-1:]],0),"prefix_valid":enc["prefix_valid"]}
 pm,pg=expert_forward(front,expert,penc,noise,time_value,d["target"])
 part_err=branch_error(pm,pg,d["target"],d["grip"],noise)
 # Auxiliary head sees local observations and common causal context; never target actions.
 local=enc["patches"];context=enc["context"]
 ae,pfuture=explain_error(front.explain,local,context,d)
 null_local=front.explain.null[None,None].expand_as(local)
 nae,nfuture=explain_error(front.explain,null_local,context,d)
 fv=d["future_valid"]&enc["patch_valid"][...,None]
 fe=(pfuture-d["future"]).square().mean(-1)
 nfe=(nfuture-d["future"]).square().mean(-1)
 valid=enc["patch_valid"].float()
 aux_a=((ae+nae)*valid).sum()/valid.sum().clamp_min(1)
 aux_f=((fe+nfe)*fv).sum()/fv.sum().clamp_min(1)
 with torch.no_grad():
  tea,tef=explain_error(ema,local.detach(),context.detach(),d)
  tna,tnf=explain_error(ema,ema.null[None,None].expand_as(local),context.detach(),d)
  # Paired null/local errors always share exactly the same observed dimensions.
  tf=((tef-d["future"]).square().mean(-1)*fv).sum(-1)/fv.sum(-1).clamp_min(1)
  tnferr=((tnf-d["future"]).square().mean(-1)*fv).sum(-1)/fv.sum(-1).clamp_min(1)
  advantage=(tna-tea)+(tnferr-tf)*fv.any(-1)
  cand_adv=(advantage*valid).sum(-1)/valid.sum(-1).clamp_min(1)
  # Measured action explanations provide responsibility from update one;
  # EMA local utility is admitted only after cross-demo qualification.
  action_adv=mixed_err[-1]-mixed_err[:-1]
  if front.utility_enabled:action_adv=action_adv+cand_adv
  q=(enc["pi"].detach().clamp_min(1e-6).log()+torch.cat([action_adv,action_adv.new_zeros(1)])).softmax(0)
  q=.98*q+.02/B
  target_score=torch.sigmoid(advantage)
 role=-(q*enc["pi"].clamp_min(1e-6).log()).sum()
 utility=(F.binary_cross_entropy(enc["scores"].clamp(1e-5,1-1e-5),target_score,reduction="none")*valid).sum()/valid.sum().clamp_min(1)
 # Only actual corresponding physical patches are constrained across cameras.
 centers=enc["anchors"].reshape(-1,3);scores=enc["scores"].reshape(-1)
 view=d["views"][:,None].expand(C,32).reshape(-1)
 cv=scores.sum()*0
 if C>1:
  with torch.no_grad():
   dist=torch.cdist(centers,centers)
   mask=(view[:,None]!=view[None])&valid.reshape(-1)[:,None].bool()&valid.reshape(-1)[None].bool()
   dist=dist.masked_fill(~mask,float("inf"));best,j=dist.min(-1)
   match=best<.005
  if match.any():cv=(scores[match]-scores[j[match]]).square().mean()
 w=CONFIG["loss_weights"]
 mixed=(enc["pi"]*mixed_err).sum()
 # This branch coefficient does not depend on lambda.
 object_weights=enc["pi"][:-1]/enc["pi"][:-1].sum().clamp_min(1e-6)
 independent=((.98*object_weights+.02/C)*part_err[:-1]).sum()
 planloss=mixed*0
 if plan is not None:
  # Compare predicted clean remaining plan; quality weight is detached.
  clean=time_value*noise+(1-time_value)*d["target"][None]-time_value*m
  reference=plan[:,:6]
  quality=torch.exp(-(reference-d["target"]).square().mean()).detach()
  planloss=quality*((clean-reference[None]).square().mean((1,2))*enc["pi"]).sum()
 loss=w["mixed"]*mixed+w["part"]*independent+w["role"]*role+w["aux_action"]*aux_a+w["aux_future"]*aux_f+w["crossview"]*cv+w["plan"]*planloss
 if front.utility_enabled:loss=loss+w["utility"]*utility
 metrics={k:float(v.detach()) for k,v in {"loss":loss,"mixed":mixed,"independent_part":independent,"role":role,"utility":utility,"aux_action":aux_a,"aux_future":aux_f,"crossview":cv,"lambda_mean":enc["lambda"].mean(),"score_std":enc["scores"][enc["patch_valid"]].std(unbiased=False),"null_probability":enc["pi"][-1]}.items()}
 return loss,metrics,enc
@torch.no_grad()
def infer(front,expert,d,noise=None,plan=None):
 enc=front.encode(d["x"],d["valid"],d["h"],d["words"],d["priors"],plan)
 selected=int(enc["pi"].argmax());e={"prefix":enc["prefix"][selected:selected+1],"prefix_valid":enc["prefix_valid"][selected:selected+1]}
 x=torch.randn(1,16,6,device=d["x"].device) if noise is None else noise.clone()
 grip=None
 # expert_forward interpolates noise and target; setting both to x evaluates x_t=x.
 for k in range(CONFIG["flow_steps_eval"]):
  t=1-k/CONFIG["flow_steps_eval"]
  v,grip=expert_forward(front,expert,e,x,t,x[0])
  x=x-v/CONFIG["flow_steps_eval"]
 return x[0],grip[0],enc
