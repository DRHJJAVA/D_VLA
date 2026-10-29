import time
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from drifting import drift_loss
def weighted(x,w):return (x*w).sum()/w.sum().clamp_min(1e-8)
def instance_loss(e,y):
 f=e["front"];m=f["masks"][0];truth=y["instance_masks"].float();valid=y["instance_valid"];n=int(valid.sum())
 small=F.interpolate(m.permute(1,0,2,3),size=32,mode="bilinear",align_corners=False).sigmoid().flatten(1)
 target=F.interpolate(truth[:,:n].permute(1,0,2,3),size=32,mode="nearest").flatten(1)
 cost=1-(2*small@target.T+1)/(small.sum(1)[:,None]+target.sum(1)[None]+1)
 ri,ci=linear_sum_assignment(cost.detach().cpu().numpy())
 ri=torch.as_tensor(ri,device=m.device);ci=torch.as_tensor(ci,device=m.device)
 assigned=torch.zeros_like(m);assigned[:,ri]=truth[:,ci]
 known=torch.ones_like(m)
 bce=F.binary_cross_entropy_with_logits(m,assigned,reduction="none").mean()
 p=m.sigmoid();dice=(1-(2*(p*assigned).sum((0,2,3))+1)/((p+assigned).sum((0,2,3))+1)).mean()
 role=torch.zeros_like(f["role"][0]);role[:,ri]=y["role_distribution"][:,ci];role[:,-1]=y["role_distribution"][:,-1]
 role=role/role.sum(-1,keepdim=True).clamp_min(1e-8)
 ce=-(role*F.log_softmax(f["role"][0],-1)).sum(-1)
 roleloss=weighted(ce,y["role_weight"])
 visibility=(assigned.sum((-1,-2))>0).float()
 vis=F.binary_cross_entropy_with_logits(f["visibility"][0],visibility)
 return bce+dice+roleloss+.1*vis,{"mask_bce":bce,"mask_dice":dice,"role":roleloss,"visibility":vis},ri,ci
def response_targets(e,y):
 p=e["points"];xyz=p["xyz"].detach();tracks=y["track_xyz"].float()
 conf=y["track_confidence"].float()
 if not len(tracks) or not bool((conf>0).any()):return xyz.new_zeros(64),xyz.new_zeros(64)
 dist=torch.cdist(xyz,tracks)
 # Local geometric binding, continuous confidence; no contact-distance decision.
 dd,ix=dist.topk(min(3,len(tracks)),largest=False)
 validpoint=p["valid"]
 sigma=(p["extent"].norm()*.03).clamp_min(1e-4)
 w=torch.exp(-.5*(dd/sigma).square())*conf[ix]
 value=y["track_response"][ix]*y["track_target"][ix]
 point=(w*value).sum(-1)/w.sum(-1).clamp_min(1e-8)
 confidence=w.sum(-1).clamp_max(1)*validpoint
 g=e["group_index"];nv=e["neighbor_valid"]
 wg=confidence[g]*nv
 return (point[g]*wg).sum(-1)/wg.sum(-1).clamp_min(1e-8),wg.mean(-1)
def surface_regularizer(e):
 geo=e["geometry"];s=e["score"][0];ok=e["neighbor_valid"].any(-1)
 d=torch.cdist(geo[:,:3],geo[:,:3]);scale=(geo[:,6:9].norm(dim=-1).mean()).clamp_min(1e-4)
 normal=(geo[:,3:6]@geo[:,3:6].T).clamp_min(0)
 w=torch.exp(-d.square()/(2*scale.square()))*normal*ok[:,None]*ok[None]
 w=w*(1-torch.eye(64,device=w.device))
 return weighted((s[:,None]-s[None]).square(),w)
def view_loss(d,e):
 # One shared 3D score projects to both views. Compare learned instance probabilities
 # only where measured depth confirms the surface is visible.
 p=e["points"];xyz=p["xyz"].detach();valid=p["valid"];terms=[]
 probs=e["target_logits"].sigmoid()
 for c in range(2):
  T=d["T"][c];cam=(xyz-T[:3,3])@T[:3,:3];z=cam[:,2]
  uv=(cam@d["K"][c].T)[:,:2]/z[:,None].clamp_min(1e-6)
  uv=(uv-uv.new_tensor([.375,.875])).round().long()
  inside=valid&(z>0)&(uv[:,0]>=0)&(uv[:,0]<256)&(uv[:,1]>=0)&(uv[:,1]<256)
  ix=inside.nonzero().flatten()
  if not len(ix):continue
  pix=uv[ix];obs=d["depth"][c,pix[:,1],pix[:,0]]
  seen=(obs-z[ix]).abs()<=.006
  ix=ix[seen];pix=pix[seen]
  if not len(ix):continue
  a=probs[p["cam"][ix],p["uv"][ix,1],p["uv"][ix,0]]
  b=probs[c,pix[:,1],pix[:,0]]
  terms.append((a-b).square().mean())
 return torch.stack(terms).mean() if terms else probs.sum()*0
def temporal_loss(e,previous):
 if previous is None:return e["score"].sum()*0
 a=e["points"];b=previous["points"];av=a["valid"]&e["surface_valid"];bv=b["valid"]&previous["surface_valid"]
 if not av.any() or not bv.any():return e["score"].sum()*0
 # The two predictions are each causal; compare only locally corresponding surfaces.
 d=torch.cdist(a["xyz"][av].detach(),b["xyz"][bv].detach());dd,ix=d.min(-1)
 sigma=a["extent"].norm().clamp_min(1e-4)*.03
 w=torch.exp(-.5*(dd/sigma).square())
 return weighted((e["surface_score"][av]-previous["surface_score"][bv][ix].detach()).square(),w)
def objective(model,batch,noise,previous=None):
 d,y=batch["d"],batch["y"]
 torch.cuda.synchronize();start=time.perf_counter()
 e=model.encode(d,batch["history"],batch["history_valid"])
 torch.cuda.synchronize();encoded=time.perf_counter()
 motion,grip=model.generate(e,noise)
 torch.cuda.synchronize();generated=time.perf_counter()
 target,detail,ri,ci=instance_loss(e,y)
 pt,pw=response_targets(e,y)
 part=weighted(F.binary_cross_entropy_with_logits(e["score_logits"][0],pt,reduction="none"),pw)
 drift=drift_loss(motion.flatten(1)[None],y["target"].float().flatten()[None,None]).mean()
 grip_loss=F.cross_entropy(grip.reshape(-1,2),y["grip"].long()[None].expand(len(noise),-1).reshape(-1))
 comp={"drift":drift,"grip":grip_loss,"target":target,"part":part,"surface":surface_regularizer(e),"temporal":temporal_loss(e,previous),"view":view_loss(d,e)}
 loss=comp["drift"]+comp["grip"]+comp["target"]+.5*comp["part"]+.02*comp["surface"]+.05*comp["temporal"]+.1*comp["view"]
 metrics={k:float(v.detach()) for k,v in comp.items()}
 metrics.update({"encode_seconds":encoded-start,"action_seconds":generated-encoded,"supervision_seconds":time.perf_counter()-generated})
 metrics.update({"part_support":float(pw.sum()),"part_target_mean":float(weighted(pt,pw)),"score_mean":float(e["score"].mean().detach()),"lambda_mean":float(e["lambda"].mean().detach())})
 return loss,metrics,e,(motion,grip)
