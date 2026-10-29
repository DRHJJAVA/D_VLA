import torch
from torch import nn
from torch.nn import functional as F
from r5_modules import InstanceFront,R5Parts,surface_score
from r5_action import R5Action
from r5_geometry import points_from_depth,fps
def covariance_eigenvalues(x):
 if len(x)<2:return x.new_zeros(3)
 centered=x-x.mean(0)
 cov=centered.T@centered/len(x)
 if int(torch.count_nonzero(cov))==0:return x.new_zeros(3)
 return torch.linalg.eigvalsh(cov).clamp_min(0)
def world_grid(d):
 device=d["depth"].device
 uv=torch.stack(torch.meshgrid(torch.arange(32,device=device),torch.arange(32,device=device),indexing="ij"),-1)[...,[1,0]].float()*8+3
 uv=uv.long();out=[]
 for c in range(2):
  z=d["depth"][c,uv[...,1],uv[...,0]]
  pix=uv.float()+uv.new_tensor([.375,.875],dtype=torch.float32)
  ray=torch.cat([pix,torch.ones_like(pix[...,:1])],-1)@torch.linalg.inv(d["K"][c]).T
  cam=ray*z[...,None];xyz=cam@d["T"][c,:3,:3].T+d["T"][c,:3,3]
  xyz=torch.where((torch.isfinite(z)&(z>0))[...,None],xyz,torch.zeros_like(xyz))
  out.append(xyz)
 return torch.stack(out)
def cloud_summary(d,prob):
 grid=world_grid(d)
 uv=torch.stack(torch.meshgrid(torch.arange(32,device=grid.device),torch.arange(32,device=grid.device),indexing="ij"),-1)[...,[1,0]]*8+3
 depth=d["depth"][:,uv[...,1],uv[...,0]]
 v=(prob[:,uv[...,1],uv[...,0]]>.5)&(~d["robot"][:,uv[...,1],uv[...,0]])&torch.isfinite(depth)&(depth>0)
 x=grid[v]
 if not len(x):return d["ee"].new_zeros(15)
 center=x.mean(0);ext=x.amax(0)-x.amin(0);rel=center-d["ee"]
 eig=covariance_eigenvalues(x)
 visible=v.float().sum((1,2))/v.float().sum().clamp_min(1)
 return torch.cat([center,ext,rel,eig,visible,v.float().mean()[None]])
def multiscale(p):
 x=p["xyz"];v=p["valid"];valid_ids=v.nonzero().flatten()
 groups=torch.zeros(64,32,device=x.device,dtype=torch.long)
 ok=torch.zeros(64,32,device=x.device,dtype=torch.bool)
 geo=x.new_zeros(64,16);rel=x.new_zeros(64,32,3)
 surface=x.new_zeros(len(x),64)
 if len(valid_ids)==0:return groups,ok,rel,geo,surface
 centers=valid_ids[fps(x[valid_ids],min(16,len(valid_ids)))]
 if len(centers)<16:centers=centers[torch.arange(16,device=x.device)%len(centers)]
 fine=valid_ids[fps(x[valid_ids],min(32,len(valid_ids)))]
 if len(fine)<32:fine=fine[torch.arange(32,device=x.device)%len(fine)]
 fine=fine[-16:];anchors=torch.cat([centers,centers,centers,fine])
 diag=(x[valid_ids].amax(0)-x[valid_ids].amin(0)).norm().clamp_min(1e-4)
 radii=diag*x.new_tensor([.08]*16+[.16]*16+[.24]*16+[.08]*16)
 dist=torch.cdist(x[anchors],x)
 dist=dist.masked_fill(~v[None],float("inf"))
 dd,idx=dist.topk(32,largest=False)
 groups=idx;ok=dd<=radii[:,None]
 n=p["normal"];center_obj=x[valid_ids].mean(0)
 for j in range(64):
  ids=idx[j][ok[j]];pts=x[ids];rel[j]=x[idx[j]]-x[anchors[j]]
  if not len(ids):continue
  center=pts.mean(0);ext=pts.amax(0)-pts.amin(0)
  centered=pts-center
  eig=covariance_eigenvalues(pts)
  normal=n[ids].mean(0);normal=normal/normal.norm().clamp_min(1e-6)
  geo[j]=torch.cat([center,normal,ext,eig,center-center_obj,ok[j].float().mean()[None]])
  # Membership is bounded to the same selected-object patch, never global interpolation.
  surface[ids,j]=torch.exp(-.5*(dd[j][ok[j]]/radii[j].clamp_min(1e-6)).square())
 return groups,ok,rel,geo,surface
class R5Network(nn.Module):
 def __init__(self,scaffold):
  super().__init__();self.front=InstanceFront();self.parts=R5Parts();self.action=R5Action(scaffold)
 def front_frame(self,d):
  return self.front(d["features"].float()[None],d["words"].float()[None],world_grid(d)[None])
 def chosen(self,front,role=0):
  i=front["role"][0,role].argmax()
  if int(i)==16:return front["masks"][0,:,0]*0-30,i
  return front["masks"][0,:,i],i
 def history_summary(self,frames):
  if not frames:return None
  existing=[(i,d) for i,d in enumerate(frames) if d is not None]
  device=next(self.parameters()).device
  values=torch.zeros(len(frames),15,device=device)
  if not existing:return values
  with torch.no_grad():
   for start in range(0,len(existing),4):
    batch=existing[start:start+4];ds=[d for _,d in batch]
    f=self.front(torch.stack([d["features"].float() for d in ds]),torch.stack([d["words"].float() for d in ds]),torch.stack([world_grid(d) for d in ds]))
    for j,(i,d) in enumerate(batch):
     idx=int(f["role"][j,0].argmax())
     prob=f["masks"][j,:,idx].sigmoid() if idx<16 else torch.zeros_like(f["masks"][j,:,0])
     values[i]=cloud_summary(d,prob)
  return values
 def encode(self,d,history_frames=None,history_valid=None,score_override=None,front_override=None,causal_summaries=None):
  front=self.front_frame(d) if front_override is None else front_override
  mask,chosen=self.chosen(front)
  p=points_from_depth(d,mask.sigmoid())
  pix=F.interpolate(front["visual"][0].permute(0,3,1,2),size=(256,256),mode="bilinear",align_corners=False)
  features=pix[p["cam"],:,p["uv"][:,1],p["uv"][:,0]]
  point=torch.cat([features,p["geom"]],-1)*p["valid"][:,None]
  group,ok,rel,geom,surface=multiscale(p)
  if history_frames is None:history_frames=[d]*16
  hs=self.history_summary(history_frames[:-1]) if causal_summaries is None else causal_summaries
  if causal_summaries is not None:assert causal_summaries.shape==(15,15)
  now=cloud_summary(d,mask.sigmoid()).detach()[None]
  summaries=torch.cat([hs,now],0) if hs is not None else now
  assert len(summaries)==16
  history=torch.cat([d["h"].float(),summaries],-1)[None]
  hv=torch.ones(1,16,device=point.device,dtype=torch.bool) if history_valid is None else history_valid[None]
  v=p["valid"];total=v.float().sum().clamp_min(1)
  mean=(features*v[:,None]).sum(0)/total
  validg=ok.any(-1);gmean=(geom*validg[:,None]).sum(0)/validg.float().sum().clamp_min(1)
  coarse=torch.cat([mean,gmean])[None]
  destid=front["role"][0,1].argmax()
  destination=front["instances"][:,destid] if int(destid)<16 else front["instances"][:,0]*0
  args=(point[None],group[None],ok[None],rel[None],geom[None],history,hv,d["fingers"][None].float(),d["h"][-1:,:].float(),front["text"],coarse,destination)
  e=self.parts(*args,score_override=score_override)
  e["part_inputs"]=args
  ps,pv=surface_score(e["score"],surface[None])
  e.update({"front":front,"points":p,"group_index":group,"neighbor_valid":ok,"geometry":geom,"surface_weights":surface,"surface_score":ps[0],"surface_valid":pv[0],"target_index":chosen,"target_logits":mask,"history_summaries":summaries})
  return e
 def rescore(self,e,score):
  q=self.parts(*e["part_inputs"],score_override=score)
  result=dict(e);result.update(q)
  ps,pv=surface_score(result["score"],e["surface_weights"][None]);result["surface_score"]=ps[0];result["surface_valid"]=pv[0]
  return result
 def generate(self,e,noise):
  return self.action(e,noise)
