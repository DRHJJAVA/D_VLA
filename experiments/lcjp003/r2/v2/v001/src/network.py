from common import *
from torch import nn
import torch.nn.functional as F
from fusion_module import expert_forward as _expert_forward
def expert_forward(front,backbone,encoded,noise,time_scalar,target):
 time_scalar=torch.as_tensor(time_scalar,device=noise.device,dtype=noise.dtype)
 return _expert_forward(front,backbone,encoded,noise,time_scalar,target)
class LoRA(nn.Module):
 def __init__(self,base,rank=16):
  super().__init__();self.base=base;self.rank=rank
  self.a=nn.Linear(base.in_features,rank,bias=False);self.b=nn.Linear(rank,base.out_features,bias=False)
  nn.init.kaiming_uniform_(self.a.weight,a=5**.5);nn.init.zeros_(self.b.weight)
  self.a.to(base.weight.device);self.b.to(base.weight.device)
 @property
 def weight(self):return self.base.weight
 @property
 def bias(self):return self.base.bias
 def forward(self,x):return self.base(x)+self.b(self.a(x))
def install_lora(expert):
 paths=[n for n,m in expert.named_modules() if n.startswith("lm_expert.") and isinstance(m,nn.Linear) and n.split(".")[-1] in ["q_proj","k_proj","v_proj","o_proj"]]
 for path in paths:
  parent=expert.get_submodule(path.rsplit(".",1)[0]);key=path.rsplit(".",1)[1]
  setattr(parent,key,LoRA(getattr(parent,key),CONFIG["lora_rank"]))
 assert len(paths)==64
 return paths
class CrossBlock(nn.Module):
 def __init__(self):
  super().__init__();self.attn=nn.MultiheadAttention(256,4,batch_first=True,dropout=0)
  self.n1=nn.LayerNorm(256);self.n2=nn.LayerNorm(256);self.ff=nn.Sequential(nn.Linear(256,512),nn.GELU(),nn.Linear(512,256))
 def forward(self,q,kv,key_padding_mask=None):
  q=self.n1(q+self.attn(q,kv,kv,key_padding_mask=key_padding_mask,need_weights=False)[0])
  return self.n2(q+self.ff(q))
class ExplainHead(nn.Module):
 def __init__(self):
  super().__init__();self.null=nn.Parameter(torch.zeros(256))
  self.trunk=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.LayerNorm(256),nn.Linear(256,256),nn.GELU())
  self.action=nn.Linear(256,16*8);self.future=nn.Linear(256,3*387)
 def forward(self,local,context):
  z=self.trunk(torch.cat([local,context.expand_as(local)],-1))
  a=self.action(z).reshape(*z.shape[:-1],16,8)
  return a[...,:6],a[...,6:],self.future(z).reshape(*z.shape[:-1],3,387)
class R2Front(nn.Module):
 def __init__(self):
  super().__init__()
  self.point=nn.Sequential(nn.Linear(396,128),nn.GELU(),nn.Linear(128,256))
  self.local=nn.ModuleList([CrossBlock(),CrossBlock()])
  self.history_in=nn.Linear(27,256)
  self.history=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,4,512,dropout=0,batch_first=True),2,enable_nested_tensor=False)
  self.word=nn.Linear(960,256);self.roles=nn.ModuleList([CrossBlock(),CrossBlock()])
  self.role_out=nn.Linear(256,3);self.null_role=nn.Linear(256,1);self.prior_strength=nn.Parameter(torch.tensor(.1))
  self.relative=nn.ModuleList([CrossBlock(),CrossBlock()])
  self.score=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))
  self.coarse=nn.Sequential(nn.Linear(12,256),nn.GELU(),nn.Linear(256,256))
  self.shared_queries=nn.Parameter(torch.randn(32,256)*.02)
  self.base_attention=CrossBlock();self.part_attention=CrossBlock()
  self.base_projection=nn.Linear(256,256);self.part_projection=nn.Linear(256,256)
  self.base_norm=nn.LayerNorm(256);self.part_norm=nn.LayerNorm(256)
  self.token_position=nn.Parameter(torch.randn(4,256)*.02)
  self.geometry_token=nn.Linear(3,256);self.plan=nn.Linear(7,256)
  self.gate=nn.Sequential(nn.Linear(770,128),nn.GELU(),nn.Linear(128,1))
  nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,float(np.log(.25/.75)))
  self.adapter=nn.Linear(256,960)
  self.motion_in=nn.Linear(6,720);self.time=nn.Sequential(nn.Linear(2,720),nn.SiLU(),nn.Linear(720,720))
  self.motion_out=nn.Linear(720,6);self.grip=nn.Linear(720,2)
  self.explain=ExplainHead()
  self.register_buffer("updates",torch.tensor(0,dtype=torch.long))
  self.register_buffer("utility_enabled",torch.tensor(False))
 def cap(self):return .2+.8*min(float(self.updates)/500,1.)
 def encode(self,x,valid,h,words,priors=None,plan=None,lambda_override=None,part_only=False):
  C,P,N,D=x.shape;pm=valid.any(-1);has=pm.any(-1)
  x=x.masked_fill(~valid[...,None],0)
  z=self.point(x).reshape(C*P,N,256)
  safevalid=valid.reshape(C*P,N).clone();safevalid[~safevalid.any(-1),0]=True
  for layer in self.local:z=layer(z,z,~safevalid)
  z=z.reshape(C,P,N,256)*valid[...,None]
  patches=z.sum(2)/valid.sum(2).clamp_min(1)[...,None]
  hist=self.history(self.history_in(h)[None],mask=torch.triu(torch.ones(16,16,device=x.device,dtype=torch.bool),diagonal=1))[0]
  word=self.word(words)
  context=torch.cat([word,hist],0)
  cond=context.mean(0)
  role=patches.sum(1)/pm.sum(1).clamp_min(1)[...,None]
  for layer in self.roles:role=layer(role[None],context[None])[0]
  logits=self.role_out(role)
  if priors is not None:
   # Frozen region-language prior is weak, never an immutable target label.
   logits=logits+F.pad(priors.clamp(.001,.999).logit(),(0,1))*self.prior_strength.clamp(0,1)
  # Always include a null candidate. The action losses are computed per branch.
  pi=F.softmax(torch.cat([logits[:,0]-logits[:,2],self.null_role(cond).reshape(1)],0),0)
  dest=F.softmax(logits[:,1],0)
  relation=torch.cat([hist[-2:],word.mean(0,keepdim=True)],0)[None].expand(C,-1,-1)
  local=patches
  for layer in self.relative:local=layer(local,relation)
  score=torch.sigmoid(self.score(local).squeeze(-1))*pm
  geom=x[...,384:387]
  mean=(geom*valid[...,None]).sum((1,2))/valid.sum((1,2)).clamp_min(1)[:,None]
  lo=geom.masked_fill(~valid[...,None],float("inf")).amin((1,2));hi=geom.masked_fill(~valid[...,None],float("-inf")).amax((1,2))
  lo=torch.where(has[:,None],lo,torch.zeros_like(lo));hi=torch.where(has[:,None],hi,torch.zeros_like(hi))
  rel=(x[...,390:396]*valid[...,None]).sum((1,2))/valid.sum((1,2)).clamp_min(1)[:,None]
  coarse=self.coarse(torch.cat([mean,hi-lo,rel],-1))
  c=torch.cat([context,(coarse*dest[:,None]).sum(0,keepdim=True),coarse.mean(0,keepdim=True)],0)
  if plan is not None:c=torch.cat([c,self.plan(plan)],0)
  q=self.shared_queries[None].expand(C,32,256)
  base=self.base_norm(self.base_projection(self.base_attention(q,torch.cat([c[None].expand(C,-1,-1),coarse[:,None]],1))))
  weights=score.masked_fill(~pm,-1e4).softmax(-1)
  top=score.masked_fill(~pm,-1e4).topk(8,-1).indices
  hard=F.one_hot(top,P).to(x.dtype)
  select=hard+weights[:,None]-weights[:,None].detach()
  # Forward chooses eight patches; backward reaches every valid scored patch.
  chosen=torch.einsum("ckp,cpd->ckd",select,local)
  anchors=(geom*valid[...,None]).sum(2)/valid.sum(2).clamp_min(1)[...,None]
  chosen_xyz=torch.einsum("ckp,cpd->ckd",select,anchors)
  parttokens=(chosen[:,:,None,:]+self.token_position[None,None]+self.geometry_token(chosen_xyz)[:,:,None,:]).reshape(C,32,256)
  selected_valid=pm.gather(1,top).repeat_interleave(4,dim=1)
  selected_valid[~selected_valid.any(-1),0]=True
  part=self.part_norm(self.part_projection(self.part_attention(q,parttokens,~selected_valid)))
  role_context=role[:,None,:].expand(-1,32,-1)
  part=part+role_context*pi[:-1,None,None]
  gate_in=torch.cat([base.mean(1),part.mean(1),cond[None].expand(C,-1),pi[:-1,None],score.mean(1,keepdim=True)],-1)
  lam=self.cap()*torch.sigmoid(self.gate(gate_in))
  lam=lam*has[:,None]
  if lambda_override is not None:lam=torch.full_like(lam,float(lambda_override))*has[:,None]
  if part_only:lam=has[:,None].to(x.dtype)
  mix=(1-lam[:,:,None])*base+lam[:,:,None]*part
  prefix=self.adapter(torch.cat([c[None].expand(C,-1,-1),mix],1))
  # Null branch has the same common context and coarse geometry, no local detail.
  nullmix=base.mean(0,keepdim=True)
  nullprefix=self.adapter(torch.cat([c[None],nullmix],1))
  prefix=torch.cat([prefix,nullprefix],0)
  return {"prefix":prefix,"prefix_valid":torch.ones(prefix.shape[:2],dtype=torch.bool,device=x.device),"scores":score,"roles":logits,"pi":pi,"lambda":lam,"base":base,"part":part,"mixed":mix,"common":c,"patches":local,"patch_valid":pm,"has_part":has,"context":cond,"anchors":anchors,"top":top,"coarse":coarse}
