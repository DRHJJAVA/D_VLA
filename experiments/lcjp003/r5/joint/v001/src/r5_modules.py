import torch
from torch import nn
from torch.nn import functional as F

def mlp(*dims):
 out=[]
 for i,(a,b) in enumerate(zip(dims,dims[1:])):
  out.append(nn.Linear(a,b))
  if i<len(dims)-2:out.append(nn.GELU())
 return nn.Sequential(*out)

class DecoderBlock(nn.Module):
 def __init__(self):
  super().__init__()
  self.self_att=nn.MultiheadAttention(256,8,batch_first=True)
  self.cross=nn.MultiheadAttention(256,8,batch_first=True)
  self.n=nn.ModuleList([nn.LayerNorm(256) for _ in range(3)])
  self.ff=mlp(256,1024,256)
 def forward(self,q,context,mask=None,query_valid=None):
  z=self.n[0](q)
  qmask=None
  if query_valid is not None:
   qmask=~query_valid.clone();qmask[:,0]=False
  q=q+self.self_att(z,z,z,key_padding_mask=qmask,need_weights=False)[0]
  q=q+self.cross(self.n[1](q),context,context,key_padding_mask=mask,need_weights=False)[0]
  q=q+self.ff(self.n[2](q))
  return q if query_valid is None else q*query_valid[...,None]

class InstanceFront(nn.Module):
 def __init__(self):
  super().__init__()
  self.visual=nn.Sequential(nn.Linear(384,256),nn.LayerNorm(256))
  self.text=nn.Sequential(nn.Linear(960,256),nn.LayerNorm(256))
  self.position=mlp(3,128,256)
  self.camera=nn.Parameter(torch.randn(2,256)*.02)
  self.queries=nn.Parameter(torch.randn(18,256)*.02)
  self.layers=nn.ModuleList([DecoderBlock() for _ in range(4)])
  self.pixel=nn.Sequential(nn.Conv2d(256,128,3,padding=1),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.GELU(),nn.Conv2d(128,64,3,padding=1),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.GELU())
  self.mask_embed=nn.Linear(256,64);self.role=nn.Linear(256,256)
  self.null=nn.Parameter(torch.zeros(1,256));self.visibility=mlp(256,128,2)
 def forward(self,features,words,position):
  b,v,h,w,_=features.shape
  f=self.visual(features.float());tokens=f+self.position(position.float())+self.camera[None,:,None,None,:]
  text=self.text(words.float());context=torch.cat([tokens.reshape(b,-1,256),text],1)
  q=self.queries[None].expand(b,-1,-1)
  for layer in self.layers:q=layer(q,context)
  inst,roles=q[:,:16],q[:,16:]
  pixels=self.pixel(f.permute(0,1,4,2,3).reshape(b*v,256,h,w)).reshape(b,v,64,h*4,w*4)
  masks=torch.einsum('bmc,bvchw->bvmhw',self.mask_embed(inst),pixels)
  masks=F.interpolate(masks.flatten(0,2)[:,None],size=(256,256),mode='bilinear',align_corners=False).reshape(b,v,16,256,256)
  candidates=torch.cat([inst,self.null[None].expand(b,-1,-1)],1)
  role=torch.einsum('brd,bmd->brm',self.role(roles),candidates)/16
  return {'masks':masks,'role':role,'visibility':self.visibility(inst).transpose(1,2),'visual':f,'text':text,'instances':inst}

class HardTopParts(nn.Module):
 def __init__(self,k=8,temperature=.5):
  super().__init__();self.k=k;self.temperature=temperature
 def forward(self,score,valid):
  b,n=score.shape;k=min(self.k,n)
  masked=score.masked_fill(~valid,-1e4)
  ids=torch.argsort(masked,dim=-1,descending=True,stable=True)[:,:k]
  hv=valid.gather(1,ids);hard=F.one_hot(ids,n).to(score.dtype)*hv[:,:,None]
  # NeuralSort: each row approximates the corresponding rank.
  diff=(masked[:,:,None]-masked[:,None,:]).abs().sum(-1)
  rank=torch.arange(1,k+1,device=score.device,dtype=score.dtype)
  logits=((n+1-2*rank)[None,:,None]*masked[:,None,:]-diff[:,None,:])/self.temperature
  logits=logits.masked_fill(~valid[:,None,:],-1e4)
  soft=logits.softmax(-1)*hv[:,:,None]
  select=hard+(soft-soft.detach())
  return ids,hv,select

class R5Parts(nn.Module):
 def __init__(self):
  super().__init__()
  self.point=mlp(268,128,256);self.local=mlp(259,256,256)
  self.self_layers=nn.ModuleList([nn.TransformerEncoderLayer(256,8,1024,0.,batch_first=True,norm_first=True) for _ in range(2)])
  self.cross_layers=nn.ModuleList([DecoderBlock() for _ in range(2)])
  self.history_proj=nn.Linear(42,256)
  self.history=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,8,1024,0.,batch_first=True,norm_first=True),2,enable_nested_tensor=False)
  self.finger=mlp(6,128,256);self.state=mlp(27,256);self.state_fuse=mlp(512,256)
  self.geometry=mlp(16,128,256);self.score=mlp(768,256,128,1)
  self.top=HardTopParts();self.part=nn.Sequential(nn.Linear(272,256),nn.GELU(),nn.LayerNorm(256))
  self.score_embed=nn.Linear(1,256);self.base=mlp(272,256,256);self.slot=nn.Parameter(torch.randn(8,256)*.02)
  self.gate=mlp(768,128,1);nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
  self.base_norm=nn.LayerNorm(256);self.missing=nn.Parameter(torch.zeros(256))
 def forward(self,point_input,group_index,neighbor_valid,relative_points,geometry,history,history_valid,fingers,state,text,coarse,destination,score_override=None):
  b,n,_=point_input.shape;g=group_index.shape[1]
  batch=torch.arange(b,device=point_input.device)[:,None,None]
  pf=self.point(point_input);local=pf[batch,group_index]
  local=self.local(torch.cat([local,relative_points],-1)).masked_fill(~neighbor_valid[...,None],-torch.finfo(pf.dtype).max)
  valid=neighbor_valid.any(-1);h=local.max(2).values;h=torch.where(valid[...,None],h,torch.zeros_like(h))
  padding=~valid
  # A sentinel avoids all-masked attention NaNs; it is never an eligible part.
  safe_padding=padding.clone();safe_padding[:,0]=False
  for layer in self.self_layers:h=layer(h,src_key_padding_mask=safe_padding)
  histpadding=~history_valid;histpadding=histpadding.clone();histpadding[:,-1]=False
  causal=torch.triu(torch.ones(history.shape[1],history.shape[1],device=h.device,dtype=torch.bool),1)
  history=self.history(self.history_proj(history),mask=causal,src_key_padding_mask=histpadding)[:,-1:]
  finger=self.finger(fingers).mean(1,keepdim=True)
  st=self.state_fuse(torch.cat([self.state(state)[:,None],finger],-1))
  context=torch.cat([history,st,text],1)
  for layer in self.cross_layers:h=layer(h,context,query_valid=valid)
  logits=self.score(torch.cat([h,self.geometry(geometry),history.expand(-1,g,-1)],-1)).squeeze(-1)
  score=logits.sigmoid()
  if score_override is not None:score=score_override
  score=torch.where(valid,score,torch.zeros_like(score))
  ids,selected_valid,selection=self.top(score,valid)
  values=self.part(torch.cat([h,geometry],-1))
  selected_score=torch.einsum('bkg,bg->bk',selection,score)
  part=torch.einsum('bkg,bgd->bkd',selection,values)*selected_score[...,None]+self.score_embed(selected_score[...,None])
  base=self.base(coarse)[:,None]+self.slot[None]
  lam=self.gate(torch.cat([base,part,history.expand(-1,8,-1)],-1)).sigmoid()
  mix=(1-lam)*self.base_norm(base)+lam*part
  mix=torch.where(selected_valid[...,None],mix,self.missing[None,None])
  prefix=torch.cat([mix,history,text.mean(1,keepdim=True),destination[:,None],st],1)
  bias=torch.cat([lam.squeeze(-1)*selected_score.clamp_min(1e-4).log(),score.new_zeros(b,4)],1)
  prefix_valid=torch.cat([selected_valid,torch.ones(b,4,dtype=torch.bool,device=h.device)],1)
  return {'prefix':prefix,'prefix_bias':bias,'prefix_valid':prefix_valid,'score':score,'score_logits':logits,'part_indices':ids,'part_valid':selected_valid,'part_geometry':geometry[torch.arange(b,device=h.device)[:,None],ids],'lambda':lam,'selection':selection}

def surface_score(score,weights):
 denom=weights.sum(-1)
 value=torch.einsum('bng,bg->bn',weights,score)/denom.clamp_min(1e-8)
 return value,denom>0
