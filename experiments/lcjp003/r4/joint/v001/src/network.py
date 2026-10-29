import math
import torch
from torch import nn
from torch.nn import functional as F
from primitives import LoRA,KVInputAdapter
def mlp(*dims):
 layers=[]
 for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
  layers.append(nn.Linear(a,b))
  if i<len(dims)-2:layers.append(nn.GELU())
 return nn.Sequential(*layers)
class Cross(nn.Module):
 def __init__(self):
  super().__init__();self.qn=nn.LayerNorm(256);self.kn=nn.LayerNorm(256)
  self.attn=nn.MultiheadAttention(256,8,batch_first=True,dropout=0)
  self.ff=nn.Sequential(nn.LayerNorm(256),mlp(256,512,256))
 def forward(self,q,kv):
  q=q+self.attn(self.qn(q),self.kn(kv),self.kn(kv),need_weights=False)[0]
  return q+self.ff(q)
class DirectSegment(nn.Module):
 def __init__(self):
  super().__init__();self.visual=nn.Conv2d(384,256,1);self.text=nn.Linear(960,256)
  self.roles=nn.Parameter(torch.randn(2,256)*.02);self.pos=nn.Linear(2,256,bias=False)
  self.layers=nn.ModuleList([Cross() for _ in range(4)])
  self.up=nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1),nn.GELU(),nn.ConvTranspose2d(128,64,4,2,1),nn.GELU(),nn.Conv2d(64,2,1))
  self.exist=mlp(256,128,2)
 def forward(self,d):
  x=self.visual(d['features'].float().permute(0,3,1,2));v,c,h,w=x.shape
  yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device),torch.linspace(-1,1,w,device=x.device),indexing='ij')
  pos=self.pos(torch.stack([xx,yy],-1).reshape(-1,2))
  tokens=x.flatten(2).transpose(1,2)+pos[None]
  text=self.text(d['words'])[None].expand(v,-1,-1)
  kv=torch.cat([text,self.roles[None].expand(v,-1,-1)],1)
  for layer in self.layers:tokens=layer(tokens,kv)
  logits=F.interpolate(self.up(tokens.transpose(1,2).reshape(v,c,h,w)),size=(256,256),mode='bilinear',align_corners=False)
  return {'logits':logits,'exist':self.exist(tokens.mean(1)),'visual':x,'text':text[:1],'coarse':x.mean((2,3)).mean(0)[None,None]}
def fps(x,n=64):
 chosen=[];distance=torch.full((len(x),),float('inf'),device=x.device)
 i=torch.argmax(((x-x.mean(0))**2).sum(-1))
 for _ in range(n):
  chosen.append(i);distance=torch.minimum(distance,((x-x[i])**2).sum(-1));i=distance.argmax()
 return torch.stack(chosen)
def points_from_depth(d,prob,teacher_mask=None):
 # Only the explicit warmup caller supplies teacher_mask.
 with torch.no_grad():
  z=d['depth'];dev=z.device
  yy,xx=torch.meshgrid(torch.arange(256,device=dev),torch.arange(256,device=dev),indexing='ij')
  k=d['K'];p=torch.stack([(xx[None]+.375-k[:,0,2,None,None])*z/k[:,0,0,None,None],
   (yy[None]+.875-k[:,1,2,None,None])*z/k[:,1,1,None,None],z],-1)
  world=torch.einsum('vij,vhwj->vhwi',d['T'][:,:3,:3],p)+d['T'][:,:3,3,None,None].permute(0,2,3,1)
  dx=torch.roll(world,-1,2)-torch.roll(world,1,2);dy=torch.roll(world,-1,1)-torch.roll(world,1,1)
  normal=F.normalize(torch.cross(dx,dy,dim=-1),dim=-1)
  eligible=torch.isfinite(world).all(-1)&(z>0)&~d['robot']
  mask=(prob.detach()>.5)&eligible
  if teacher_mask is not None:mask|=(teacher_mask>0.5)&eligible
  flat=torch.nonzero(mask.flatten()).flatten();count=len(flat)
  if count:
   ids=flat[torch.linspace(0,count-1,min(count,2048),device=dev).long()]
   valid=torch.arange(2048,device=dev)<len(ids)
   ids=ids[torch.arange(2048,device=dev)%len(ids)]
  else:ids=torch.zeros(2048,dtype=torch.long,device=dev);valid=torch.zeros(2048,dtype=torch.bool,device=dev)
  xyz=world.reshape(-1,3)[ids];norm=normal.reshape(-1,3)[ids]
  xyz=torch.nan_to_num(xyz);norm=torch.nan_to_num(norm)
  rel=(xyz-d['ee'])@d['rotation']
  normrel=norm@d['rotation'];extent=xyz[valid].amax(0)-xyz[valid].amin(0) if valid.any() else xyz.new_ones(3)
  center=xyz[valid].mean(0) if valid.any() else d['ee']
  unit=(xyz-center)/extent.clamp_min(1e-4)
  rgb=d['rgb'].reshape(-1,3)[ids].float()/255.
  geom=torch.cat([rel,unit,normrel,rgb],-1)*valid[:,None]
  cam=ids//65536;pix=ids%65536;uv=torch.stack([pix%256,pix//256],-1)
  anchors=fps(rel,64);dist=torch.cdist(rel[anchors],rel)
  dd,kn=dist.topk(32,largest=False)
  # Continuous physical scale plus surface orientation; never join distant surfaces.
  radius=torch.linalg.vector_norm(extent).clamp(.016,1.)/8
  neighbor=(dd<=radius)&valid[kn]&((norm[anchors,None]*norm[kn]).sum(-1)>.25)
  neighbor[:,0]=valid[anchors]
  di,inter=torch.cdist(rel,rel[anchors]).topk(3,largest=False)
  iw=1/(di+1e-6);iw=iw*((norm[:,None]*norm[anchors][inter]).sum(-1)>.25)*(di<=radius)
  iw=iw/iw.sum(-1,keepdim=True).clamp_min(1e-8)
 return {'xyz':xyz,'rel':rel,'normal':norm,'geom':geom,'ids':ids,'cam':cam,'uv':uv,
 'valid':valid,'anchors':anchors,'knn':kn,'neighbor':neighbor,'interp':inter,'iw':iw,'extent':extent,'point_count':count}
class R4Network(nn.Module):
 def __init__(self,expert):
  super().__init__();self.expert=expert
  for p in expert.parameters():p.requires_grad=False
  for i,layer in enumerate(expert.layers):
   for name in ['q_proj','k_proj','v_proj','o_proj']:
    original=getattr(layer.self_attn,name);adapted=LoRA(original)
    if name in ['k_proj','v_proj'] and original.in_features!=720:
     reference=getattr(expert.layers[i-1].self_attn,name).base
     adapted=KVInputAdapter(adapted,reference)
    setattr(layer.self_attn,name,adapted)
  self.segment=DirectSegment()
  self.point=mlp(268,128,256);self.neighbor=mlp(259,256,256)
  self.hist_proj=nn.Linear(27,256);self.hist_pos=nn.Parameter(torch.randn(1,16,256)*.02)
  self.hist=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,8,512,dropout=0,batch_first=True,norm_first=True,activation='gelu'),2,enable_nested_tensor=False)
  self.finger=mlp(6,128,256)
  self.part_self=Cross();self.part_context=Cross();self.part_finger=Cross()
  self.score=mlp(512,256,128,1);self.refine=mlp(256,128,1)
  self.queries=nn.Parameter(torch.randn(1,8,256)*.02)
  self.value_norm=nn.LayerNorm(256);self.value=nn.Linear(256,256,bias=False)
  self.coordinate_value=nn.Linear(6,256,bias=False)
  self.key=nn.Linear(256,256,bias=False);self.base_read=Cross()
  self.base_norm=nn.LayerNorm(256);self.gate=mlp(260,128,1)
  nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,math.log(.05/.95))
  self.statistics=nn.Linear(4,256,bias=False)
  self.prefix_proj=nn.Linear(256,720);self.plan_proj=nn.Linear(7,720);self.noise_proj=nn.Linear(6,720)
  self.action_pos=nn.Parameter(torch.randn(1,16,720)*.02)
  self.motion=mlp(720,256,256,6);self.grip=mlp(720,128,2);self.switch=mlp(720,64,1)
 def encode(self,d,teacher_mask=None,score_override=None,step=3000):
  seg=self.segment(d);prob=seg['logits'][:,0].sigmoid()
  p=points_from_depth(d,prob,teacher_mask)
  cam=p['cam'];uv=p['uv'];grid=(uv.float()/255.*2-1)
  # Projected visual feature gathering reuses the very same encoder map.
  visual=F.interpolate(seg['visual'],size=(256,256),mode='bilinear',align_corners=False).permute(0,2,3,1).reshape(-1,256)[p['ids']]
  feat=self.point(torch.cat([visual,p['geom']],-1))*p['valid'][:,None]
  kn=p['knn'];anc=p['anchors'];rel=p['rel']
  nh=self.neighbor(torch.cat([feat[kn],rel[kn]-rel[anc,None]],-1))
  masked=nh.masked_fill(~p['neighbor'][...,None],-1e4)
  grouped=masked.max(1).values;group_valid=p['neighbor'].any(1)
  grouped=torch.where(group_valid[:,None],grouped,torch.zeros_like(grouped))[None]
  hist=self.hist(self.hist_proj(d['h'])[None]+self.hist_pos);fingers=self.finger(d['fingers'])[None]
  context=torch.cat([hist,fingers,seg['text']],1)
  h=self.part_self(grouped,grouped);h=self.part_finger(h,fingers);h=self.part_context(h,context)
  cond=context.mean(1,keepdim=True).expand(-1,64,-1)
  logits=self.score(torch.cat([h,cond],-1)).squeeze(0).squeeze(-1)
  residual=.25*torch.tanh(self.refine(feat).squeeze(-1))
  point_logit=(logits[p['interp']]*p['iw']).sum(-1)+residual
  point_score=point_logit.sigmoid()*p['valid']
  group_score=logits.sigmoid()*group_valid
  if score_override=='uniform':
   group_score=torch.ones_like(group_score)*group_score[group_valid].mean() if group_valid.any() else group_score
  elif score_override=='shuffle':group_score=group_score.roll(23)
  elif score_override=='wrong':group_score=1-group_score
  elif score_override=='zero':group_score=torch.zeros_like(group_score)
  elif score_override=='one':group_score=torch.ones_like(group_score)*group_valid
  group_score=group_score*group_valid
  # All 64 groups remain available. Gate values AFTER normalization.
  query=self.queries
  key=self.key(h)
  attention=(query@key.transpose(-1,-2)/16).masked_fill(~group_valid[None,None],-1e4).softmax(-1)
  coord=torch.cat([rel[anc],p['extent'][None].expand(64,-1)],-1)
  value=self.value(self.value_norm(h))+self.coordinate_value(coord)[None]
  part=(attention*group_score[None,None])@value
  stats=torch.stack([group_score.mean(),group_score.std(unbiased=False),prob.mean(),group_valid.float().mean()])
  part=part+self.statistics(stats)[None,None]
  # No object token/ungated dense point path reaches the action expert.
  coarse=torch.cat([seg['coarse'],hist[:,-1:],fingers.mean(1,keepdim=True),seg['text'].mean(1,keepdim=True)],1)
  base=self.base_norm(self.base_read(query,coarse))
  lam=self.gate(torch.cat([coarse.mean(1),stats[None]],-1)).sigmoid()
  lam=lam*(.1+.9*min(float(step)/250,1.)) if self.training else lam
  mix=(1-lam[...,None])*base+lam[...,None]*part
  return {'seg':seg,'points':p,'point_score':point_score,'point_logit':point_logit,'group_score':group_score,
   'group_logit':logits,'mix':mix,'base':base,'part':part,'lambda':lam,'hist':hist[:,-1:],
   'finger':fingers.mean(1,keepdim=True),'text':seg['text'],'stats':stats}
 def generate(self,e,noise,plan=None):
  seq=torch.cat([e['mix'],e['hist'],e['finger'],e['text']],1)
  prefix=self.prefix_proj(seq)
  g=noise.shape[0];prefix=prefix.expand(g,-1,-1)
  if plan is None:plan=noise.new_zeros((g,16,7))
  elif plan.shape[0]==1:plan=plan.expand(g,-1,-1)
  prefix=torch.cat([prefix,self.plan_proj(plan)],1)
  action=self.noise_proj(noise)+self.action_pos
  tok=torch.cat([prefix,action],1);length=tok.shape[1];pn=prefix.shape[1]
  mask=tok.new_zeros((g,1,length,length));mask[:,:,:pn,pn:]=torch.finfo(tok.dtype).min
  out=self.expert(inputs_embeds=tok,attention_mask=mask,use_cache=False).last_hidden_state[:,-16:]
  return self.motion(out),self.grip(out),self.switch(out).squeeze(-1)
