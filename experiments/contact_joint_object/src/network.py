import math
import torch
from torch import nn
import torch.nn.functional as F
def mlp(sizes):
 layers=[]
 for i,(a,b) in enumerate(zip(sizes[:-1],sizes[1:])):
  layers.append(nn.Linear(a,b))
  if i<len(sizes)-2:layers.append(nn.GELU())
 return nn.Sequential(*layers)
class SA(nn.Module):
 def __init__(self,cin,cout):super().__init__();self.net=mlp([cin+3,cout//2,cout])
 def forward(self,x,f,centers,idx):
  xc=x[centers];return xc,self.net(torch.cat([x[idx]-xc[:,None],f[idx]],-1)).max(1).values
class FP(nn.Module):
 def __init__(self,cin,cout):super().__init__();self.net=mlp([cin,cout,cout])
 def forward(self,skip,coarse,idx,w):return self.net(torch.cat([skip,(coarse[idx]*w[...,None]).sum(1)],-1))
class PointEncoder(nn.Module):
 def __init__(self):
  super().__init__();self.visual=nn.Linear(384,128);self.initial=mlp([134,128,128])
  self.sa1=SA(128,128);self.sa2=SA(128,256);self.sa3=SA(256,256)
  self.fp3=FP(512,256);self.fp2=FP(384,128);self.fp1=FP(256,128)
 def forward(self,x,normal,feat,g):
  f=self.initial(torch.cat([self.visual(feat),x,normal],-1))
  x1,f1=self.sa1(x,f,g['s1'],g['n1']);x2,f2=self.sa2(x1,f1,g['s2'],g['n2']);x3,f3=self.sa3(x2,f2,g['s3'],g['n3'])
  u2=self.fp3(f2,f3,g['i32'],g['w32']);u1=self.fp2(f1,u2,g['i21'],g['w21'])
  return self.fp1(f,u1,g['i10'],g['w10'])
class Cross(nn.Module):
 def __init__(self):
  super().__init__();self.attn=nn.MultiheadAttention(256,4,batch_first=True);self.ln1=nn.LayerNorm(256);self.ln2=nn.LayerNorm(256);self.ff=mlp([256,512,256])
 def forward(self,p,f):
  z=self.attn(p[None],f[None],f[None],need_weights=False)[0][0];p=self.ln1(p+z);return self.ln2(p+self.ff(p))
class JointModel(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=PointEncoder();self.point=nn.Linear(128,256);self.finger=mlp([8,128,128]);self.finger_proj=nn.Linear(128,256)
  self.cross=nn.ModuleList([Cross(),Cross()]);self.pos=mlp([3,128,256])
  self.context=nn.Linear(402,256)
  layer=nn.TransformerEncoderLayer(256,8,1024,.1,batch_first=True,norm_first=True)
  self.temporal=nn.TransformerEncoder(layer,4);self.relation=mlp([647,256,128,3])
  self.preference=mlp([256,128,64,3]);self.visual_decoder=mlp([256,256,384])
 def forward(self,b,mask_visual=False):
  x=b['xyz'];center=x.mean(0);feat=b['features']
  masked=torch.zeros(len(x),dtype=torch.bool,device=x.device)
  if mask_visual:
   masked=torch.rand(len(x),device=x.device)<.5;feat=feat.masked_fill(masked[:,None],0)
  f=self.encoder(x-center,b['normal'],feat,b['graph'])
  fn=self.finger(torch.cat([b['finger_nodes']-center,b['finger_normals'],b['finger_side']],-1))
  fp=self.finger_proj(fn)+self.pos(b['finger_nodes']-center)
  p=self.point(f)+self.pos(x-center)
  for layer in self.cross:p=layer(p,fp)
  z=torch.sigmoid(self.preference(p));lo=z[:,0];hi=lo+(1-lo)*z[:,1];score=lo+(hi-lo)*z[:,2]
  # Current score path ends here. No temporal context or future pose reaches it.
  ctx=self.context(b['context'])
  length=len(ctx);position=torch.arange(length,device=x.device).float()[:,None];scale=torch.exp(torch.arange(0,256,2,device=x.device).float()*(-math.log(10000)/256))
  pe=torch.zeros_like(ctx);pe[:,0::2]=torch.sin(position*scale);pe[:,1::2]=torch.cos(position*scale)
  valid=b['context_valid']
  ctx=self.temporal((ctx+pe)[None],src_key_padding_mask=(~valid)[None])[0]
  temporal=(ctx*valid[:,None]).sum(0)/valid.sum().clamp_min(1)
  edge=[]
  for side in range(2):
   nodes=fn[side*64:(side+1)*64];xyz=b['finger_nodes'][side*64:(side+1)*64]
   ids=torch.cdist(x[None],xyz[None])[0].topk(8,largest=False).indices
   near=nodes[ids].mean(1)
   geom=torch.cat([b['closest'][:,side]-x,b['surface_normal'][:,side],b['sdf'][:,side,None]],-1)
   edge.append(self.relation(torch.cat([p,temporal.expand(len(x),-1),near,geom],-1)))
  edge=torch.stack(edge,1)
  return {'contact':edge[...,0].sigmoid(),'stick':edge[...,1].sigmoid(),'sigma_multiplier':.5+3.5*edge[...,2].sigmoid(),'score':score,'lower':lo,'upper':hi,'masked':masked,'visual_prediction':self.visual_decoder(p),'point_features':p}

 def score_current(self,b):
  x=b['xyz'];center=x.mean(0);f=self.encoder(x-center,b['normal'],b['features'],b['graph'])
  fn=self.finger(torch.cat([b['finger_nodes']-center,b['finger_normals'],b['finger_side']],-1))
  fp=self.finger_proj(fn)+self.pos(b['finger_nodes']-center);p=self.point(f)+self.pos(x-center)
  for layer in self.cross:p=layer(p,fp)
  z=torch.sigmoid(self.preference(p));lo=z[:,0];hi=lo+(1-lo)*z[:,1];score=lo+(hi-lo)*z[:,2]
  return {'score':score,'lower':lo,'upper':hi}

def skew(x):
 a,b,c=x.unbind(-1);z=torch.zeros_like(a)
 return torch.stack([z,-c,b,c,z,-a,-b,a,z],-1).reshape(*x.shape[:-1],3,3)
def apply_twist(x,xi,center):
 v=xi[:3];w=xi[3:];K=skew(w);theta=w.norm().clamp_min(1e-8);R=torch.eye(3,device=x.device)+torch.sin(theta)/theta*K+(1-torch.cos(theta))/theta.square()*(K@K)
 return (x-center)@R.T+center+v
def solve_motion(b,o,free=False):
 x=b['xyz'];center=x.mean(0);sigma=b['measurement_sigma'];prior=b['history_twist']
 xi=prior.clone();I=torch.eye(6,device=x.device);linear_scale=torch.tensor([1.,1.,1.,.05,.05,.05],device=x.device)
 # Fixed metric scaling: no arbitrary learned gain multiplying contact.
 reg=torch.diag(linear_scale.square())/(sigma.square()+1e-8)
 for _ in range(3):
  xp=apply_twist(x,xi,center);J=torch.cat([torch.eye(3,device=x.device).expand(len(x),-1,-1),-skew(xp-center)],-1)
  A=reg.clone();rhs=reg@(prior-xi)
  if not free:
   for side in range(2):
    n=b['surface_normal'][:,side];d=b['finger_displacement'][:,side]-(xp-x)
    cn=o['contact'][:,side];stick=o['stick'][:,side]
    cov=(sigma*o['sigma_multiplier'][:,side]).square().clamp_min(1e-8)
    W=(n[:,:,None]*n[:,None,:]+stick[:,None,None]*(torch.eye(3,device=x.device)[None]-n[:,:,None]*n[:,None,:]))*(cn/cov*b['point_area'])[:,None,None]
    A=A+torch.einsum('nai,nab,nbj->ij',J,W,J)
    rhs=rhs+torch.einsum('nai,nab,nb->i',J,W,d)
  # Environment is a measured dominant plane and only resists penetration.
  if b['environment_plane_valid']:
   plane=b['environment_plane'];dist=xp@plane[:3]+plane[3]
   weight=torch.sigmoid(-dist/sigma)/(len(x)*sigma.square())
   Jn=torch.einsum('a,nai->ni',plane[:3],J)
   A=A+torch.einsum('ni,n,nj->ij',Jn,weight,Jn)
   rhs=rhs-torch.einsum('ni,n,n->i',Jn,weight,dist.clamp_max(0))
  step=torch.linalg.solve(A+I*1e-4,rhs);xi=xi+step
 return xi
def joint_loss(model,b,online=None,free=False):
 o=model(b,mask_visual=True);xi=solve_motion(b,o,free)
 pred=apply_twist(b['motion_source'],xi,b['xyz'].mean(0));err=(pred-b['motion_target'])
 sigma=b['measurement_sigma']
 # Held-out observed correspondences are not used to solve xi.
 motion=(F.smooth_l1_loss(err/sigma,torch.zeros_like(err),reduction='none')).mean()
 gap_likelihood=torch.exp(-.5*(b['sdf']/sigma).square())
 geom=(-torch.log((1-o['contact']+o['contact']*gap_likelihood).clamp_min(1e-7)).mean(1)*b['point_area']).sum()
 pen=((o['contact']*F.relu(-b['sdf']/sigma).square()).mean(1)*b['point_area']).sum()
 complexity=(o['contact'].mean(1)*b['point_area']).sum()
 mask=o['masked'];visual=F.mse_loss(o['visual_prediction'][mask],b['features'][mask]) if mask.any() else pred.sum()*0
 # Reprojection/depth supervision uses measured next-view pixel/depth.
 K=b['target_K'];C=b['target_camera']
 pc=(pred-C[:3,3])@C[:3,:3]
 uv=pc[:,:2]/pc[:,2:].clamp_min(.001)*torch.stack([K[0,0],K[1,1]])+K[:2,2]
 reproj=F.smooth_l1_loss((uv-b['motion_target_uv'])/256,torch.zeros_like(uv))
 depth=F.smooth_l1_loss((pc[:,2]-b['motion_target_depth'])/sigma,torch.zeros_like(pc[:,2]))
 # Gaussian NLL restricts uncertainty inflation, independently of geometric penalties.
 mult=o['sigma_multiplier'].mean();nll=.5*(err.square().sum(-1)/(sigma*mult).square()+6*torch.log(mult)).mean()
 if len(pred)==0:
  motion=o['contact'].sum()*0;depth=motion;reproj=motion;nll=motion
 scoreloss=o['contact'].sum()*0;consistency=o['contact'].sum()*0;active=0
 if online is not None:
  low=online['low'];high=online['high'];weight=(1-(high-low)).clamp(0,1)
  midpoint=(low+high)/2;scoreloss=(weight*(o['score']-midpoint).square()).sum()/weight.sum().clamp_min(1)
  scoreloss=scoreloss+(F.relu(o['lower']-low).square()+F.relu(high-o['upper']).square()).mean()
  active=int((weight>0).sum())
  if 'contact' in online:
   consistency=((o['contact']-online['contact']).square()*online['contact_reliability']).mean()
 total=motion+depth+reproj+.1*visual+.01*nll+.1*(geom+pen+.01*complexity)+.1*consistency+scoreloss
 return total,{'loss':float(total.detach()),'motion':float(motion.detach()),'depth':float(depth.detach()),'reprojection':float(reproj.detach()),'masked_feature':float(visual.detach()),'gap':float(geom.detach()),'penetration':float(pen.detach()),'score_loss':float(scoreloss.detach()),'active_score_points':active,'mean_contact':float(o['contact'].mean().detach())},o
