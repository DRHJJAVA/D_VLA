import torch
from torch import nn
from torch.nn import functional as F
class SurfacePredictor(nn.Module):
 def __init__(self):
  super().__init__()
  self.encoder=nn.Sequential(nn.Conv2d(28,32,3,padding=1),nn.GroupNorm(4,32),nn.GELU(),nn.Conv2d(32,48,3,stride=2,padding=1),nn.GroupNorm(6,48),nn.GELU(),nn.Conv2d(48,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU())
  self.condition=nn.Sequential(nn.Linear(50,64),nn.GELU(),nn.Linear(64,128))
  self.head=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.GELU(),nn.Conv2d(64,12,1))
 def forward(self,images,state,conditioned):
  f=self.encoder(images)
  c=state if conditioned else torch.zeros_like(state)
  gamma,beta=self.condition(c).chunk(2,-1)
  f=f*(1+.1*gamma[:,:,None,None])+.1*beta[:,:,None,None]
  return self.head(f)
def at_points(dense,grid):
 v=F.grid_sample(dense,grid[:,None],mode='bilinear',padding_mode='zeros',align_corners=False).squeeze(2).transpose(1,2)
 v=v.reshape(v.shape[0],v.shape[1],3,4).permute(0,2,1,3)
 mean=v[...,:3]
 sigma=.2+1.8*torch.sigmoid(v[...,3])
 return mean,sigma
def gaussian(mean,sigma,target):
 return .5*((mean-target)/sigma[...,None]).square().sum(-1)+3*torch.log(sigma)+1.5*1.8378770664093453
