from common13 import *
import torch.nn as nn
import torch.nn.functional as F
from model import SurfacePredictor,at_points,gaussian
class MotionPredictor(nn.Module):
 def __init__(self,initial):
  super().__init__();self.base=SurfacePredictor();self.base.load_state_dict(initial)
  first=self.base.encoder[0]
  conv=nn.Conv2d(32,32,3,padding=1)
  with torch.no_grad():
   conv.weight.zero_();conv.weight[:,:28].copy_(first.weight);conv.bias.copy_(first.bias)
  self.base.encoder[0]=conv
  self.local=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,12))
  nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)
 def forward(self,images,state,grid,motion96,point_motion,residual):
  mv=motion96.clone();mv[:,:3]/=.01
  dense=self.base(torch.cat([images,mv],1),state,True)
  raw=F.grid_sample(dense,grid.unsqueeze(2),align_corners=False).squeeze(-1).transpose(1,2)
  p=point_motion.clone();p[:,:,:3]/=.01
  raw=raw+self.local(torch.cat([raw,p],-1))
  v=raw.reshape(raw.shape[0],raw.shape[1],3,4).permute(0,2,1,3)
  mean=v[...,:3];sigma=.2+1.8*torch.sigmoid(v[...,3])
  if residual:
   h=torch.tensor([1.,2.,4.],device=mean.device)[None,:,None,None]
   mean=mean+h*p[:,None,:,:3]*p[:,None,:,3:4]
  return mean,sigma
