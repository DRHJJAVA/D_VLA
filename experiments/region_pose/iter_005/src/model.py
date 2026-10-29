import torch
from torch import nn
from pointnet2_grasp import PointNet2Grasp
class RegionPoseNet(nn.Module):
 def __init__(self):
  super().__init__();self.backbone=PointNet2Grasp();self.backbone.head=nn.Identity()
  self.project=nn.Sequential(nn.Linear(128,256),nn.ReLU())
  self.base=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1))
  self.pose=nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
  self.film=nn.Linear(64,512);nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
  self.conditional=nn.Sequential(nn.Linear(259,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,xyz,dino,graph,pose,hand_R,hand_t):
  # Preregistered geometry ablation: no image descriptor enters either head.
  dino=torch.zeros_like(dino)
  feat=self.project(self.backbone(xyz/.1,dino,graph))
  base=self.base(feat).squeeze(-1)
  z=self.pose(pose);gamma,beta=self.film(z).chunk(2)
  rel=(xyz-hand_t)@hand_R/.1
  cond=self.conditional(torch.cat([feat*(1+gamma)+beta,rel],-1)).squeeze(-1)
  return base,cond
