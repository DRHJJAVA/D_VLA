#!/usr/bin/env python3
"""Pure PyTorch PointNet++ segmentation backbone for 4,096-point grasp-region scores."""
import torch
from torch import nn


class SharedMLP(nn.Module):
    def __init__(self,sizes):
        super().__init__()
        layers=[]
        for left,right in zip(sizes[:-1],sizes[1:]):
            layers.extend((nn.Linear(left,right),nn.ReLU()))
        self.net=nn.Sequential(*layers)
    def forward(self,x):return self.net(x)


class SetAbstraction(nn.Module):
    def __init__(self,input_dim,channels):
        super().__init__()
        self.mlp=SharedMLP([input_dim+3,*channels])
    def forward(self,xyz,features,center_idx,neighbors):
        centers=xyz[center_idx]
        local_xyz=xyz[neighbors]-centers[:,None,:]
        local_features=features[neighbors]
        values=self.mlp(torch.cat((local_xyz,local_features),dim=-1))
        return centers,values.max(dim=1).values


class FeaturePropagation(nn.Module):
    def __init__(self,input_dim,channels):
        super().__init__()
        self.mlp=SharedMLP([input_dim,*channels])
    def forward(self,fine_features,coarse_features,neighbor_idx,weight):
        interpolated=(coarse_features[neighbor_idx]*weight[...,None]).sum(dim=1)
        return self.mlp(torch.cat((fine_features,interpolated),dim=-1))


class PointNet2Grasp(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress=nn.Sequential(nn.Linear(384,64),nn.ReLU())
        self.sa1=SetAbstraction(64,[64,64,128])
        self.sa2=SetAbstraction(128,[128,128,256])
        self.sa3=SetAbstraction(256,[256,256,512])
        self.fp3=FeaturePropagation(256+512,[256,256])
        self.fp2=FeaturePropagation(128+256,[256,128])
        self.fp1=FeaturePropagation(64+128,[128,128])
        self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,xyz,dino,graph):
        centered=xyz-xyz.mean(dim=0,keepdim=True)
        f0=self.compress(dino)
        x1,f1=self.sa1(centered,f0,graph['idx1'],graph['nb1'])
        x2,f2=self.sa2(x1,f1,graph['idx2'],graph['nb2'])
        x3,f3=self.sa3(x2,f2,graph['idx3'],graph['nb3'])
        up2=self.fp3(f2,f3,graph['fp3_idx'],graph['fp3_weight'])
        up1=self.fp2(f1,up2,graph['fp2_idx'],graph['fp2_weight'])
        up0=self.fp1(f0,up1,graph['fp1_idx'],graph['fp1_weight'])
        return self.head(up0).squeeze(-1)
