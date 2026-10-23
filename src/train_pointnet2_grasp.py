#!/usr/bin/env python3
"""Small-sample fit check for single-frame PointNet++ grasp-region prediction."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pointnet2_grasp import PointNet2Grasp


def load_items(sample_dir,graph_dir,min_samples,timeout):
    start=time.monotonic()
    while len(list(graph_dir.glob('*.npz')))<min_samples:
        if time.monotonic()-start>timeout:raise TimeoutError('Point graphs not ready')
        time.sleep(10)
    items=[]
    for path in sorted(sample_dir.glob('*.npz')):
        graph_path=graph_dir/path.name
        if not graph_path.exists():continue
        with np.load(path) as source:
            required={'xyz','dino_feat','target','valid_mask','object_ids'}
            if not required.issubset(source.files):raise ValueError(f'Missing sample fields: {path}')
            item={k:source[k].copy() for k in required}
        with np.load(graph_path) as source:
            graph={k:source[k].copy() for k in source.files}
        if item['xyz'].shape!=(4096,3) or item['dino_feat'].shape!=(4096,384):
            raise ValueError(f'Wrong point sample size: {path}')
        if not (item['target']>0.05).any():raise ValueError(f'No positive soft labels: {path}')
        items.append((path.name,item,graph))
    return items[:min_samples]


def tensors(item,graph,device):
    xyz=torch.as_tensor(item['xyz'],device=device,dtype=torch.float32)
    feature=torch.as_tensor(item['dino_feat'],device=device,dtype=torch.float32)
    target=torch.as_tensor(item['target'],device=device,dtype=torch.float32)
    valid=torch.as_tensor(item['valid_mask'],device=device,dtype=torch.bool)
    graph={k:torch.as_tensor(value,device=device,dtype=torch.float32 if k.endswith('weight') else torch.long)
           for k,value in graph.items()}
    return xyz,feature,target,valid,graph


def balanced_loss(logits,target,valid):
    positive=valid&(target>0.05)
    negative=valid&~positive
    raw=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    if not positive.any():
        return raw[negative].mean()
    if not negative.any():
        return raw[positive].mean()
    return 0.5*(raw[positive].mean()+raw[negative].mean())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sample-dir',type=Path,required=True)
    p.add_argument('--graph-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--min-samples',type=int,default=20)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--wait-seconds',type=int,default=3600)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    items=load_items(a.sample_dir,a.graph_dir,a.min_samples,a.wait_seconds)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PointNet2Grasp().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'samples':len(items),'device':str(device),'model':'PointNet++ pure PyTorch',
                      'input':'single-frame xyz + frozen DINOv2 patch features',
                      'segmentation_model_input':False}),flush=True)
    best=float('inf');history=[]
    for epoch in range(1,a.epochs+1):
        random.shuffle(items);losses=[]
        model.train()
        for name,item,graph in items:
            xyz,feature,target,valid,graph=tensors(item,graph,device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(xyz,feature,graph)
            loss=balanced_loss(logits,target,valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss=float(np.mean(losses))
        record={'epoch':epoch,'train_balanced_bce':mean_loss}
        history.append(record)
        print(json.dumps(record),flush=True)
        if mean_loss<best:
            best=mean_loss
            torch.save({'model':model.state_dict(),'epoch':epoch,'train_loss':mean_loss,
                        'input':'single-frame 4096 XYZ points + DINOv2 384d features'},
                       a.output_dir/'best.pt')
        (a.output_dir/'metrics.json').write_text(json.dumps({'samples':[x[0] for x in items],
            'history':history,'best_loss':best,'label_quality':'provisional_unreviewed',
            'evaluation':'training fit only; no validation claim'},indent=2)+'\n')
    print(json.dumps({'completed':str(a.output_dir/'best.pt'),'best_loss':best}),flush=True)

if __name__=='__main__':main()
