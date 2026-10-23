#!/usr/bin/env python3
"""Episode-held-out evaluation for provisional single-frame grasp-region training."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from pointnet2_grasp import PointNet2Grasp
from train_pointnet2_grasp import tensors,balanced_loss
from evaluate_pointnet2_fit import ranking_metrics


def load_partition(root,partition):
    items=[]
    for path in sorted((root/partition).glob('*.npz')):
        graph_path=root/'graphs'/partition/path.name
        if not graph_path.exists():raise FileNotFoundError(graph_path)
        with np.load(path) as s:
            item={key:s[key].copy() for key in ('xyz','dino_feat','target','valid_mask')}
        with np.load(graph_path) as s:
            graph={key:s[key].copy() for key in s.files}
        items.append((path.name,item,graph))
    if not items:raise RuntimeError(f'No {partition} items')
    return items


def evaluate(model,items,device):
    model.eval();losses=[];aps=[];random_aps=[]
    with torch.inference_mode():
        for _,item,graph in items:
            xyz,feature,target,valid,graph=tensors(item,graph,device)
            logits=model(xyz,feature,graph)
            losses.append(float(balanced_loss(logits,target,valid).cpu()))
            target_np=item['target'];valid_np=item['valid_mask'].astype(bool)
            high=(target_np>0.5)&valid_np
            if high.sum() and (~high&valid_np).sum():
                ap,_=ranking_metrics(high[valid_np],torch.sigmoid(logits)[valid].cpu().numpy())
                aps.append(ap);random_aps.append(float(high[valid_np].mean()))
    return {'balanced_bce':float(np.mean(losses)),
            'mean_ap':float(np.mean(aps)) if aps else None,
            'random_ap':float(np.mean(random_aps)) if random_aps else None,
            'samples':len(items),'samples_with_high_labels':len(aps)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    train=load_partition(a.samples,'train')
    val=load_partition(a.samples,'val')
    test=load_partition(a.samples,'test')
    names={part:{name.split('_frame_')[0] for name,_,_ in items}
           for part,items in (('train',train),('val',val),('test',test))}
    if names['train']&names['val'] or names['train']&names['test'] or names['val']&names['test']:
        raise ValueError('Episode leakage between splits')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PointNet2Grasp().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'device':str(device),'train':len(train),'val':len(val),'test':len(test),
                      'episodes':{k:len(v) for k,v in names.items()},
                      'input':'single-frame xyz + DINOv2, no segmentation'}),flush=True)
    history=[];best=float('inf')
    for epoch in range(1,a.epochs+1):
        model.train();random.shuffle(train);losses=[]
        for _,item,graph in train:
            xyz,feature,target,valid,graph=tensors(item,graph,device)
            optimizer.zero_grad(set_to_none=True)
            loss=balanced_loss(model(xyz,feature,graph),target,valid)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();losses.append(float(loss.detach().cpu()))
        val_result=evaluate(model,val,device)
        row={'epoch':epoch,'train_balanced_bce':float(np.mean(losses)),'validation':val_result}
        history.append(row);print(json.dumps(row),flush=True)
        if val_result['balanced_bce']<best:
            best=val_result['balanced_bce']
            torch.save({'model':model.state_dict(),'epoch':epoch,
                        'val_balanced_bce':best,'input':'single-frame pointwise RGB-D'},
                       a.output_dir/'best.pt')
        (a.output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    checkpoint=torch.load(a.output_dir/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model'])
    result={'best_epoch':checkpoint['epoch'],'validation':evaluate(model,val,device),
            'test':evaluate(model,test,device),'train_count':len(train),
            'val_count':len(val),'test_count':len(test),
            'episode_splits':{k:sorted(v) for k,v in names.items()},
            'label_quality':'provisional_unreviewed',
            'scope':'known object/task; no unseen-asset claim'}
    (a.output_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
