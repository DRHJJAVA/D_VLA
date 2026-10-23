#!/usr/bin/env python3
"""Assess whether the 20-sample fit learned to rank observed contact-region points."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
def ranking_metrics(binary,score):
    order=np.argsort(-score,kind='stable')
    labels=binary[order].astype(bool)
    npos=int(labels.sum());nneg=len(labels)-npos
    precision=np.cumsum(labels)/(np.arange(len(labels))+1)
    ap=float(precision[labels].mean())
    rank=np.empty(len(score),dtype=np.int64)
    rank[np.argsort(score,kind='stable')]=np.arange(1,len(score)+1)
    auc=float((rank[binary].sum()-npos*(npos+1)/2)/(npos*nneg))
    return ap,auc
from pointnet2_grasp import PointNet2Grasp


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--graphs',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PointNet2Grasp().to(device)
    checkpoint=torch.load(a.checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model']);model.eval()
    rows=[]
    with torch.inference_mode():
        for path in sorted(a.samples.glob('*.npz')):
            graph_path=a.graphs/path.name
            if not graph_path.exists():continue
            with np.load(path) as source:
                xyz=source['xyz'].astype(np.float32);feature=source['dino_feat'].astype(np.float32)
                target=source['target'].astype(np.float32);valid=source['valid_mask'].astype(bool)
            with np.load(graph_path) as source:
                graph={key:torch.as_tensor(source[key],device=device,
                    dtype=torch.float32 if key.endswith('weight') else torch.long)
                    for key in source.files}
            logits=model(torch.as_tensor(xyz,device=device),torch.as_tensor(feature,device=device),graph)
            score=torch.sigmoid(logits).cpu().numpy()
            high=(target>0.5)&valid
            if high.sum():
                ap,auc=ranking_metrics(high[valid],score[valid])
                pos_score=float(score[high].mean())
                prevalence=float(high[valid].mean())
            else:ap=auc=pos_score=prevalence=None
            background=(target<0.05)&valid
            rows.append({'sample':path.name,'positive_count':int(high.sum()),'ap':ap,'auc':auc,
                         'prevalence':prevalence,'mean_positive_score':pos_score,
                         'mean_background_score':float(score[background].mean()),
                         'max_score':float(score.max())})
    valid_rows=[row for row in rows if row['ap'] is not None]
    summary={'checkpoint_epoch':checkpoint['epoch'],'samples':len(rows),
             'samples_with_positive_points':len(valid_rows),
             'mean_ap':float(np.mean([r['ap'] for r in valid_rows])),
             'mean_random_ap':float(np.mean([r['prevalence'] for r in valid_rows])),
             'mean_auc':float(np.mean([r['auc'] for r in valid_rows])),
             'rows':rows,'scope':'training samples only; unreviewed labels'}
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
