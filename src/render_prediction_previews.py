#!/usr/bin/env python3
"""Render compact held-out RGB-D and point-score previews on the server."""
import json
import math
from pathlib import Path
import sys
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

ROOT=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(ROOT/'src'))
from pointnet2_grasp import PointNet2Grasp
from train_pointnet2_grasp import tensors
from evaluate_contact_coverage import unique_contacts

CAMERAS=('agentview','robot0_eye_in_hand')
CASES=(('low','demo_15',34),('middle','demo_47',0),('high','demo_35',34))
OUTPUT=ROOT/'outputs/alphabet_soup_pointnet2_16k_provisional/previews'

def project_visible(world,group,frame,tolerance):
    T=group['camera_to_world'][frame]
    local=(world-T[:3,3])@T[:3,:3]
    z=local[:,2]
    K=group['K'][frame]
    safe=np.where(z>0,z,1)
    u=np.rint(K[0,0]*local[:,0]/safe+K[0,2]).astype(int)
    v=np.rint(K[1,1]*local[:,1]/safe+K[1,2]).astype(int)
    depth=group['depth_m'][frame]
    inside=(z>0)&(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
    valid=np.zeros(len(world),dtype=bool)
    idx=np.flatnonzero(inside)
    if len(idx):
        measured=depth[v[idx],u[idx]]
        valid[idx]=np.isfinite(measured)&(np.abs(measured-z[idx])<=tolerance)
    return u[valid],v[valid]

def main():
    torch.set_num_threads(4)
    ck=torch.load(ROOT/'outputs/alphabet_soup_pointnet2_16k_provisional/best.pt',map_location='cpu',weights_only=False)
    model=PointNet2Grasp().eval()
    model.load_state_dict(ck['model'])
    metrics=json.loads((ROOT/'outputs/alphabet_soup_pointnet2_16k_provisional/contact_coverage_test_0.010m.json').read_text())
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for label,episode,frame in CASES:
        name=f'{episode}_frame_{frame}'
        with np.load(ROOT/'data/samples/single_frame_16k/test'/f'{name}.npz') as f:
            item={k:f[k].copy() for k in ('xyz','dino_feat','target','valid_mask')}
            camera_id=f['camera_id'].copy()
            uv=f['pixel_uv'].copy()
            object_ids=f['object_ids'].copy()
            sigma=float(f['sigma_m'])
        with np.load(ROOT/'data/samples/single_frame_16k/graphs/test'/f'{name}.npz') as f:
            graph={k:f[k].copy() for k in f.files}
        xyz,feature,target,valid,graph=tensors(item,graph,'cpu')
        with torch.inference_mode():
            score=torch.sigmoid(model(xyz,feature,graph)).numpy()
        object_index=np.flatnonzero(object_ids==1)
        count=max(1,math.ceil(.2*len(object_index)))
        top_index=object_index[np.argsort(-score[object_index],kind='stable')[:count]]
        event_rows=json.loads((ROOT/'data/events'/f'alphabet_soup_{episode}_grasp_candidates.json').read_text())['events']
        event=max((e for e in event_rows if e['passes_numeric_filter']),key=lambda e:e['duration_frames'])
        contact_object=unique_contacts(event)
        probe=json.loads((ROOT/'data/events'/f'alphabet_soup_{episode}_contact_probe.json').read_text())
        pose=probe['records'][frame]['target_body_poses']['alphabet_soup_1_main']
        world=contact_object@np.asarray(pose['rotation_world']).T+np.asarray(pose['position_world'])
        result=next(r for r in metrics['per_sample'] if r['episode']==episode and r['frame']==frame)
        fig,axes=plt.subplots(2,4,figsize=(20,9),layout='constrained')
        for ci,camera in enumerate(CAMERAS):
            with h5py.File(ROOT/'data/rendered'/f'alphabet_soup_{episode}_full_rgbd.h5') as video:
                group=video[camera]
                rgb=group['rgb'][frame]
                depth=group['depth_m'][frame]
                contact_u,contact_v=project_visible(world,group,frame,max(.01,2*sigma))
            camera_points=np.flatnonzero(camera_id==ci)
            high=camera_points[score[camera_points]>=.1]
            local_top=top_index[camera_id[top_index]==ci]
            local_obj=object_index[camera_id[object_index]==ci]
            axes[ci,0].imshow(rgb)
            axes[ci,0].set_title('RGB input')
            axes[ci,0].set_ylabel('fixed camera' if ci==0 else 'wrist camera',fontsize=12)
            depth_limits=(.6,1.5) if ci==0 else (.03,.5)
            axes[ci,1].imshow(depth,cmap='cividis',vmin=depth_limits[0],vmax=depth_limits[1])
            axes[ci,1].set_title(f'Depth (m): {depth_limits[0]}-{depth_limits[1]}')
            axes[ci,2].imshow(rgb)
            axes[ci,2].scatter(uv[high,0],uv[high,1],c=score[high],s=3.5,
                               cmap='magma',vmin=0,vmax=.55,alpha=.75,linewidths=0)
            axes[ci,2].set_title('Model scores > 0.1, whole scene')
            axes[ci,3].imshow(rgb)
            axes[ci,3].scatter(uv[local_obj,0],uv[local_obj,1],s=3,c='#68cde0',alpha=.5,linewidths=0)
            axes[ci,3].scatter(uv[local_top,0],uv[local_top,1],s=11,c='#ff304f',alpha=.9,linewidths=0)
            axes[ci,3].scatter(contact_u,contact_v,s=85,c='#19ed6d',marker='x',linewidths=2.2)
            axes[ci,3].set_title('Top 20% target points + contacts')
            for ax in axes[ci]:
                ax.set_xlim(0,448);ax.set_ylim(448,0);ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f'{label.upper()} example | {episode}, frame {frame} | top-20% contact coverage {result["coverage_at_20"]:.0%}',fontsize=17)
        path=OUTPUT/f'{label}_{name}.jpg'
        fig.savefig(path,dpi=135,pil_kwargs={'quality':88},bbox_inches='tight',facecolor='white')
        plt.close(fig)
        print(json.dumps({'case':label,'episode':episode,'frame':frame,'coverage20':result['coverage_at_20'],
                          'preview':str(path),'bytes':path.stat().st_size}),flush=True)

if __name__=='__main__':main()
