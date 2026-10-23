#!/usr/bin/env python3
"""Prepare single-frame point clouds, frozen DINOv2 features and contact soft labels."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import timm
from restore_audit import OffScreenRenderEnv,libero_utils
from asset_paths import localize_assets
import robosuite.macros as macros

CAMERAS=('agentview','robot0_eye_in_hand')
OBJECT_BODY='alphabet_soup_1_main'
OBJECT_GEOM_PREFIX='alphabet_soup_1'


def make_split(labels_path,split_path,seed):
    episodes=sorted([x['episode'] for x in json.loads(labels_path.read_text())['labels']],
                    key=lambda x:int(x.split('_')[-1]))
    rng=random.Random(seed);rng.shuffle(episodes)
    n=len(episodes);n_train=round(n*0.7);n_val=round(n*0.15)
    split={'train':episodes[:n_train],'val':episodes[n_train:n_train+n_val],
           'test':episodes[n_train+n_val:],'seed':seed,'unit':'episode',
           'label_library_policy':'separate within each split'}
    if split_path.exists():
        old=json.loads(split_path.read_text())
        if any(old[k]!=split[k] for k in ('train','val','test')):
            raise ValueError('Existing split differs from deterministic split')
    else:
        split_path.parent.mkdir(parents=True,exist_ok=True)
        split_path.write_text(json.dumps(split,indent=2)+'\n')
    return split


def contact_library(events_dir,episodes):
    points=[]
    for episode in episodes:
        report=json.loads((events_dir/f'alphabet_soup_{episode}_grasp_candidates.json').read_text())
        for event in report['events']:
            if event['passes_numeric_filter']:
                points.extend(c['position_object'] for c in event['left_contacts']+event['right_contacts'])
    if not points:raise ValueError('No successful contact points in split')
    return np.asarray(points,dtype=np.float32)


def dinov2(device):
    model=timm.create_model('vit_small_patch14_dinov2',pretrained=False,dynamic_img_size=True)
    weights=torch.load('/home/drh/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth',
                       map_location='cpu',weights_only=False)
    result=model.load_state_dict(weights,strict=False)
    if result.missing_keys or result.unexpected_keys!=['mask_token']:
        raise RuntimeError(f'Unexpected DINOv2 weights: {result}')
    model=model.eval().to(device)
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model


def patch_features(model,rgb,device):
    image=torch.from_numpy(np.ascontiguousarray(rgb)).to(device).permute(2,0,1).float().unsqueeze(0)/255
    mean=torch.tensor([0.485,0.456,0.406],device=device)[None,:,None,None]
    std=torch.tensor([0.229,0.224,0.225],device=device)[None,:,None,None]
    with torch.inference_mode():
        tokens=model.forward_features((image-mean)/std)
    return tokens[:,1:,:].transpose(1,2).reshape(1,384,32,32)


def sample_dino(feature_map,u,v):
    # Bilinear interpolation from 14-pixel patch centers to the selected pixels.
    gx=2*((u.astype(np.float32)+0.5)/14-0.5)/31-1
    gy=2*((v.astype(np.float32)+0.5)/14-0.5)/31-1
    grid=torch.from_numpy(np.stack((gx,gy),axis=-1)).to(feature_map.device)[None,None]
    with torch.inference_mode():
        values=F.grid_sample(feature_map,grid,mode='bilinear',align_corners=True,
                             padding_mode='border')[0,:,0,:].T
    return values.cpu().numpy().astype(np.float16)


def unproject(depth,K,T):
    h,w=depth.shape
    v,u=np.indices((h,w))
    x=(u-K[0,2])*depth/K[0,0]
    y=(v-K[1,2])*depth/K[1,1]
    camera=np.stack((x,y,depth),axis=-1)
    return (camera@T[:3,:3].T+T[:3,3]).astype(np.float32)


def prepare_frame(env,video,episode,frame,model,library,probe,output,seed,points):
    env.sim.set_state_from_flattened(np.asarray(episode['states'][frame],dtype=np.float64))
    env.sim.forward();env._update_observables(force=True)
    obs=env.env._get_observations()
    target_ids={i for i in range(env.sim.model.ngeom)
                if (env.sim.model.geom_id2name(i) or '').startswith(OBJECT_GEOM_PREFIX)}
    if not target_ids:raise RuntimeError('Target geometry missing')
    rng=np.random.default_rng(seed)
    all_xyz=[];all_camera=[];all_u=[];all_v=[];all_ids=[];all_names=[]
    feature_maps=[];rgb_match=[]
    for c,camera in enumerate(CAMERAS):
        rgb=video[camera]['rgb'][frame]
        depth=video[camera]['depth_m'][frame]
        K=video[camera]['K'][frame]
        T=video[camera]['camera_to_world'][frame]
        rendered=np.asarray(obs[camera+'_image'])
        seg=np.asarray(obs[camera+'_segmentation_element'])[...,0]
        if macros.IMAGE_CONVENTION=='opengl':
            rendered=np.flipud(rendered);seg=np.flipud(seg)
        match=float(np.mean(rendered==rgb));rgb_match.append(match)
        if match<0.999:raise RuntimeError(f'RGB and offline segmentation mismatch: {camera} {frame} {match}')
        xyz=unproject(depth,K,T)
        valid=np.isfinite(depth)&(depth>0)&(depth<5)
        valid&=(xyz[...,0]>-0.75)&(xyz[...,0]<0.65)
        valid&=(xyz[...,1]>-0.75)&(xyz[...,1]<0.50)
        valid&=(xyz[...,2]>-0.05)&(xyz[...,2]<0.85)
        v,u=np.where(valid)
        all_xyz.append(xyz[v,u]);all_camera.append(np.full(len(u),c,dtype=np.uint8))
        all_u.append(u.astype(np.int16));all_v.append(v.astype(np.int16));all_ids.append(seg[v,u].astype(np.int16))
        feature_maps.append(patch_features(model,rgb,next(model.parameters()).device))
    xyz=np.concatenate(all_xyz);camera=np.concatenate(all_camera)
    u=np.concatenate(all_u);v=np.concatenate(all_v);seg_id=np.concatenate(all_ids)
    order=rng.permutation(len(xyz))
    voxel=np.floor(xyz[order]/0.003).astype(np.int32)
    _,first=np.unique(voxel,axis=0,return_index=True)
    available=order[first]
    if len(available)<points:raise RuntimeError(f'Only {len(available)} unique workspace points')
    chosen=rng.choice(available,size=points,replace=False)
    xyz=xyz[chosen];camera=camera[chosen];u=u[chosen];v=v[chosen];seg_id=seg_id[chosen]
    dino=np.empty((points,384),dtype=np.float16)
    for c in range(2):
        at=np.where(camera==c)[0]
        dino[at]=sample_dino(feature_maps[c],u[at],v[at])
    target_mask=np.isin(seg_id,np.fromiter(target_ids,dtype=np.int32))
    geom_names=[env.sim.model.geom_id2name(int(i)) or '' for i in seg_id]
    bg_mask=np.array([('table' in name or 'floor' in name or 'ground' in name) for name in geom_names])
    valid_mask=target_mask|bg_mask
    object_ids=np.where(target_mask,1,np.where(bg_mask,0,-1)).astype(np.int8)
    pose=probe['records'][frame]['target_body_poses'][OBJECT_BODY]
    position=np.asarray(pose['position_world'],dtype=np.float32)
    rotation=np.asarray(pose['rotation_world'],dtype=np.float32)
    object_xyz=(xyz-position)@rotation
    visible_obj=xyz[target_mask]
    if len(visible_obj)<10:raise RuntimeError(f'Target only {len(visible_obj)} sampled points')
    diagonal=float(np.linalg.norm(np.ptp(visible_obj,axis=0)))
    sigma=max(0.002,0.02*diagonal)
    target=np.zeros(points,dtype=np.float32)
    obj_idx=np.where(target_mask)[0]
    delta=object_xyz[obj_idx,None,:]-library[None,:,:]
    dist2=np.min(np.sum(delta*delta,axis=-1),axis=1)
    target[obj_idx]=np.exp(-dist2/(2*sigma*sigma))
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,xyz=xyz,dino_feat=dino,target=target,
                        valid_mask=valid_mask,object_ids=object_ids,
                        camera_id=camera,pixel_uv=np.stack((u,v),axis=1),
                        episode_id=episode.name.split('/')[-1],frame_id=frame,
                        sigma_m=sigma)
    return {'sample':str(output),'frame':frame,'points':points,'target_points':int(target_mask.sum()),
            'background_points':int(bg_mask.sum()),'supervised_points':int(valid_mask.sum()),
            'high_score_points':int((target>0.5).sum()),'max_target':float(target.max()),
            'sigma_m':sigma,'rgb_match':rgb_match}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--rendered-dir',type=Path,required=True)
    p.add_argument('--events-dir',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--libero-root',type=Path,required=True)
    p.add_argument('--partition',choices=['train','val','test'],default='train')
    p.add_argument('--limit-episodes',type=int,default=5)
    p.add_argument('--points',type=int,default=4096)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    os.environ.setdefault('MUJOCO_GL','egl')
    split=make_split(a.labels,a.split,a.seed)
    selected=split[a.partition] if a.limit_episodes<=0 else split[a.partition][:a.limit_episodes]
    library=contact_library(a.events_dir,split[a.partition])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=dinov2(device)
    report={'partition':a.partition,'episodes':selected,'library_contact_count':len(library),'samples':[]}
    with h5py.File(a.raw) as raw:
        data=raw['data']
        bddl=a.libero_root/'libero/libero/bddl_files/libero_object'/Path(data.attrs['bddl_file_name']).name
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_names=list(CAMERAS),
            camera_heights=448,camera_widths=448,camera_depths=True,camera_segmentations='element')
        try:
            for name in selected:
                episode=data[name]
                xml=episode.attrs['model_file']
                if isinstance(xml,bytes):xml=xml.decode()
                xml=libero_utils.postprocess_model_xml(xml,{})
                xml,_=localize_assets(xml,a.libero_root)
                env.reset();env.reset_from_xml_string(xml);env.sim.reset()
                probe=json.loads((a.events_dir/f'alphabet_soup_{name}_contact_probe.json').read_text())
                labels=json.loads((a.events_dir/f'alphabet_soup_{name}_grasp_candidates.json').read_text())
                event=max((x for x in labels['events'] if x['passes_numeric_filter']),
                          key=lambda x:x['duration_frames'])
                contact=event['contact_frame']
                frames=sorted({0,*[max(0,contact-offset) for offset in (30,20,10)]})
                with h5py.File(a.rendered_dir/f'alphabet_soup_{name}_full_rgbd.h5') as video:
                    for frame in frames:
                        output=a.output_dir/a.partition/f'{name}_frame_{frame}.npz'
                        digest=int(hashlib.sha256(f'{name}:{frame}'.encode()).hexdigest()[:8],16)
                        if output.exists():
                            with np.load(output) as cached:
                                if cached['xyz'].shape!=(a.points,3) or cached['dino_feat'].shape!=(a.points,384):
                                    raise ValueError(f'Invalid cached point sample: {output}')
                            row={'sample':str(output),'frame':frame,'reused':True}
                        else:
                            row=prepare_frame(env,video,episode,frame,model,library,probe,
                                              output,a.seed+digest,a.points)
                        report['samples'].append(row)
                        print(json.dumps(row),flush=True)
                (a.output_dir/f'{a.partition}_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        finally:env.close()
    print(json.dumps({'completed':len(report['samples']),'partition':a.partition}),flush=True)

if __name__=='__main__':main()
