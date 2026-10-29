from pathlib import Path
import json,sys,hashlib,os
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,h5py,torch,torch.nn.functional as F
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA');O=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src/v3'),str(R/'src')]
from stage2_extract_dino import model,preproc
from stage2_probe_motion_fingerpads import cloud
from stage2_robot_only_depth import RobotDepth
from build_point_graphs import fps,neighbors,interpolate
def buildgraph(x):
 i1=fps(x,min(512,len(x)));x1=x[i1];i2=fps(x1,min(128,len(x1)));x2=x1[i2];i3=fps(x2,min(32,len(x2)));x3=x2[i3]
 a,b=interpolate(x3,x2);c,d=interpolate(x2,x1);e,f=interpolate(x1,x)
 return dict(idx1=i1,idx2=i2,idx3=i3,nb1=neighbors(x,x1,.02,k=min(32,len(x))),nb2=neighbors(x1,x2,.04,k=min(32,len(x1))),nb3=neighbors(x2,x3,.08,k=min(32,len(x2))),fp3_idx=a,fp3_weight=b,fp2_idx=c,fp2_weight=d,fp1_idx=e,fp1_weight=f)
def main():
 torch.set_num_threads(4)
 cfg=json.loads((O/'config.json').read_text());fusion=Path(cfg['fusion_dir']);summs=json.loads((fusion/'summary.json').read_text())
 oldproto=R/'audits/v3/finger_contact_repair_pilot_v1/protocol.json'
 refs={r['id']:r for r in json.loads(oldproto.read_text())['episodes']}
 net=None;rows=[];skips=[]
 for s in summs:
  if not s['held']:continue
  ident=s['id'];split='train' if ident in cfg['train_episodes'] else 'val'
  assert ident in cfg[split+'_episodes']
  name=ident+'_attempt'+str(s['attempt']);zf=np.load(fusion/(name+'.npz'));rr=json.loads((fusion/(name+'.json')).read_text())['registration_records']
  pre=[x for x in rr if x.get('valid') and x['phase']=='pre']
  motion=np.load(R/'audits/v3/motion_finger_contact_pilot_v1'/(ident+'.npz'))
  with np.load(refs[ident]['component']) as comp:pix0=comp['pixel'].astype(np.int64)
  with h5py.File(refs[ident]['rgbd'],'r') as f:
   rob=RobotDepth(f['agentview/rgb'].shape[1]);d0=f['agentview/depth_m'][0];g=f['agentview'];dep0=d0[pix0[:,1],pix0[:,0]]
   for r in pre:
    t=r['frame'];key=name+'_query'+str(t);dest=O/'samples'/(key+'.npz');graph=O/'graphs'/(key+'.npz')
    if dest.exists() or graph.exists():raise RuntimeError('existing sample: inspect instead of overwrite')
    depth=g['depth_m'][t];K=g['K'][t];Tc=g['camera_to_world'][t];rgb=g['rgb'][t];H=motion['hand_to_world'][t];q=motion['measured_finger_q'][t]
    rd=rob.render(q,np.linalg.inv(H)@Tc,K);pu,pv=pix0[:,0],pix0[:,1]
    # Input object selector uses ONLY initial object foreground + query RGBD/robot pose.
    valid=(abs(depth[pv,pu]-dep0)<.005)&(depth[pv,pu]<rd[pv,pu]-.002)&np.isfinite(depth[pv,pu])&(depth[pv,pu]>0)
    coverage=float(valid.mean());pixel=pix0[valid]
    if len(pixel)<cfg['min_points']:skips.append(dict(id=ident,frame=t,reason='causal static-object foreground insufficient',coverage=coverage));continue
    world=cloud(depth,K,Tc);points=world[pixel[:,1]*depth.shape[1]+pixel[:,0]]
    # Sampling is independent of labels. Keep exact current-frame observations.
    if len(points)>cfg['max_points']:
     sel=fps(points,cfg['max_points']);points=points[sel];pixel=pixel[sel]
    center=points.mean(0);xyz=(points-center)@Tc[:3,:3];Rh=Tc[:3,:3].T@H[:3,:3];th=(H[:3,3]-center)@Tc[:3,:3]
    gap=q[0]-q[1]-.001;pose=np.r_[th/.1,Rh[:,0],Rh[:,1],gap/.1]
    T=np.array(r['source_world_to_anchor_hand']);mapped=points@T[:3,:3].T+T[:3,3]
    lo,hi=zf['volume_min_hand_m'],zf['volume_max_hand_m']
    surf_dist=cKDTree(zf['fused_surface_hand_m']).query(mapped)[0]
    reg_dist=cKDTree(zf['fused_region_hand_m']).query(mapped)[0]
    positive=np.all(mapped>=lo-.001,1)&np.all(mapped<=hi+.001,1)&(reg_dist<=.003)
    negative=(np.any(mapped<lo-.004,1)|np.any(mapped>hi+.004,1))&(surf_dist<=.003)
    known=positive|negative
    if positive.sum()<3 or negative.sum()<16:
     skips.append(dict(id=ident,frame=t,reason='insufficient reliable region supervision',pos=int(positive.sum()),neg=int(negative.sum())));continue
    target=positive.astype(np.float32)
    confidence=np.exp(-surf_dist/.004)*min(r['source_overlap_4mm'],r['reference_overlap_4mm']);confidence=np.where(known,confidence,0).astype(np.float32)
    # Strict masked query RGB, no future-derived teacher image/mask.
    mask=np.zeros(depth.shape,bool);mask[pixel[:,1],pixel[:,0]]=True
    image=np.full_like(rgb,127);image[mask]=rgb[mask]
    x1,y1=pixel.min(0);x2,y2=pixel.max(0)+1
    if net is None:net=model()
    with torch.inference_mode():patch=net.forward_features(preproc(image[None,y1:y2,x1:x2]))[:,1:,:].reshape(1,32,32,384).permute(0,3,1,2)
    gx=(pixel[:,0]-x1+.5)/(x2-x1)*2-1;gy=(pixel[:,1]-y1+.5)/(y2-y1)*2-1
    grid=torch.as_tensor(np.stack([gx,gy],1)[None,None],dtype=patch.dtype,device=patch.device)
    with torch.inference_mode():feat=F.grid_sample(patch,grid,align_corners=False).squeeze(0).squeeze(1).T.cpu().numpy()
    np.savez_compressed(dest,xyz=xyz.astype(np.float32),xyz_world=points.astype(np.float32),dino_feat=feat.astype(np.float16),pixel=pixel.astype(np.int16),rgb=rgb[pixel[:,1],pixel[:,0]],pose=pose.astype(np.float32),hand_R=Rh.astype(np.float32),hand_t=th.astype(np.float32),target=target,known=known,confidence=confidence,query_frame=t,event=np.asarray(name),episode=np.asarray(ident),split=np.asarray(split),query_rgbd=np.asarray(refs[ident]['rgbd']),coordinate_basis_camera_to_world=Tc,center_world=center,target_world_to_anchor_hand=T)
    gr=buildgraph(xyz);np.savez_compressed(graph,**gr)
    for k,v in gr.items():
     if 'weight' in k:assert np.isfinite(v).all() and np.allclose(v.sum(1),1)
    assert np.isfinite(feat).all()
    row=dict(key=key,event=name,episode=ident,split=split,query_frame=t,points=len(points),positive=int(positive.sum()),negative=int(negative.sum()),unknown=int((~known).sum()),foreground_coverage=coverage,target_source=str(fusion/(name+'.npz')))
    rows.append(row);print(json.dumps(row),flush=True)
   rob.close()
 manifest=dict(status='complete',rows=rows,skipped=skips,source_lock_sha256=cfg['source_lock_sha256'],input_mask_independent_of_target=True,source_only_trainval=True)
 train={r['episode'] for r in rows if r['split']=='train'};val={r['episode'] for r in rows if r['split']=='val'}
 (O/'audits/preparation_diagnostics.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps(dict(skipped=skips)),flush=True)
 assert train and val and not train&val
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps(dict(event='data_complete',train_episodes=len(train),val_episodes=len(val),samples=len(rows),skipped=len(skips))),flush=True)
if __name__=='__main__':main()
