"""Create visual-only interaction pseudo targets for a first-pose object student.
The complete RGB-D sequence determines the training cue. No simulator contact/pose/mesh is read."""
from pathlib import Path
import json
import h5py
import numpy as np
from scipy.spatial import cKDTree
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_point_graphs import fps,neighbors,interpolate
R=Path('/mnt/nvme1/drh/D_VLA')
M=R/'data/v3/manifests/original_sequences.jsonl'
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
E={x['id']:x for x in (json.loads(z) for z in M.read_text().splitlines() if z)}
OUT=R/'data/v3/samples/visual_interaction_v1'
def build(e):
 asset=e['asset_group'];ep=e['source_episode']
 rgbd=Path(e['rgbd_path'])
 proposal=R/'data/v3/proposals/motion_rgbd_rerank'/asset/(ep+'.json')
 x1,y1,x2,y2=json.loads(proposal.read_text())['selected']['box']
 cropfile=R/'data/v3/features/dinov2_object_crops'/asset/(ep+'.h5')
 with h5py.File(rgbd) as f:
  depth=f['agentview/depth_m'][:]
  rgb=f['agentview/rgb'][0]
  K=f['agentview/K'][:];T=f['agentview/camera_to_world'][:]
 with h5py.File(cropfile) as f:
  assert f.attrs['status']=='complete'
  patch=f['patch'][0].astype(np.float32)
  cropbox=json.loads(f.attrs['crop_box'])
 first=depth[0]
 ring=first[max(0,y1-8):min(first.shape[0],y2+8),max(0,x1-8):min(first.shape[1],x2+8)].copy()
 ry=y1-max(0,y1-8);rx=x1-max(0,x1-8)
 ring[ry:ry+y2-y1,rx:rx+x2-x1]=np.nan
 background=float(np.nanmedian(ring))
 d=first[y1:y2,x1:x2]
 mask=(d>.05)&(d<background-.015)
 yy,xx=np.nonzero(mask)
 if len(xx)<64:return dict(id=e['id'],reason='insufficient_visual_foreground',points=len(xx))
 px=x1+xx;py=y1+yy;z=d[yy,xx]
 cam=np.stack(((px-K[0,0,2])*z/K[0,0,0],(py-K[0,1,2])*z/K[0,1,1],z),1)
 xyz=cam@T[0,:3,:3].T+T[0,:3,3]
 cx1,cy1,cx2,cy2=cropbox
 gx=np.clip(((px-cx1)*32/max(1,cx2-cx1)).astype(int),0,31)
 gy=np.clip(((py-cy1)*32/max(1,cy2-cy1)).astype(int),0,31)
 dino=patch[gy,gx]
 removed=((depth[:,y1:y2,x1:x2]>d[None,:,:]+.025)&mask[None]).sum((1,2))/mask.sum()
 # Each sustained crossing is an event; no assumption about a fixed grasp phase.
 onset=[i for i in range(1,len(removed)-2) if removed[i-1]<=.2 and np.all(removed[i:i+3]>.2)]
 if not onset:return dict(id=e['id'],reason='no_visual_removal_event',points=len(xyz))
 xlo=max(0,x1-30);ylo=max(0,y1-30);xhi=min(first.shape[1],x2+30);yhi=min(first.shape[0],y2+30)
 scores=[]
 for event in onset:
  nearest=np.full(len(xyz),np.inf)
  for frame in range(max(0,event-8),event):
   now=depth[frame,ylo:yhi:2,xlo:xhi:2]
   old=first[ylo:yhi:2,xlo:xhi:2]
   sy,sx=np.nonzero((now<old-.02)&(now>.05))
   if not len(sx):continue
   pu=xlo+2*sx;pv=ylo+2*sy;zz=now[sy,sx];kt=K[frame];tt=T[frame]
   cam2=np.stack(((pu-kt[0,2])*zz/kt[0,0],(pv-kt[1,2])*zz/kt[1,1],zz),1)
   moving=cam2@tt[:3,:3].T+tt[:3,3]
   nearest=np.minimum(nearest,cKDTree(moving).query(xyz)[0])
  if np.isfinite(nearest).any():scores.append(nearest)
 if not scores:return dict(id=e['id'],reason='no_pre_event_moving_surface',points=len(xyz))
 distance=np.min(np.stack(scores),axis=0)
 # Continuous relation support; far surfaces get small, not categorical negative grasp labels.
 target=np.exp(-np.minimum(distance,.2)/.02).astype(np.float32)
 sample=OUT/e['split']/(e['id']+'.npz')
 graph=OUT/'graphs'/e['split']/(e['id']+'.npz')
 sample.parent.mkdir(parents=True,exist_ok=True);graph.parent.mkdir(parents=True,exist_ok=True)
 if sample.exists() or graph.exists():raise FileExistsError(f'Partial or completed sample exists: {sample}')
 np.savez_compressed(sample,xyz=xyz.astype(np.float32),dino_feat=dino.astype(np.float32),rgb=rgb[py,px],pixel=np.stack((px,py),1).astype(np.int16),target=target,depth=d[yy,xx].astype(np.float32),distance=distance.astype(np.float32),onsets=np.asarray(onset,dtype=np.int32),source_box=np.asarray([x1,y1,x2,y2]))
 idx1=fps(xyz,512);xyz1=xyz[idx1]
 idx2=fps(xyz1,128);xyz2=xyz1[idx2]
 idx3=fps(xyz2,32);xyz3=xyz2[idx3]
 nb1=neighbors(xyz,xyz1,.04);nb2=neighbors(xyz1,xyz2,.08);nb3=neighbors(xyz2,xyz3,.16)
 fp3_idx,fp3_weight=interpolate(xyz3,xyz2)
 fp2_idx,fp2_weight=interpolate(xyz2,xyz1)
 fp1_idx,fp1_weight=interpolate(xyz1,xyz)
 np.savez_compressed(graph,idx1=idx1,idx2=idx2,idx3=idx3,nb1=nb1,nb2=nb2,nb3=nb3,fp3_idx=fp3_idx,fp3_weight=fp3_weight,fp2_idx=fp2_idx,fp2_weight=fp2_weight,fp1_idx=fp1_idx,fp1_weight=fp1_weight)
 return dict(id=e['id'],split=e['split'],points=len(xyz),events=len(onset),onsets=onset,target_mean=float(target.mean()),target_max=float(target.max()))
def main():
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   e=dict(E[id],split=split)
   sample=OUT/split/(id+'.npz');graph=OUT/'graphs'/split/(id+'.npz')
   if sample.exists() and graph.exists():
    rows.append(dict(id=id,split=split,reused=True));continue
   r=build(e);rows.append(r);print(json.dumps(r),flush=True)
 summary=dict(source='complete RGB-D sequences; image-only automatic removal and 3D interaction cue',no_simulator_truth=True,no_contact_training=True,split={k:len(S[k]) for k in ('train','val','test')},rows=rows)
 (OUT/'manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
 print(json.dumps(dict(complete=True,produced=sum('points' in x for x in rows),skipped=sum('reason' in x for x in rows))),flush=True)
if __name__=='__main__':main()
