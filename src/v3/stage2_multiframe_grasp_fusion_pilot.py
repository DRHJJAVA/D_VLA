import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,hashlib,numpy as np,h5py
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from stage2_probe_motion_fingerpads import cloud,project
from stage2_robot_only_depth import RobotDepth
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'audits/v3/motion_finger_contact_pilot_v1'
OLD=R/'outputs/v3_stage2/motion_contact_pointcloud_review_v1'
REG=R/'outputs/v3_stage2/interfinger_region_review_v1'
OUT=R/'outputs/v3_stage2/multiframe_grasp_fusion_v1'
def apply(p,T):return p@T[:3,:3].T+T[:3,3]
def voxel(p,mm):
 return np.unique(np.rint(p/mm).astype(np.int64),axis=0,return_index=True)[1] if len(p) else np.array([],int)
def rigid(a,b):
 ac=a.mean(0);bc=b.mean(0);u,s,vt=np.linalg.svd((a-ac).T@(b-bc));rot=vt.T@u.T
 if np.linalg.det(rot)<0:vt[-1]*=-1;rot=vt.T@u.T
 T=np.eye(4);T[:3,:3]=rot;T[:3,3]=bc-rot@ac;return T
def fit(p,c,ref,rc):
 tree=cKDTree(ref);T=np.eye(4);ix=voxel(p,.002);a=p[ix];col=c[ix]
 for k in range(20):
  x=apply(a,T);ds,nn=tree.query(x);match=(ds<.012)&(np.linalg.norm((col-rc[nn])/255.,axis=1)<.4)
  if match.sum()<40:return T,{'valid':False,'reason':'fewer_than_40_correspondences'}
  threshold=np.quantile(ds[match],.8);use=match&(ds<=threshold)
  dT=rigid(x[use],ref[nn[use]]);T=dT@T
  if np.linalg.norm(dT[:3,3])<1e-5 and Rotation.from_matrix(dT[:3,:3]).magnitude()<1e-4:break
 x=apply(a,T);ds,nn=tree.query(x);ok=(ds<.004)&(np.linalg.norm((col-rc[nn])/255.,axis=1)<.4)
 rev=cKDTree(x).query(ref)[0]
 trans=np.linalg.norm(T[:3,3]);deg=Rotation.from_matrix(T[:3,:3]).magnitude()*180/np.pi
 return T,dict(valid=bool(ok.mean()>=.45 and (rev<.004).mean()>=.30 and trans<=.025 and deg<=15),source_overlap_4mm=float(ok.mean()),reference_overlap_4mm=float((rev<.004).mean()),median_distance_mm=float(np.median(ds)*1000),correction_translation_mm=float(trans*1000),correction_angle_deg=float(deg))
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 protocol={'examples':'same four already-seen development demonstrations, six attempts','no_training':True,'no_object_truth_or_contact_truth':True,'motion_prior':'measured hand poses; actual observations aligned t-1','pregrasp_prior':'object stationary in world at first stable nonempty aperture; refined by RGBD ICP, assumption only','registration':'color-gated trimmed point ICP; reject under 40 matches, source overlap<0.45 or reference overlap<0.30 at4mm, correction>25mm or15deg','novel_point_gate':'support within2mm in >=2 DISTINCT non-anchor frames including >=1 held frame; pre-only kept separately uncertain','object_extraction':'robot-depth exclusion; reference visible object bounding box +15mm and RGB compatibility; no exact instance truth','important':'same-shape alignment is not exact surface identity; residual slip/symmetry uncertainty; no synthetic surface generation'}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 refs={x['id']:x for x in json.loads((R/'audits/v3/finger_contact_repair_pilot_v1/protocol.json').read_text())['episodes']}
 items=json.loads((D/'review_frames.json').read_text());summ=[]
 for it in items:
  ident,t=it['id'],it['frame'];name=ident+'_attempt'+str(it['attempt'])
  with np.load(OLD/(name+'.npz')) as z:base=z['xyz_world_m'][z['contact_estimate']==0];bc=z['rgb'][z['contact_estimate']==0].astype(float)
  with np.load(D/(ident+'.npz')) as z:H=z['hand_to_world'];q=z['measured_finger_q']
  inv=np.linalg.inv(H[t]);ref=apply(base,inv);sel=voxel(ref,.002);ref=ref[sel];rc=bc[sel]
  reflo=ref.min(0)-.015;refhi=ref.max(0)+.015
  with np.load(REG/(name+'.npz')) as z:lo=z['volume_min_hand_m'];hi=z['volume_max_hand_m'];anchorreg=z['xyz_hand_m'];anchorworld=z['xyz_world_m'];anchorcams=z['source_camera'];anchoruv=z['source_pixel']
  a=json.loads((D/(ident+'.json')).read_text())['attempts'][it['attempt']-1]
  if not it['accepted_by_motion_visual_gate']:
   np.savez_compressed(OUT/(name+'.npz'),anchor_surface_hand_m=ref,anchor_region_hand_m=anchorreg,fused_surface_hand_m=ref,fused_region_hand_m=np.empty((0,3)),uncertain_pre_hand_m=np.empty((0,3)),hand_to_world=H[t],volume_min_hand_m=lo,volume_max_hand_m=hi)
   s=dict(id=ident,attempt=it['attempt'],anchor_frame=t,held=False,anchor_region_points=0,fused_region_voxels=0,accepted_frames=[],rejected_frames=[],reason='failed attempt; no successful grasp region');summ.append(s);continue
  gap=q[:,0]-q[:,1]-.001;first=None
  for u in range(a['start_frame'],t+1):
   if u+4<len(q) and np.all(abs(gap[u:u+4]-gap[t])<.0015):first=u;break
  if first is None:first=a['accepted_contact_frames'][0]
  pretimes=[max(1,a['start_frame']-6),max(1,a['start_frame']-3),max(1,a['start_frame']-1)]
  early=list(range(max(a['start_frame'],first-4),first,2))
  heldtimes=sorted(set([first]+a['accepted_contact_frames'][::max(1,len(a['accepted_contact_frames'])//12)]+[t]))
  ts=sorted(set(pretimes+early+heldtimes))
  pts=[];rgbs=[];frames=[];cameras=[];pixels=[];original=[];transforms=[];records=[]
  with h5py.File(refs[ident]['rgbd'],'r') as f:
   robot=RobotDepth(f['agentview/rgb'].shape[1])
   for u in ts:
    if u==t:continue
    phase='pre' if u<first else 'held'
    prior=np.linalg.inv(H[first] if phase=='pre' else H[u])
    P=[];C=[];CAM=[];UV=[];W=[]
    for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
     g=f[cam];dep=g['depth_m'][u];K=g['K'][u];camT=g['camera_to_world'][u];rd=robot.render(q[u],np.linalg.inv(H[u])@camT,K)
     world=cloud(dep,K,camT);p=apply(world,prior);rgb=g['rgb'][u].reshape(-1,3).astype(float)
     valid=np.isfinite(p).all(1)&(dep.reshape(-1)>0)&(dep<rd-.002).reshape(-1)&np.all(p>reflo-.015,axis=1)&np.all(p<refhi+.015,axis=1)
     ix=np.flatnonzero(valid);v,us=np.unravel_index(ix,dep.shape)
     P.append(p[ix]);C.append(rgb[ix]);CAM.append(np.full(len(ix),ci));UV.append(np.stack([us,v],1));W.append(world[ix])
    p=np.concatenate(P);c=np.concatenate(C);camidx=np.concatenate(CAM);uv=np.concatenate(UV);world=np.concatenate(W)
    ds,nn=cKDTree(ref).query(p);fitmask=(ds<.02)&(np.linalg.norm((c-rc[nn])/255,axis=1)<.4)
    record=dict(frame=u,phase=phase,initial_points=len(p),registration_points=int(fitmask.sum()))
    if fitmask.sum()<40:record.update(valid=False,reason='too_few_initial_points');records.append(record);continue
    correction,diag=fit(p[fitmask],c[fitmask],ref,rc);record.update(diag)
    if not diag['valid']:records.append(record);continue
    p=apply(p,correction);ds,nn=cKDTree(ref).query(p)
    keep=np.all(p>=reflo,axis=1)&np.all(p<=refhi,axis=1)&(ds<.015)&(np.linalg.norm((c-rc[nn])/255,axis=1)<.4)
    # Reject points contradicting visible anchor depth; hidden/behind-surface observations remain possible.
    mapped=apply(p,H[t]);conflicts=np.zeros(len(p),bool)
    for cam in ['agentview','robot0_eye_in_hand']:
     g=f[cam];dep=g['depth_m'][t];pv,zv=project(mapped,g['K'][t],g['camera_to_world'][t]);pix=np.rint(pv).astype(int)
     inside=(zv>0)&(pix[:,0]>=0)&(pix[:,0]<dep.shape[1])&(pix[:,1]>=0)&(pix[:,1]<dep.shape[0]);ii=np.flatnonzero(inside);d=dep[pix[ii,1],pix[ii,0]]
     conflicts[ii]|=(d>0)&(zv[ii]<d-.004)
    keep&=~conflicts
    record['retained_points']=int(keep.sum());record['anchor_free_space_rejected']=int(conflicts.sum());record['source_world_to_anchor_hand']=(correction@prior).tolist();records.append(record)
    if keep.sum():
     pts.append(p[keep]);rgbs.append(c[keep]);frames.append(np.full(keep.sum(),u));cameras.append(camidx[keep]);pixels.append(uv[keep]);original.append(world[keep]);transforms.append(correction@prior)
   robot.close()
  p=np.concatenate(pts) if pts else np.empty((0,3));c=np.concatenate(rgbs) if pts else np.empty((0,3))
  fr=np.concatenate(frames) if pts else np.empty(0,int);cam=np.concatenate(cameras) if pts else np.empty(0,int);uv=np.concatenate(pixels) if pts else np.empty((0,2),int);world=np.concatenate(original) if pts else np.empty((0,3))
  # Choose an actually observed sample in each voxel, not an averaged/synthetic point.
  chosen=voxel(p,.001);cand=p[chosen];support=np.zeros(len(cand),int);heldsupport=np.zeros(len(cand),int)
  for u in np.unique(fr):
   ok=cKDTree(p[fr==u]).query(cand)[0]<.002;support+=ok;heldsupport+=ok*(u>=first)
  stable=(support>=2)&(heldsupport>=1);use=chosen[stable]
  # Preserve observed reference surface and new surface; support numbers describe frames, not grasp frequency.
  allsurf=np.r_[ref,p[use]];vi=voxel(allsurf,.001);fused=allsurf[vi]
  region=lambda x:np.all(x>=lo,axis=1)&np.all(x<=hi,axis=1)
  # Include all single-frame region observations, then add supported multi-frame samples.
  rr=np.r_[anchorreg,p[use][region(p[use])]];freg=rr[voxel(rr,.001)]
  pre=chosen[(support>=2)&(heldsupport==0)];prepts=p[pre];uncertain=prepts[region(prepts)]
  added=freg[cKDTree(anchorreg).query(freg)[0]>.002] if len(anchorreg) else freg
  np.savez_compressed(OUT/(name+'.npz'),anchor_surface_hand_m=ref,anchor_region_hand_m=anchorreg,fused_surface_hand_m=fused,fused_region_hand_m=freg,added_region_hand_m=added,uncertain_pre_hand_m=uncertain,
   source_aligned_hand_m=p,source_world_m=world,source_rgb=c.astype(np.uint8),source_frame=fr,source_camera=cam,source_pixel=uv,chosen_sample_indices=chosen,chosen_frame_support=support,chosen_held_frame_support=heldsupport,accepted_sample_indices=use,hand_to_world=H[t],volume_min_hand_m=lo,volume_max_hand_m=hi)
  report=dict(id=ident,attempt=it['attempt'],anchor_frame=t,held=True,first_stable_aperture=first,anchor_region_points=len(anchorreg),anchor_region_voxels=len(voxel(anchorreg,.001)),fused_region_voxels=len(freg),new_region_voxels_farther_than_2mm=len(added),uncertain_pre_region_voxels=len(uncertain),anchor_surface_voxels=len(ref),fused_surface_voxels=len(fused),accepted_frames=[r['frame'] for r in records if r.get('valid')],accepted_pre_frames=[r['frame'] for r in records if r.get('valid') and r['phase']=='pre'],rejected_frames=[r['frame'] for r in records if not r.get('valid')],registration_records=records)
  (OUT/(name+'.json')).write_text(json.dumps(report,indent=2))
  summ.append({k:v for k,v in report.items() if k!='registration_records'});print(json.dumps(summ[-1]),flush=True)
 (OUT/'summary.json').write_text(json.dumps(summ,indent=2));print('complete',flush=True)
if __name__=='__main__':main()
