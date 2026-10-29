"""Diagnostic only: co-moving near occluders of tracked partial object surface.
Reads RGB-D and frozen visual poses, never contact/actions/object simulator labels.
"""
from pathlib import Path
import json,hashlib
import cv2,h5py,numpy as np
from scipy.spatial import cKDTree
from stage2_track_surface_klt_rgbd import cloud
R=Path('/mnt/nvme1/drh/D_VLA')
BASE=R/'audits/v3/surface_klt_rgbd_pilot_v1'
DENSE=R/'audits/v3/surface_klt_rgbd_dense_reprojection_v1'
OUT=R/'audits/v3/comoving_occluder_pilot_v1'
CAMS=['agentview','robot0_eye_in_hand']
def normals(points,camera):
 dist,ind=cKDTree(points).query(points,k=min(20,len(points)))
 nn=np.zeros_like(points);good=np.zeros(len(points),bool)
 for i in range(len(points)):
  a=points[ind[i][dist[i]<.015]]
  if len(a)<10:continue
  cov=(a-a.mean(0)).T@(a-a.mean(0));e,v=np.linalg.eigh(cov)
  n=v[:,0]
  if n@(camera-points[i])<0:n=-n
  nn[i]=n;good[i]=(e[0]/max(e.sum(),1e-12)<.1 and e[1]/max(e[2],1e-12)>.05)
 return nn,good
def runs(mask):
 starts=np.flatnonzero(mask&~np.r_[False,mask[:-1]]);ends=np.flatnonzero(mask&~np.r_[mask[1:],False])
 return [(int(a),int(b)) for a,b in zip(starts,ends)]
def candidates(points,normals,normal_good,rot,tr,cam,t,tree):
 moved=points@rot.T+tr;normal=normals@rot.T
 K=cam['K'][t];T=cam['camera_to_world'][t];dep=cam['depth_m'][t]
 c=(moved-T[:3,3])@T[:3,:3];z=c[:,2]
 uv=np.stack((K[0,0]*c[:,0]/np.maximum(z,.001)+K[0,2],K[1,1]*c[:,1]/np.maximum(z,.001)+K[1,2]),1)
 tocam=T[:3,3]-moved;cos=np.sum(normal*tocam,1)/np.maximum(np.linalg.norm(tocam,axis=1),1e-8)
 inside=(z>.05)&(uv[:,0]>=1)&(uv[:,0]<dep.shape[1]-2)&(uv[:,1]>=1)&(uv[:,1]<dep.shape[0]-2)
 ix=np.flatnonzero(inside)
 pix=np.rint(uv[ix]).astype(int)
 # Z-buffer excludes modeled self-occlusion using all reconstructed points.
 linear=pix[:,1]*dep.shape[1]+pix[:,0];order=np.lexsort((z[ix],linear))
 _,first=np.unique(linear[order],return_index=True);ix=ix[order[first]];pix=np.rint(uv[ix]).astype(int)
 ix=ix[normal_good[ix]&(cos[ix]>.3)];pix=np.rint(uv[ix]).astype(int)
 observed=dep[pix[:,1],pix[:,0]];gap=z[ix]-observed
 raw=(observed>.05)&np.isfinite(observed)&(gap>.005)&(gap<.040)
 selected=ix[raw];pixel=np.rint(uv[selected]).astype(int)
 if len(selected):
  qw,_=cloud(pixel.astype(float),dep,K,T);q0=(qw-tr)@rot
  dist,_=tree.query(q0,k=1)
  other=dist>.004
  selected=selected[other];q0=q0[other];pixel=pixel[other]
 else:q0=np.empty((0,3));pixel=np.empty((0,2),int)
 q=np.full((len(points),3),np.nan,np.float32);q[selected]=q0
 return q,dict(front_facing_zbuffer_points=len(ix),near_occluders_before_surface_exclusion=int(raw.sum()),
   near_unmodeled_occluders=len(selected)),uv
def main():
 cv2.setNumThreads(4);OUT.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((BASE/'protocol.json').read_text())
 config=dict(source_protocol_sha256=hashlib.sha256((BASE/'protocol.json').read_bytes()).hexdigest(),
   episodes=[{'id':x['id'],'split':x['split']} for x in manifest['episodes']],
   normal_neighbors=20,normal_radius_m=.015,normal_min_points=10,normal_plane_ratio=.1,
   facing_cosine=.3,occluder_depth_gap_m=[.005,.040],exclude_known_surface_distance_m=.004,
   minimum_consecutive_pose_frames=5,minimum_window_centroid_span_m=.01,
   persistence_frames=3,minimum_three_frame_centroid_motion_m=.005,
   max_occluder_object_coordinate_spread_m=.008,both_view_occluder_agreement_m=.010,
   output='unidentified co-moving near occluder hypotheses only; never grasp/contact/wrapped labels',
   no_contact=True,no_actions=True,no_object_truth=True,no_reserved=True)
 cp=OUT/'protocol.json';text=json.dumps(config,indent=2)
 if cp.exists():assert cp.read_text()==text
 else:cp.write_text(text)
 sha=hashlib.sha256(cp.read_bytes()).hexdigest();print('protocol_locked',sha,flush=True)
 summary=[]
 for row in manifest['episodes']:
  dest=OUT/(row['id']+'.json')
  if dest.exists():summary.append(json.loads(dest.read_text()));continue
  audit=json.loads((BASE/(row['id']+'.json')).read_text())
  depthaudit=json.loads((DENSE/(row['id']+'.json')).read_text())
  with np.load(BASE/(row['id']+'.npz')) as z:poses=z['pose']
  with np.load(row['component']) as z:pix=z['pixel'].astype(int)
  n=row['frames'];rawcounts=[];images=[]
  with h5py.File(row['rgbd'],'r') as f:
   a=f['agentview'];rgb0=a['rgb'][0];p,valid=cloud(pix.astype(float),a['depth_m'][0],a['K'][0],a['camera_to_world'][0])
   p=p[valid];pix=pix[valid];norm,ng=normals(p,a['camera_to_world'][0][:3,3]);tree=cKDTree(p);center=p.mean(0)
   centers=np.full((n,3),np.nan)
   accepted=np.array([v['pose_candidate'] for v in audit['rows']],bool)
   for t in np.flatnonzero(accepted):centers[t]=poses[t,:3,:3]@center+poses[t,:3,3]
   windows=[]
   for start,end in runs(accepted):
    if end-start+1<5:continue
    c=centers[start:end+1];span=np.max(np.linalg.norm(c[:,None]-c[None],axis=2))
    if span>=.01:windows.append(dict(start=start,end=end,span_m=float(span)))
   active=np.zeros(n,bool)
   for w in windows:active[w['start']:w['end']+1]=True
   observations=np.full((n,2,len(p),3),np.nan,np.float32)
   for t in np.flatnonzero(active):
    tr=poses[t,:3,3];rot=poses[t,:3,:3]
    for ci,cam in enumerate(CAMS):
     q,stats,_=candidates(p,norm,ng,rot,tr,f[cam],t,tree)
     observations[t,ci]=q;rawcounts.append(dict(frame=int(t),camera=cam,**stats))
   persistent=np.zeros((n,2,len(p)),bool)
   for w in windows:
    for t in range(w['start']+2,w['end']+1):
     c=centers[t-2:t+1];motion=np.max(np.linalg.norm(c[:,None]-c[None],axis=2))
     if motion<.005:continue
     for ci in range(2):
      q=observations[t-2:t+1,ci];finite=np.isfinite(q).all((0,2))
      ix=np.flatnonzero(finite)
      if not len(ix):continue
      sub=q[:,ix];spread=np.max(np.linalg.norm(sub[:,None]-sub[None],axis=-1),axis=(0,1))
      persistent[t,ci,ix[spread<.008]]=True
   rawvalid=np.isfinite(observations).all(-1)
   both=rawvalid[:,0]&rawvalid[:,1]
   pairdist=np.linalg.norm(observations[:,0]-observations[:,1],axis=-1)
   two_view=both&(pairdist<.010)&(persistent[:,0]|persistent[:,1])
   counts=persistent.sum(2);union=persistent.any(1).any(0)
   framecounts=persistent.any(1).sum(1)
   if active.any():
    # Representative frame deliberately maximizes hypothesis count, not a test metric.
    t=int(np.argmax(framecounts)) if framecounts.max()>0 else int(np.flatnonzero(active)[-1])
    surface=union if framecounts.max()==0 else persistent[t].any(0)
    im0=cv2.cvtColor(rgb0,cv2.COLOR_RGB2BGR)
    for u,v in pix[surface]:cv2.circle(im0,(int(u),int(v)),2,(0,220,255),-1)
    panels=[im0]
    for ci,cam in enumerate(CAMS):
     im=cv2.cvtColor(f[cam]['rgb'][t],cv2.COLOR_RGB2BGR)
     idx=np.flatnonzero(persistent[t,ci])
     if len(idx):
      qw=observations[t,ci,idx]@poses[t,:3,:3].T+poses[t,:3,3]
      K=f[cam]['K'][t];T=f[cam]['camera_to_world'][t];cc=(qw-T[:3,3])@T[:3,:3]
      uv=np.rint(np.stack((K[0,0]*cc[:,0]/cc[:,2]+K[0,2],K[1,1]*cc[:,1]/cc[:,2]+K[1,2]),1)).astype(int)
      for u,v in uv:
       if 0<=u<im.shape[1] and 0<=v<im.shape[0]:cv2.circle(im,(int(u),int(v)),2,(0,220,255),-1)
     panels.append(im)
    canvas=np.concatenate(panels,1);cv2.rectangle(canvas,(0,0),(canvas.shape[1],40),(0,0,0),-1)
    title=row['id']+' t'+str(t)+' first-surface / main occluder / wrist occluder - NOT GRASP LABEL'
    cv2.putText(canvas,title,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    ip=OUT/(row['id']+'_representative.jpg');assert not ip.exists();cv2.imwrite(str(ip),canvas,[cv2.IMWRITE_JPEG_QUALITY,90])
    representative=t
   else:representative=None
   npz=OUT/(row['id']+'.npz');assert not npz.exists()
   np.savez_compressed(npz,first_xyz=p,first_pixel=pix,normals=norm,normal_reliable=ng,active=active,
     observed_occluder_object_frame=observations,persistent_hypothesis=persistent,both_view_consistent=two_view,protocol_sha256=sha)
  result=dict(id=row['id'],split=row['split'],protocol_sha256=sha,frames=n,initial_surface_points=len(p),
    normal_supported_points=int(ng.sum()),windows=windows,eligible_frames=int(active.sum()),
    raw_frame_counts=rawcounts,persistent_frame_counts=[{'frame':int(t),'main':int(counts[t,0]),'wrist':int(counts[t,1]),'two_view':int(two_view[t].sum())} for t in np.flatnonzero(active)],
    unique_surface_points_with_hypothesis=int(union.sum()),unique_points_two_view=int(two_view.any(0).sum()),
    max_hypothesis_points_in_frame=int(framecounts.max()),representative_frame=representative,
    caveat='candidate may be unseen object self-surface, neighboring object, gripper or pose error; no semantic identity, no contact/wrapped supervision')
  dest.write_text(json.dumps(result,indent=2));summary.append(result)
  print(json.dumps({k:v for k,v in result.items() if k not in ['raw_frame_counts','persistent_frame_counts']}),flush=True)
 out=OUT/'summary.json';assert not out.exists();out.write_text(json.dumps({'protocol':config,'episodes':summary},indent=2))
 print('complete',len(summary),flush=True)
if __name__=='__main__':main()
