import os
os.environ['MUJOCO_GL']='egl'
import sys,json,hashlib,traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,h5py
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
B=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(B/'src/v3'))
from stage2_robot_only_depth import RobotDepth
R=B/'experiments/contact_preference/multiview_temporal_fusion_v1'
cfg=json.loads((R/'config.json').read_text())
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def apply(p,T):return np.c_[p,np.ones(len(p))]@T.T[:,:3]
def back(dep,K,T):
 y,x=np.indices(dep.shape);uv=np.stack([x,y,np.ones_like(x)],-1)
 pc=(uv@np.linalg.inv(K).T)*dep[...,None]
 return apply(pc.reshape(-1,3),T).reshape(*dep.shape,3)
def project(x,K,T):
 p=apply(x,np.linalg.inv(T));v=p@K.T
 return v[:,:2]/v[:,2:],p[:,2]
def main():
 renderer=RobotDepth(448);summary=[]
 for ci,c in enumerate(cfg['cases']):
  dump(R/'stage_state.json',{'stage':'processing','case':ci,'time':now()})
  key=c['key'];qframe=int(key.split('query')[-1]);frames=[qframe]+c['frames']
  reg=np.load(B/'experiments/region_pose/iter_024/train'/(key+'.npz'))
  mot=np.load(B/'experiments/region_pose/iter_003/evidence/motion'/(c['asset']+'_'+c['demo']+'.npz'))
  ref=reg['query_xyz'];rgbref=reg['query_rgb']/255.;tree=cKDTree(ref)
  sig=np.concatenate([np.load(B/'experiments/contact_preference/stage3_pilot_v2/points_evidence'/(key+'_frame%d.npz'%t))['sigma_proxy_m'] for t in c['frames']])
  expansion=max(.003,3*float(np.median(sig)));lo=ref.min(0)-expansion;hi=ref.max(0)+expansion
  views=[];colorerrors=[];slicehash={}
  with h5py.File(B/'data/v3/trainval_rgbd_448'/c['asset']/(c['demo']+'.h5'),'r') as h:
   for t in frames:
    H=mot['hand_to_world'][t].astype(float);q=mot['measured_finger_q'][t]
    if t==qframe:W=np.eye(4);loos=np.eye(4)[None]
    else:
     assert np.allclose(H,reg['hand_to_world_%d'%t],atol=1e-7)
     W=H@reg['query_to_current_hand_%d'%t]
     a=np.load(B/'experiments/region_pose/iter_024/train_loo'/(key+'_frame%d.npz'%t))
     loos=np.stack([np.linalg.inv(H@x) for x in a['loo_Q']])
    for camera,cam in enumerate(cfg['cameras']):
     dep=h[cam+'/depth_m'][t].astype(float);rgb=h[cam+'/rgb'][t];K=h[cam+'/K'][t].astype(float);T=h[cam+'/camera_to_world'][t].astype(float)
     for field in ['depth_m','rgb','K','camera_to_world']:
      a=h[cam+'/'+field][t];slicehash[cam+'/'+field+'/'+str(t)]=hashlib.sha256(a.tobytes()).hexdigest()
     rd=renderer.render(q,np.linalg.solve(H,T),K).copy()
     world=back(dep,K,T);common=apply(world.reshape(-1,3),np.linalg.inv(W)).reshape(448,448,3)
     valid=np.isfinite(dep)&(dep>0)&(dep<5)
     roi=valid&np.all((common>=lo)&(common<=hi),axis=2)
     keep=roi&(dep<rd-.002)
     y,x=np.nonzero(keep);pw=world[y,x];p=common[y,x];cols=rgb[y,x]
     d,nn=tree.query(p)
     shift=np.zeros(len(p))
     for Linv in loos:shift=np.maximum(shift,np.linalg.norm(apply(pw,Linv)-p,axis=1))
     # Calibrate appearance on visible prior source projections, never contact labels.
     uv,z=project(apply(ref,W),K,T);ij=np.rint(uv).astype(int)
     ok=(z>0)&(ij[:,0]>=0)&(ij[:,0]<448)&(ij[:,1]>=0)&(ij[:,1]<448)
     ids=np.flatnonzero(ok);u,v=ij[ids].T
     ok2=(abs(z[ids]-dep[v,u])<=.003)&(dep[v,u]<rd[v,u]-.002)
     ids=ids[ok2];u,v=ij[ids].T
     cerr=np.linalg.norm(rgb[v,u]/255.-rgbref[ids],axis=1)
     if len(cerr):colorerrors.append(cerr)
     views.append(dict(frame=t,camera=camera,p=p,world=pw,rgb=cols,pixel=np.c_[x,y],ref_distance=d,ref_index=nn,shift=shift,
       roi_count=int(roi.sum()),robot_rejected=int((roi&~keep).sum()),color_anchor_count=len(cerr),K=K,T=T,W=W,depth=dep,robot_depth=rd))
  color_limit=float(np.quantile(np.concatenate(colorerrors),.95)) if colorerrors else 0.
  for vi,v in enumerate(views):
   p=v['p'];pix=v['pixel'];n=len(p);cerr=np.linalg.norm(v['rgb']/255.-rgbref[v['ref_index']],axis=1)
   seed=(v['ref_distance']<=.003)&(cerr<=color_limit+1e-12)&(v['shift']<=.002)
   index=np.full((448,448),-1,dtype=int);index[pix[:,1],pix[:,0]]=np.arange(n)
   rows=[];cols=[]
   for dy,dx in [(0,1),(1,0)]:
    inb=(pix[:,0]+dx<448)&(pix[:,1]+dy<448);a=np.flatnonzero(inb)
    other=index[pix[a,1]+dy,pix[a,0]+dx];yes=other>=0;a=a[yes];other=other[yes]
    # Surface continuity only, not semantic object identity.
    footprint=np.maximum(v['depth'][pix[a,1],pix[a,0]],v['depth'][pix[other,1],pix[other,0]])/min(v['K'][0,0],v['K'][1,1])
    yes=np.linalg.norm(p[a]-p[other],axis=1)<=np.maximum(.003,3*footprint)
    rows.extend(a[yes]);cols.extend(other[yes])
   graph=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)).tocsr()
   nc,labels=connected_components(graph,directed=False) if n else (0,np.zeros(0,dtype=int))
   connected=np.isin(labels,np.unique(labels[seed]))
   v.update(seed=seed,connected=connected,color_distance=cerr)
  # Consolidate actual sources without inventing intermediate positions.
  starts=np.cumsum([0]+[len(v['p']) for v in views])
  p=np.concatenate([v['p'] for v in views]);pw=np.concatenate([v['world'] for v in views]);rgb=np.concatenate([v['rgb'] for v in views]);pixels=np.concatenate([v['pixel'] for v in views])
  frame=np.concatenate([np.full(len(v['p']),v['frame']) for v in views]);camera=np.concatenate([np.full(len(v['p']),v['camera']) for v in views])
  viewid=np.concatenate([np.full(len(v['p']),i) for i,v in enumerate(views)])
  connected=np.concatenate([v['connected'] for v in views]);stable=np.concatenate([v['shift']<=.002 for v in views]);refdist=np.concatenate([v['ref_distance'] for v in views])
  sh=np.concatenate([v['shift'] for v in views]);colordist=np.concatenate([v['color_distance'] for v in views])
  eligible=connected&stable
  frame_support=np.zeros(len(p),np.int16);camera_support=np.zeros(len(p),np.int8)
  witness=np.full((len(p),len(frames)),-1,dtype=np.int32)
  for j,t in enumerate(frames):
   ids=np.flatnonzero((frame==t)&eligible)
   if not len(ids):continue
   dd,ii=cKDTree(p[ids]).query(p,k=min(4,len(ids)))
   if dd.ndim==1:dd=dd[:,None];ii=ii[:,None]
   near=ids[ii];good=(dd<=.003)&(np.linalg.norm(rgb[:,None,:]/255.-rgb[near]/255.,axis=2)<=color_limit+1e-12)
   has=good.any(1);which=good.argmax(1);witness[has,j]=near[np.arange(len(p))[has],which[has]]
   frame_support+=has
  for cameraid in [0,1]:
   ids=np.flatnonzero((camera==cameraid)&eligible)
   if len(ids):
    dd,ii=cKDTree(p[ids]).query(p);camera_support+=(dd<=.003)&(np.linalg.norm(rgb/255.-rgb[ids[ii]]/255.,axis=1)<=color_limit+1e-12)
  supported=eligible&(frame_support>=2)&(refdist<=.003)&(colordist<=color_limit+1e-12)
  # All other source candidates remain unknown, including beyond known surface identity.
  unknown=~supported
  vox=np.floor(p/.001).astype(np.int64);refvox=np.floor(ref/.001).astype(np.int64)
  refset=set(map(tuple,refvox));new=np.array([tuple(x) not in refset for x in vox])
  representatives=[]
  for mask in [supported,unknown]:
   ids=np.flatnonzero(mask)
   if len(ids):
    # Pick one measured source; favor small LOO shift, then source index.
    ids=ids[np.lexsort((ids,sh[ids]))];_,ix=np.unique(vox[ids],axis=0,return_index=True);representatives.append(ids[ix])
   else:representatives.append(np.array([],dtype=int))
  si,ui=representatives
  # Separate wrist contributions from points already supported by main-only observation.
  main_frames=np.zeros(len(p),np.int16)
  for j,t in enumerate(frames):
   ids=np.flatnonzero((frame==t)&(camera==0)&eligible)
   if len(ids):
    dd,ii=cKDTree(p[ids]).query(p,k=min(4,len(ids)))
    if dd.ndim==1:dd=dd[:,None];ii=ii[:,None]
    near=ids[ii];main_frames+=((dd<=.003)&(np.linalg.norm(rgb[:,None,:]/255.-rgb[near]/255.,axis=2)<=color_limit+1e-12)).any(1)
  wrist_added=supported&(main_frames<2)
  main_supported=supported&(camera==0)&(main_frames>=2)
  main_vox=set(map(tuple,vox[main_supported]))
  wrist_new_voxel=np.array([tuple(x) not in main_vox for x in vox])&supported
  meta={'key':key,'title':c['title'],'case':ci,'query_frame':qframe,'frames':frames,'roi_expansion_m':expansion,'color_l2_limit_q95':color_limit,'color_anchor_count':sum(v['color_anchor_count'] for v in views),'source_observations':len(p),'supported_observations':int(supported.sum()),'unknown_observations':int(unknown.sum()),'supported_voxels':len(si),'unknown_voxels':len(ui),'reference_voxels':len(refset),'new_supported_voxels':int(new[si].sum()),'wrist_contributed_beyond_main_supported_voxels':int(wrist_new_voxel[si].sum()),'wrist_required_observations':int(wrist_added.sum()),'rows':[]}
  for vi,v in enumerate(views):
   sl=slice(starts[vi],starts[vi+1])
   meta['rows'].append({k:v[k] for k in ['frame','camera','roi_count','robot_rejected','color_anchor_count']}|{'source_count':len(v['p']),'connected':int(v['connected'].sum()),'stable':int(stable[sl].sum()),'supported':int(supported[sl].sum()),'unknown':int(unknown[sl].sum())})
  np.savez_compressed(R/'cases'/(key+'.npz'),common_xyz=p,source_world_xyz=pw,rgb=rgb,pixel=pixels,frame=frame,camera=camera,view_id=viewid,reference_xyz=ref,reference_rgb=reg['query_rgb'],reference_pixel=reg['query_pixel'],ref_distance=refdist,loo_shift=sh,connected=connected,stable=stable,color_distance=colordist,distinct_frame_support=frame_support,camera_support=camera_support,witness=witness,supported=supported,unknown=unknown,voxel=vox,new_vs_reference=new,supported_representatives=si,unknown_representatives=ui,main_frame_support=main_frames,wrist_new_voxel=wrist_new_voxel)
  dump(R/'cases'/(key+'.json'),meta);dump(R/'cases'/(key+'_slices_locked.json'),slicehash)
  summary.append(meta);dump(R/'summary.json',{'time':now(),'cases':summary,'training':False,'contact_labels_generated':False})
  print(json.dumps({'case':ci,'source':len(p),'supported_voxels':len(si),'new_voxels':int(new[si].sum()),'wrist_extra':int(wrist_new_voxel[si].sum())}),flush=True)
 renderer.close()
 dump(R/'stage_state.json',{'stage':'computed_pending_audit','time':now()})
if __name__=='__main__':
 try:main()
 except Exception as e:
  (R/'failure.log').write_text(traceback.format_exc())
  dump(R/'failure.json',{'time':now(),'type':type(e).__name__,'message':str(e)})
  dump(R/'stage_state.json',{'stage':'failed','time':now(),'type':type(e).__name__})
  print(json.dumps({'error':type(e).__name__}),flush=True)
  raise SystemExit(1)
