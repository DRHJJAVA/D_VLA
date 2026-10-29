import numpy as np,json,h5py,traceback
from pathlib import Path
from scipy.spatial import cKDTree
B=Path('/mnt/nvme1/drh/D_VLA');R=B/'experiments/contact_preference/multiview_temporal_fusion_v1'
def normals(p,cam,k):
 if len(p)<k:return np.zeros_like(p),np.zeros(len(p),bool)
 tree=cKDTree(p);out=np.zeros_like(p);ok=np.zeros(len(p),bool)
 for start in range(0,len(p),4096):
  ids=np.arange(start,min(start+4096,len(p)));_,nn=tree.query(p[ids],k=k)
  q=p[nn];q=q-q.mean(1)[:,None,:];cov=np.einsum('nki,nkj->nij',q,q)/k
  e,v=np.linalg.eigh(cov);n=v[:,:,0];n*=np.where(np.sum(n*(cam-p[ids]),1)>=0,1.,-1.)[:,None]
  out[ids]=n;ok[ids]=(e[:,1]>1e-12)&np.isfinite(n).all(1)
 return out,ok
def main():
 s=json.loads((R/'summary.json').read_text())
 for c in s['cases']:
  key=c['key'];z=dict(np.load(R/'cases'/(key+'.npz')));p=z['common_xyz'];rgb=z['rgb'];frames=c['frames'];limit=c['color_l2_limit_q95']
  reg=np.load(B/'experiments/region_pose/iter_024/train'/(key+'.npz'))
  ref=z['reference_xyz'];qcam=reg['query_camera_to_world'][:3,3]
  refn={k:normals(ref,qcam,k) for k in [12,24]};_,nearref=cKDTree(ref).query(p)
  normals_ok=np.ones(len(p),bool);dots=np.zeros((2,len(p)))
  asset=key.split('_demo_')[0];demo='demo_'+key.split('_demo_')[1].split('_')[0]
  with h5py.File(B/'data/v3/trainval_rgbd_448'/asset/(demo+'.h5'),'r') as h:
   for vi in np.unique(z['view_id']):
    ids=np.flatnonzero(z['view_id']==vi);t=int(z['frame'][ids[0]]);camera=int(z['camera'][ids[0]]);cam=['agentview','robot0_eye_in_hand'][camera]
    W=np.eye(4) if t==c['query_frame'] else reg['hand_to_world_%d'%t]@reg['query_to_current_hand_%d'%t]
    cp=np.linalg.solve(W,np.r_[h[cam+'/camera_to_world'][t,:3,3],1.])[:3]
    for j,k in enumerate([12,24]):
     n,ok=normals(p[ids],cp,k);rn,rok=refn[k]
     d=np.einsum('ij,ij->i',n,rn[nearref[ids]]);dots[j,ids]=d
     normals_ok[ids]&=ok&rok[nearref[ids]]&(d>0)
  eligible=z['connected']&z['stable']&normals_ok
  witness=np.full((len(p),len(frames)),-1,np.int32);fs=np.zeros(len(p),np.int16);mf=np.zeros(len(p),np.int16)
  for j,t in enumerate(frames):
   for mainonly in [False,True]:
    ids=np.flatnonzero((z['frame']==t)&eligible&((z['camera']==0) if mainonly else True))
    if not len(ids):continue
    dd,ii=cKDTree(p[ids]).query(p,k=min(4,len(ids)))
    if dd.ndim==1:dd=dd[:,None];ii=ii[:,None]
    near=ids[ii];good=(dd<=.003)&(np.linalg.norm(rgb[:,None,:]/255.-rgb[near]/255.,axis=2)<=limit+1e-12)
    has=good.any(1)
    if mainonly:mf+=has
    else:
     fs+=has;which=good.argmax(1);witness[has,j]=near[np.arange(len(p))[has],which[has]]
  supported=eligible&(fs>=2)&(z['ref_distance']<=.003)&(z['color_distance']<=limit+1e-12)
  reps=[]
  for mask in [supported,~supported]:
   ids=np.flatnonzero(mask);ids=ids[np.lexsort((ids,z['loo_shift'][ids]))]
   _,ix=np.unique(z['voxel'][ids],axis=0,return_index=True);reps.append(ids[ix])
  si,ui=reps
  mainset=set(map(tuple,z['voxel'][supported&(z['camera']==0)&(mf>=2)]))
  wextra=np.array([tuple(v) not in mainset for v in z['voxel']])&supported
  z.update(normal_consistent=normals_ok,normal_dot=dots,supported=supported,unknown=~supported,distinct_frame_support=fs,main_frame_support=mf,witness=witness,supported_representatives=si,unknown_representatives=ui,wrist_new_voxel=wextra)
  np.savez_compressed(R/'refined_cases'/(key+'.npz'),**z)
  c.update(initial_supported_voxels=c['supported_voxels'],normal_excluded_observations=int((~normals_ok).sum()),supported_observations=int(supported.sum()),unknown_observations=int((~supported).sum()),supported_voxels=len(si),unknown_voxels=len(ui),new_supported_voxels=int(z['new_vs_reference'][si].sum()),wrist_contributed_beyond_main_supported_voxels=int(wextra[si].sum()),wrist_required_observations=int((supported&(mf<2)).sum()))
  for row in c['rows']:
   sel=(z['frame']==row['frame'])&(z['camera']==row['camera']);row.update(supported=int((supported&sel).sum()),unknown=int((~supported&sel).sum()),normal_consistent=int((normals_ok&sel).sum()))
  (R/'refined_cases'/(key+'.json')).write_text(json.dumps(c,ensure_ascii=False,indent=2))
  print(json.dumps({'case':c['case'],'supported_voxels':len(si),'new_voxels':c['new_supported_voxels'],'wrist_extra':c['wrist_contributed_beyond_main_supported_voxels']}),flush=True)
 s['refinement']='two_scale_source_normal_identity'
 (R/'refined_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:(R/'refine_failure.log').write_text(traceback.format_exc());print(json.dumps({'error':type(e).__name__}));raise SystemExit(1)
