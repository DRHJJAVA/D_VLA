from pathlib import Path
import os,sys,json,hashlib,importlib.util,traceback
os.environ['MUJOCO_GL']='egl'
import numpy as np,h5py
from scipy.spatial import cKDTree,ConvexHull
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/occluded_surface_fit_v1';F=R/'experiments/contact_preference/multiview_temporal_fusion_v1';B=R/'experiments/contact_preference/stage3_pilot_v2'
sys.path.insert(0,str(R))
from src.v3.stage2_robot_only_depth import RobotDepth
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
geo=module('known_geometry',R/'experiments/contact_preference/stage3_pilot_v1/src/build_supports_v2.py')
vismod=module('current_visibility',R/'experiments/contact_preference/stage3_temporal_support_v3/src/run.py')
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def apply(T,p):return p@T[:3,:3].T+T[:3,3]
def features(uv,quadratic):
 u,v=uv.T
 return np.column_stack([np.ones(len(u)),u,v,u*u,u*v,v*v]) if quadratic else np.column_stack([np.ones(len(u)),u,v])
def robust_fit(A,y):
 if np.linalg.cond(A)>1e6:return None
 c=np.linalg.lstsq(A,y,rcond=None)[0]
 for _ in range(3):
  res=y-A@c;scale=max(float(1.4826*np.median(abs(res-np.median(res)))),1e-7)
  w=np.minimum(1,1.345*scale/np.maximum(abs(res),1e-12))
  c=np.linalg.lstsq(A*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
 return c,float(np.sqrt(np.mean((y-A@c)**2)))
def fit_patch(center,pts,indices,radius,camera):
 if len(indices)<12:return None
 pts=pts[indices];C=np.mean(pts,0);ev,E=np.linalg.eigh(np.cov(pts.T))
 if ev[1]<1e-12:return None
 normal=E[:,0];normal*=1 if np.dot(normal,camera-center)>0 else -1
 e1=E[:,2];e2=np.cross(normal,e1);basis=np.column_stack([e1,e2,normal])
 local=(pts-C)@basis;uv=local[:,:2]/radius
 try:hull=ConvexHull(uv).equations
 except Exception:return None
 models=[]
 for q in [False,True]:
  result=robust_fit(features(uv,q),local[:,2])
  if result is None:return None
  co,res=result
  if not q:co=np.r_[co,0,0,0]
  models.append({'coef':co.tolist(),'rms':res,'quadratic':q})
 return {'center':C.tolist(),'basis':basis.tolist(),'radius':radius,'indices':indices.tolist(),'hull':hull.tolist(),'models':models,'principal_scales':np.sqrt(np.maximum(ev,0)).tolist()}
def intersect(patch,g,n,model,search=.008):
 C=np.array(patch['center']);E=np.array(patch['basis']);r=patch['radius'];c=np.array(model['coef'])
 l=(g-C)@E;v=n@E
 u0,v0=l[:2]/r;du,dv=v[:2]/r
 A=-(c[3]*du*du+c[4]*du*dv+c[5]*dv*dv)
 B=v[2]-c[1]*du-c[2]*dv-2*c[3]*u0*du-c[4]*(u0*dv+v0*du)-2*c[5]*v0*dv
 D=l[2]-c[0]-c[1]*u0-c[2]*v0-c[3]*u0*u0-c[4]*u0*v0-c[5]*v0*v0
 if abs(A)<1e-9:
  roots=[] if abs(B)<1e-10 else [-D/B]
 else:
  disc=B*B-4*A*D;roots=[] if disc<0 else [(-B-np.sqrt(disc))/(2*A),(-B+np.sqrt(disc))/(2*A)]
 roots=sorted([float(x) for x in roots if abs(x)<=search],key=abs)
 hull=np.array(patch['hull'])
 for gap in roots:
  uv=np.array([u0+gap*du,v0+gap*dv])
  if np.max(hull[:,:2]@uv+hull[:,2])>1e-8:continue
  derivative=np.array([c[1]+2*c[3]*uv[0]+c[4]*uv[1],c[2]+c[4]*uv[0]+2*c[5]*uv[1]])/r
  normal=E@np.r_[-derivative,1];normal/=np.linalg.norm(normal)
  if np.dot(normal,n)>=0:continue
  return gap,g+gap*n,normal
 return None
def sample_mesh(mesh,seed,n=512):
 rng=np.random.default_rng(seed);areas=mesh.area_faces
 ids=rng.choice(len(areas),n,p=areas/areas.sum());uv=rng.random((n,2));a=np.sqrt(uv[:,0]);b=uv[:,1]
 bary=np.column_stack([1-a,a*(1-b),a*b]);points=np.einsum('ni,nij->nj',bary,mesh.triangles[ids])
 return points,mesh.face_normals[ids],ids,bary
def main():
 cfg=json.loads((O/'config.json').read_text());cases=cfg['cases'];inputs=[F/'completed_locked.json',O/'config.json',R/'experiments/contact_preference/stage3_temporal_support_v3/src/run.py',R/'experiments/contact_preference/stage3_pilot_v1/src/build_supports_v2.py',R/'src/v3/stage2_robot_only_depth.py']
 for c in cases:
  k=c['key'];inputs.extend([F/'refined_cases'/f'{k}.npz',R/'experiments/region_pose/iter_024/train'/f'{k}.npz',R/'data/v3/trainval_rgbd_448'/c['asset']/(c['demo']+'.h5'),R/'experiments/region_pose/iter_003/evidence/motion'/f"{c['asset']}_{c['demo']}.npz"])
  for t in c['frames']:inputs.extend([B/'points_evidence'/f'{k}_frame{t}.npz',R/'experiments/region_pose/iter_024/train_loo'/f'{k}_frame{t}.npz'])
 lock={str(p):sha(p) for p in inputs};save(O/'input_locked.json',lock);save(O/'source_locked.json',{str(O/'src/fit_surfaces.py'):sha(O/'src/fit_surfaces.py')})
 save(O/'stage_state.json',{'time':now(),'status':'fitting'})
 robot=RobotDepth(448);summary=[]
 for ci,c in enumerate(cases):
  key=c['key'];src=dict(np.load(F/'refined_cases'/f'{key}.npz'));reg=dict(np.load(R/'experiments/region_pose/iter_024/train'/f'{key}.npz'))
  ix=src['supported_representatives'];cloud=src['common_xyz'][ix];tree=cKDTree(cloud);ref=src['reference_xyz'];rtree=cKDTree(ref);camera=reg['query_camera_to_world'][:3,3];patches={};rows=[]
  motion=dict(np.load(R/'experiments/region_pose/iter_003/evidence/motion'/f"{c['asset']}_{c['demo']}.npz"))
  with h5py.File(R/'data/v3/trainval_rgbd_448'/c['asset']/(c['demo']+'.h5'),'r') as h:
   for t in c['frames']:
    Q=reg[f'query_to_current_hand_{t}'];H=reg[f'hand_to_world_{t}'];W=H@Q
    z=dict(np.load(B/'points_evidence'/f'{key}_frame{t}.npz'));loo=np.load(R/'experiments/region_pose/iter_024/train_loo'/f'{key}_frame{t}.npz')['loo_Q']
    j=int(np.where(h['frame_indices'][:]==t)[0][0]);q=motion['measured_finger_q'][t]
    robot.render(q,np.linalg.solve(H,h['agentview/camera_to_world'][j]),h['agentview/K'][j])
    pp=[];nn=[];gg=[];faces=[];barys=[]
    for gid in cfg['active_geoms']:
     mesh,_=geo.surface(robot,[gid]);p,n,ids,bary=sample_mesh(mesh,2026+gid,cfg['samples_per_geom'])
     pp.append(p);nn.append(n);gg.extend([gid]*len(p));faces.extend(ids);barys.extend(bary)
    hand=np.concatenate(pp);normalhand=np.concatenate(nn);invQ=np.linalg.inv(Q);gref=apply(invQ,hand);nref=normalhand@invQ[:3,:3].T;nref/=np.linalg.norm(nref,axis=1)[:,None]
    distances,anchors=rtree.query(gref);N=len(hand)
    gap=np.full(N,np.nan);lo=gap.copy();hi=gap.copy();error=gap.copy();rms=gap.copy();spread=gap.copy();pts=np.full((N,3),np.nan);normals=pts.copy()
    status=np.full(N,3,np.int8);reason=np.full(N,1,np.int8);modelgaps=np.full((N,4),np.nan);sourcecount=np.zeros((N,2),int);patch_keys=[]
    for a in np.unique(anchors[distances<=cfg['local_search_m']]):
     if str(a) in patches:continue
     sub=[]
     for radius in cfg['fit_radii_m']:
      inds=np.array(tree.query_ball_point(ref[a],radius),int)
      if len(inds):
       inds=inds[np.argsort(np.sum((cloud[inds]-ref[a])**2,axis=1),kind='stable')][:cfg['max_patch_cells']]
      sub.append(fit_patch(ref[a],cloud,inds,radius,camera))
     patches[str(a)]=sub
    for k in np.flatnonzero(distances<=cfg['local_search_m']):
     a=int(anchors[k]);sub=patches[str(a)]
     if any(p is None for p in sub):reason[k]=2;continue
     if np.dot(np.array(sub[0]['basis'])[:,2],np.array(sub[1]['basis'])[:,2])<=0:reason[k]=3;continue
     results=[intersect(p,gref[k],nref[k],m,cfg['local_search_m']) for p in sub for m in p['models']]
     if any(v is None for v in results):reason[k]=4;continue
     mg=np.array([v[0] for v in results]);nom=results[-1];point=nom[1]
     pred=apply(Q,point[None])[0]
     shifted=np.einsum('nij,j->ni',loo[:,:3,:3],point)+loo[:,:3,3]
     lerr=float(np.linalg.norm(shifted-pred,axis=1).max())
     localerr=max(m['rms'] for p in sub for m in p['models'])
     ee=np.sqrt(max(float(z['sigma_proxy_m'][a]),lerr)**2+localerr**2)
     gap[k]=nom[0];lo[k]=mg.min()-ee;hi[k]=mg.max()+ee;error[k]=ee;rms[k]=localerr;spread[k]=mg.max()-mg.min();pts[k]=point;normals[k]=nom[2];modelgaps[k]=mg;sourcecount[k]=[len(p['indices']) for p in sub]
     status[k]=0 if lo[k]>0 else 2 if hi[k]<0 else 1;reason[k]=0
    world=apply(W,pts);valid=np.isfinite(pts[:,0]);vc=np.full((2,N),4,np.int8);visibility=np.full(N,4,np.int8)
    if valid.any():
     camlabels,combined,_=vismod.visibility(world[valid],H,q,h,j,robot);vc[:,valid]=camlabels;visibility[valid]=combined
    # Reject object-surface claims conflicting with observed free space or unidentifiable occlusion.
    conflict=valid&np.isin(visibility,[2,3,4]);status[conflict]=3;reason[conflict]=5
    output=dict(gripper_hand=hand,gripper_ref=gref,gripper_world=apply(H,hand),gripper_normal_hand=normalhand,gripper_geom=np.array(gg),gripper_face=np.array(faces),barycentric=np.array(barys),anchor=anchors,anchor_distance=distances,estimated_ref=pts,estimated_world=world,estimated_normal_ref=normals,gap_m=gap,gap_lower_m=lo,gap_upper_m=hi,uncertainty_m=error,fit_rms_m=rms,model_spread_m=spread,model_gaps_m=modelgaps,status=status,reason=reason,sourcecount=sourcecount,visibility=visibility,camera_visibility=vc,Q=Q,H=H)
    np.savez_compressed(O/'points'/f'{key}_frame{t}.npz',**output)
    row={'frame':t,'sampled_gripper_points':N,'surface_interpolations':int(valid.sum()),'status_counts':np.bincount(status,minlength=4).tolist(),'reason_counts':np.bincount(reason,minlength=6).tolist(),'robot_occluded_estimates':int((valid&(visibility==1)).sum()),'occluded_zero_gap_compatible_unknown':int(((status==1)&(visibility==1)).sum()),'occluded_separated':int(((status==0)&(visibility==1)).sum()),'gap_mm_quantiles':np.nanpercentile(gap,[0,25,50,75,100]).tolist() if valid.any() else [],'uncertainty_mm_quantiles':(np.nanpercentile(error,[0,50,100])*1000).tolist() if valid.any() else []}
    row['gap_mm_quantiles']=[v*1000 for v in row['gap_mm_quantiles']]
    rows.append(row);save(O/'summary_partial.json',summary+[dict(**c,rows=rows)])
  save(O/'patches'/f'case{ci}.json',patches)
  np.savez_compressed(O/'patches'/f'sources{ci}.npz',source_indices=ix,cloud_ref=cloud,query_xyz=ref,query_pixel=src['reference_pixel'],query_rgb=src['reference_rgb'])
  summary.append(dict(**c,rows=rows,patch_anchor_count=len(patches),valid_two_scale_patches=sum(all(p is not None for p in ps) for ps in patches.values())))
 robot.close()
 assert all(sha(p)==h for p,h in lock.items())
 save(O/'summary.json',{'time':now(),'cases':summary,'scoring_network_training':False,'contact_truth_labels':False,'interpolation_not_unobserved_completion':True})
 save(O/'stage_state.json',{'time':now(),'status':'computed_pending_audit'})
 print('surface_fit_complete')
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(O/'failure.json',{'time':now(),'type':type(e).__name__,'message':str(e),'stack':[(x.filename,x.lineno) for x in traceback.extract_tb(e.__traceback__)]})
  traceback.print_exc(file=(O/'failure.log').open('w'))
