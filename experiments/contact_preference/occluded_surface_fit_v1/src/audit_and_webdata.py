from pathlib import Path
import os,sys,json,hashlib,importlib.util,traceback
os.environ['MUJOCO_GL']='egl'
import numpy as np,h5py
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/occluded_surface_fit_v1';F=R/'experiments/contact_preference/multiview_temporal_fusion_v1';B=R/'experiments/contact_preference/stage3_pilot_v2'
sys.path.insert(0,str(R))
sp=importlib.util.spec_from_file_location('fit',O/'src/fit_surfaces.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def arr(x):return np.asarray(x).tolist()
def project(P,C,K):
 x=m.apply(np.linalg.inv(C),P);z=-x[:,2];uv=np.column_stack([K[0,0]*x[:,0]/z+K[0,2],K[1,2]-K[1,1]*x[:,1]/z])
 return uv,z
def ray_hit(origin,points,triangles):
 # Independent Moller-Trumbore first forward intersection, in world metres.
 out=np.full(len(points),np.inf);V0=triangles[:,0];E1=triangles[:,1]-V0;E2=triangles[:,2]-V0;S=origin-V0
 for i,p in enumerate(points):
  ray=p-origin;ray/=np.linalg.norm(ray);h=np.cross(np.broadcast_to(ray,E2.shape),E2);a=np.einsum('ij,ij->i',E1,h);ok=abs(a)>1e-12;inv=np.zeros(len(a));inv[ok]=1/a[ok]
  u=inv*np.einsum('ij,ij->i',S,h);q=np.cross(S,E1);v=inv*(q@ray);dist=inv*np.einsum('ij,ij->i',E2,q)
  good=ok&(u>=0)&(v>=0)&(u+v<=1)&(dist>0)
  if good.any():out[i]=dist[good].min()
 return out
def main():
 cfg=json.loads((O/'config.json').read_text());summary=json.loads((O/'summary.json').read_text());audit={'time':m.now(),'checks':{},'cases':[]}
 for name,path in [('inputs',O/'input_locked.json'),('source',O/'source_locked.json'),('fusion',F/'completed_locked.json'),('baseline',B/'completed_locked.json'),('temporal',R/'experiments/contact_preference/stage3_temporal_support_v3/completed_locked.json')]:
  lock=json.loads(path.read_text());bad=[p for p,h in lock.items() if m.sha(p)!=h];assert not bad,name
  audit['checks'][name]={'files':len(lock),'changed':0}
 robot=m.RobotDepth(448);maxeq=0.;maxco=0.;allrows=[]
 # Independent synthetic signed-gap tests: plane coefficients represent object normal -z.
 patch={'center':[0,0,0],'basis':np.diag([1,-1,-1]).tolist(),'radius':.01,'hull':[[1,0,-1],[-1,0,-1],[0,1,-1],[0,-1,-1]]}
 for gap in [-.002,0,.002]:
  model={'coef':[-gap,0,0,0,0,0]}
  hit=m.intersect(patch,np.zeros(3),np.array([0,0,1.]),model)
  assert hit is not None and abs(hit[0]-gap)<1e-12
 assert m.intersect(patch,np.array([.02,0,0]),np.array([0,0,1.]),{'coef':[-.002,0,0,0,0,0]}) is None
 audit['checks']['synthetic_signed_gap_and_outside_hull']=True
 for ci,c in enumerate(cfg['cases']):
  key=c['key'];patches=json.loads((O/'patches'/f'case{ci}.json').read_text());source=dict(np.load(O/'patches'/f'sources{ci}.npz'));cloud=source['cloud_ref'];ref=source['query_xyz'];holdout=[]
  reg=dict(np.load(R/'experiments/region_pose/iter_024/train'/f'{key}.npz'))
  # Independent IRLS coefficient and tangent-normal residual reconstruction.
  for ps in patches.values():
   if not all(p is not None for p in ps):continue
   for p in ps:
    E=np.array(p['basis']);local=(cloud[p['indices']]-p['center'])@E;uv=local[:,:2]/p['radius'];u,v=uv.T
    for md in p['models']:
     A=np.column_stack([np.ones(len(u)),u,v,u*u,u*v,v*v]) if md['quadratic'] else np.column_stack([np.ones(len(u)),u,v])
     y=local[:,2];co=np.linalg.lstsq(A,y,rcond=None)[0]
     for _ in range(3):
      e=y-A@co;s=max(1.4826*np.median(abs(e-np.median(e))),1e-7);w=np.minimum(1,1.345*s/np.maximum(abs(e),1e-12));co=np.linalg.lstsq(A*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
     maxco=max(maxco,float(np.max(abs(co-np.array(md['coef'])[:len(co)]))))
     # Deterministic every-fifth-source holdout is a surface-shape diagnostic, not contact truth.
     test=np.arange(len(y))%5==0;train=~test
     if train.sum()>=len(co)+2 and np.linalg.cond(A[train])<=1e6:
      fit=np.linalg.lstsq(A[train],y[train],rcond=None)[0];holdout.extend(abs(y[test]-A[test]@fit).tolist())
  oldweb=json.loads((F/'webdata_v2'/f'case{ci}.json').read_text());web={'title':c['title'],'key':key,'query':int(key.split('query')[-1]),'frames':[]};rows=[]
  with h5py.File(R/'data/v3/trainval_rgbd_448'/c['asset']/(c['demo']+'.h5'),'r') as h:
   for row in summary['cases'][ci]['rows']:
    t=row['frame'];z=dict(np.load(O/'points'/f'{key}_frame{t}.npz'));base=dict(np.load(B/'points_evidence'/f'{key}_frame{t}.npz'));assert np.array_equal(ref,base['query_xyz'])
    valid=np.flatnonzero(np.isfinite(z['gap_m']));Q=z['Q'];H=z['H'];W=H@Q;j=int(np.where(h['frame_indices'][:]==t)[0][0]);q=reg[f'finger_q_{t}']
    robot.render(q,np.linalg.solve(H,h['agentview/camera_to_world'][j]),h['agentview/K'][j])
    meshes=[m.geo.surface(robot,[gid])[0] for gid in cfg['active_geoms']]
    triangles=np.concatenate([m.apply(H,mesh.triangles.reshape(-1,3)).reshape(-1,3,3) for mesh in meshes])
    # Ray-equation, all-model hull, fit, normal opposition and uncertainty audit.
    for idx in valid:
     a=int(z['anchor'][idx]);ps=patches[str(a)];n=z['gripper_normal_hand'][idx]@np.linalg.inv(Q)[:3,:3].T;n/=np.linalg.norm(n);g=z['gripper_ref'][idx]
     for mi,(p,md) in enumerate((p,md) for p in ps for md in p['models']):
      d=z['model_gaps_m'][idx,mi];pt=g+d*n;loc=(pt-p['center'])@np.array(p['basis']);u,v=loc[:2]/p['radius'];co=np.array(md['coef']);calc=np.dot(co,[1,u,v,u*u,u*v,v*v]);maxeq=max(maxeq,abs(loc[2]-calc))
      hull=np.array(p['hull']);assert np.max(hull[:,:2]@[u,v]+hull[:,2])<=1e-7
     assert np.linalg.norm(z['estimated_ref'][idx]-(g+z['gap_m'][idx]*n))<1e-9
    assert np.allclose(m.apply(W,z['estimated_ref'][valid]),z['estimated_world'][valid])
    expected=np.where(z['gap_lower_m']>0,0,np.where(z['gap_upper_m']<0,2,1));expected[z['reason']!=0]=3;assert np.array_equal(expected,z['status'])
    views=[];activeOcclusion=np.zeros((2,len(valid)),bool);visibleErrors=[]
    sourceWorld=m.apply(W,cloud);sourceVisibility=[]
    for cami,cam in enumerate(['agentview','robot0_eye_in_hand']):
     C=h[cam+'/camera_to_world'][j];K=h[cam+'/K'][j];dep=h[cam+'/depth_m'][j];rd=robot.render(q,np.linalg.solve(H,C),K)
     pu,zz=project(z['estimated_world'][valid],C,K);hit=ray_hit(C[:3,3],z['estimated_world'][valid],triangles)
     raylen=np.linalg.norm(z['estimated_world'][valid]-C[:3,3],axis=1)
     activeOcclusion[cami]=(hit<raylen-.002)&(z['camera_visibility'][cami,valid]==1)
     suv,sz=project(sourceWorld,C,K);ij=np.rint(suv).astype(int);inside=(sz>0)&(ij[:,0]>=0)&(ij[:,0]<448)&(ij[:,1]>=0)&(ij[:,1]<448);ids=np.flatnonzero(inside);ix,iy=ij[ids].T
     obs=dep[iy,ix];rob=rd[iy,ix];notRobot=(obs>0)&((rob>1)|(obs<rob-.002))
     residual=sz[ids]-obs
     # Every projected source with valid nonrobot depth; report behind/foreground separately without tuning.
     dv=residual[notRobot];near=abs(dv)<=.003
     sourceVisibility.append({'camera':cam,'nonrobot_projection_count':len(dv),'depth_consistent_3mm':int(near.sum()),'source_front_of_depth_3mm':int((dv<-.003).sum()),'source_behind_depth_3mm':int((dv>.003).sum()),'depth_consistent_abs_residual_mm_median':float(np.median(abs(dv[near]))*1000) if near.any() else None})
     oldUV,oldZ=project(base['mapped_world_xyz'],C,K);views.append({'image':oldweb['images'][f'{t}_{cami}'],'K':arr(K),'C':arr(C),'estimate_uv':arr(pu),'old_uv':arr(oldUV),'old_infront':arr(oldZ>0),'name':'固定主相机' if cami==0 else '腕部相机'})
    active=np.any(activeOcclusion,axis=0)&(z['visibility'][valid]==1)
    np.savez_compressed(O/'audits'/f'{key}_frame{t}_active_occlusion.npz',valid_indices=valid,per_camera=activeOcclusion,active_gripper_occluded=active)
    observed=cloud[np.linspace(0,len(cloud)-1,min(3500,len(cloud)),dtype=int)]
    objectHand=m.apply(Q,observed)
    # Serialize no simulated object truth, only real source cloud and known robot mesh.
    vv=[];ff=[];offset=0
    for mesh in meshes:vv.extend(arr(mesh.vertices));ff.extend(arr(mesh.faces+offset));offset+=len(mesh.vertices)
    points=[]
    for vi,idx in enumerate(valid):
     p5=patches[str(int(z['anchor'][idx]))][1];localSource=m.apply(Q,cloud[p5['indices']])
     points.append({'id':int(idx),'hand':arr(m.apply(Q,z['estimated_ref'][idx:idx+1])[0]),'gripper':arr(z['gripper_hand'][idx]),'normal':arr(z['gripper_normal_hand'][idx]),'status':int(z['status'][idx]),'visibility':int(z['visibility'][idx]),'reason':int(z['reason'][idx]),'gap':float(z['gap_m'][idx]*1000),'lower':float(z['gap_lower_m'][idx]*1000),'upper':float(z['gap_upper_m'][idx]*1000),'sigma':float(z['uncertainty_m'][idx]*1000),'rms':float(z['fit_rms_m'][idx]*1000),'modelspread':float(z['model_spread_m'][idx]*1000),'sourcecount':arr(z['sourcecount'][idx]),'active_occluded':bool(active[vi]),'patch':arr(localSource),'anchor_pixel':arr(source['query_pixel'][z['anchor'][idx]])})
    fr={'frame':t,'summary':row,'points':points,'views':views,'object':arr(objectHand),'gripper_vertices':vv,'gripper_faces':ff,'old_class':arr(base['display_class']),'old_positive':arr(base['positive']),'active_occluded_count':int(active.sum()),'source_depth_diagnostic':sourceVisibility}
    web['frames'].append(fr);rows.append({'frame':t,'all_robot_occluded':row['robot_occluded_estimates'],'active_gripper_occluded':int(active.sum()),'source_depth_diagnostic':sourceVisibility})
   query=web['query'];jq=int(np.where(h['frame_indices'][:]==query)[0][0]);web['query_image']=oldweb['images'][f'{query}_0']
  save(O/'webdata'/f'case{ci}.json',web)
  audit['cases'].append({'key':key,'heldout_patch_abs_residual_mm_quantiles':arr(np.percentile(holdout,[50,95,100])*1000) if holdout else [],'rows':rows})
 robot.close();assert maxco<1e-10 and maxeq<1e-7
 audit['checks'].update({'independent_coefficient_max_difference':maxco,'independent_ray_surface_equation_max_m':maxeq,'baseline_query_arrays_equal':True,'signed_gap_labels_independent':True,'model_training':False,'physical_contact_truth':False})
 save(O/'audits/surface_audit.json',audit);print(json.dumps({'audit':'passed','active_occluded_by_case':[[r['active_gripper_occluded'] for r in c['rows']] for c in audit['cases']],'max_equation_m':maxeq}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(O/'audit_failure.json',{'type':type(e).__name__,'message':str(e),'stack':[(x.filename,x.lineno) for x in traceback.extract_tb(e.__traceback__)]});traceback.print_exc(file=(O/'audit_failure.log').open('w'))
