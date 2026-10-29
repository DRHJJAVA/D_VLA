import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import sys,json,hashlib,importlib.util,traceback
import numpy as np,h5py,trimesh,mujoco
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/occluded_surface_fit_v1';D=R/'experiments/contact_preference/contact_location_5cam_v1/reliability'
sp=importlib.util.spec_from_file_location('surface_original',O/'src/fit_surfaces.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
m.F=R/'experiments/contact_preference/contact_location_5cam_v1/fusion'
from PIL import Image
def save(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False))
def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(np.bool_,)):return bool(x)
 if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
 return x
def stats(a):
 a=np.asarray(a);a=a[np.isfinite(a)]
 return {'n':len(a),'q05_q50_q95':np.percentile(a,[5,50,95]).tolist() if len(a) else []}
def unique_ids(x):
 _,ix=np.unique(np.floor(x/.001).astype(np.int64),axis=0,return_index=True);return np.sort(ix)
def render_group(robot,ids):
 groups=robot.model.geom_group.copy()
 try:
  robot.model.geom_group[:]=5;robot.model.geom_group[ids]=1
  robot.renderer.update_scene(robot.data,camera=robot.camera,scene_option=robot.option)
  return robot.renderer.render().copy()
 finally:robot.model.geom_group[:]=groups
def independent_mesh(robot,ids):
 parts=[]
 for gid in ids:
  mid=robot.model.geom_dataid[gid];assert int(robot.model.geom_type[gid])==7
  va=robot.model.mesh_vertadr[mid];vn=robot.model.mesh_vertnum[mid];fa=robot.model.mesh_faceadr[mid];fn=robot.model.mesh_facenum[mid]
  vertices=robot.model.mesh_vert[va:va+vn].astype(float)@robot.data.geom_xmat[gid].reshape(3,3).T+robot.data.geom_xpos[gid]
  faces=robot.model.mesh_face[fa:fa+fn].copy()
  parts.append(trimesh.Trimesh(vertices=vertices,faces=faces,process=False))
 return trimesh.util.concatenate(parts)
def project(T,p,K):
 c=m.apply(T,p);uv=c[:,:2]/c[:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
 return c,uv
def observation(points,H,views):
 result=[]
 for v in views:
  K=np.array(v['K']);T=np.array(v['hand_to_camera']);cp,uv=project(T,points,K)
  good=np.isfinite(uv).all(1)&(cp[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<448)&(uv[:,1]>=0)&(uv[:,1]<448)
  xy=np.rint(np.nan_to_num(uv,nan=-999)).astype(int);xy=np.clip(xy,0,447)
  depth=v['_depth'][xy[:,1],xy[:,0]];rd=v['_robot'][xy[:,1],xy[:,0]]
  residual=depth-cp[:,2];labels=np.full(len(points),4,int);good&=np.isfinite(depth)&(depth>0)
  labels[good&(abs(residual)<=.003)]=0
  labels[good&(residual<-.003)]=2;labels[good&(residual>.003)]=3
  robotfront=good&(rd>0)&(rd<cp[:,2]-.003)&(abs(depth-rd)<=.003)
  labels[robotfront]=1
  fd=v['_finger'][xy[:,1],xy[:,0]]
  active=robotfront&(abs(fd-rd)<1e-5)
  result.append({'uv':uv,'depth_residual_mm':residual*1000,'state':labels,'active_visual_finger_occluded':active,'robot_depth_residual_mm':(depth-rd)*1000})
 return result
def fit_variant(cloud,ref,camera,Q,H,loo,old,meshes,geomids):
 tree=cKDTree(cloud);rtree=cKDTree(ref);patches={};hand=[];norm=[];geom=[]
 for gid in geomids:
  p,n,_,_=m.sample_mesh(meshes[gid],2026+gid,512);hand.extend(p);norm.extend(n);geom.extend([gid]*len(p))
 hand=np.array(hand);norm=np.array(norm);invQ=np.linalg.inv(Q);gref=m.apply(invQ,hand);nref=norm@invQ[:3,:3].T;nref/=np.linalg.norm(nref,axis=1)[:,None]
 dist,anchor=rtree.query(gref);points=[];failure={'beyond_search':int((dist>.008).sum()),'patch_insufficient':0,'intersection_missing':0};maxr=0.
 for i in np.flatnonzero(dist<=.008):
  a=int(anchor[i])
  if a not in patches:
   patches[a]=[]
   for radius in [.003,.005]:
    ids=np.array(tree.query_ball_point(ref[a],radius),int)
    if len(ids):ids=ids[np.argsort(np.sum((cloud[ids]-ref[a])**2,axis=1),kind='stable')][:160]
    patches[a].append(m.fit_patch(ref[a],cloud,ids,radius,camera))
  ps=patches[a]
  if any(p is None for p in ps):failure['patch_insufficient']+=1;continue
  if np.dot(np.array(ps[0]['basis'])[:,2],np.array(ps[1]['basis'])[:,2])<=0:failure['intersection_missing']+=1;continue
  fits=[m.intersect(p,gref[i],nref[i],m0,.008) for p in ps for m0 in p['models']]
  if any(p is None for p in fits):failure['intersection_missing']+=1;continue
  mg=np.array([p[0] for p in fits]);point=fits[-1][1];ph=m.apply(Q,point[None])[0]
  shifted=np.einsum('nij,j->ni',loo[:,:3,:3],point)+loo[:,:3,3]
  spread=float(np.linalg.norm(shifted-ph,axis=1).max());rms=max(md['rms'] for p in ps for md in p['models'])
  base=max(float(old['sigma_base_m'][a]),spread);sigma=np.sqrt(base*base+rms*rms)
  low=float(mg.min()-sigma);high=float(mg.max()+sigma)
  points.append({'id':int(i),'anchor':a,'ref':point,'hand':ph,'gripper':hand[i],'normal':norm[i],'geom':int(geom[i]),'gap':float(mg[-1]*1000),'lower':low*1000,'upper':high*1000,'sigma':float(sigma*1000),'rms':float(rms*1000),'sourcecount':[len(p['indices']) for p in ps],'patch':m.apply(Q,cloud[np.array(ps[-1]['indices'])]),'status':'unknown','reason':'separation_supported' if low>0 else 'penetration_conflict' if high<0 else 'zero_compatible_not_confirmed','anchor_pixel':[]})
 return points,dict(failure,accepted=len(points),source_cells=len(cloud),sampled=len(hand))
def main():
 cfg=json.loads((D/'config.json').read_text());oldlocks=[O/'completed_locked.json',O/'review_fix_01/completed_locked.json']
 allold={}
 for p in oldlocks:allold.update(json.loads(p.read_text()))
 assert all(m.sha(p)==v for p,v in allold.items())
 inputs=[D/'config.json',O/'src/fit_surfaces.py']+oldlocks
 for c in cfg['cases']:
  k=c['key'];inputs += [m.F/'refined_cases'/f'{k}.npz',R/'experiments/region_pose/iter_024/train'/f'{k}.npz',R/'experiments/contact_preference/contact_location_5cam_v1/rgbd'/c['asset']/(c['demo']+'.h5')]
  for t in c['frames']:inputs += [m.B/'points_evidence'/f'{k}_frame{t}.npz',R/'experiments/region_pose/iter_024/train_loo'/f'{k}_frame{t}.npz',O/'points'/f'{k}_frame{t}.npz']
 locks={str(p):m.sha(p) for p in inputs};save(D/'input_locked.json',locks)
 save(D/'source_locked.json',{str(D/'src/run.py'):m.sha(D/'src/run.py')})
 save(D/'stage_state.json',{'time':m.now(),'status':'running'})
 robot=m.RobotDepth(448);rows=[];cases=[];maxmesh=0.;maxroundtrip=0.
 for ci,c in enumerate(cfg['cases']):
  k=c['key'];reg=np.load(R/'experiments/region_pose/iter_024/train'/f'{k}.npz');src=np.load(m.F/'refined_cases'/f'{k}.npz')
  ref=reg['query_xyz'];qpix=reg['query_pixel'];camera=reg['query_camera_to_world'][:3,3]
  oldweb=json.loads((O/'review_fix_01/webdata'/f'case{ci}.json').read_text());web={'camera_names':cfg['cameras'],'title':c['title'],'key':k,'query':oldweb['query'],'query_image':oldweb['query_image'],'frames':[]}
  querycloud=ref[unique_ids(ref)]
  # fixed actual query points, selected by old candidate anchors only for review scope
  anchors=sorted(set(int(a) for t in c['frames'] for a in np.load(O/'points'/f'{k}_frame{t}.npz')['anchor'][np.isfinite(np.load(O/'points'/f'{k}_frame{t}.npz')['gap_m'])]))
  traces=[{'anchor':a,'pixel':qpix[a].tolist(),'ref':ref[a].tolist(),'frames':[]} for a in anchors]
  with h5py.File(R/'experiments/contact_preference/contact_location_5cam_v1/rgbd'/c['asset']/(c['demo']+'.h5'),'r') as h:
   for fi,t in enumerate(c['frames']):
    print(json.dumps({'case':ci,'frame':t,'phase':'fit'}),flush=True)
    Q=reg[f'query_to_current_hand_{t}'];H=reg[f'hand_to_world_{t}'];q=reg[f'finger_q_{t}']
    loo=np.load(R/'experiments/region_pose/iter_024/train_loo'/f'{k}_frame{t}.npz')['loo_Q'];old=np.load(m.B/'points_evidence'/f'{k}_frame{t}.npz')
    j=int(np.where(h['frame_indices'][:]==t)[0][0]);fr=oldweb['frames'][fi];views=[];geometry=[]
    fullids=src['supported_representatives'];drawids=fullids[np.linspace(0,len(fullids)-1,min(5000,len(fullids)),dtype=int)]
    fr['object']=m.apply(Q,src['common_xyz'][drawids]).tolist();rgb=src['rgb'][drawids];rgb=rgb*255 if rgb.max()<=1 else rgb
    fr['object_rgb']=rgb.astype(int).tolist()
    for vi,cn in enumerate(cfg['cameras']):
     C=h[cn+'/camera_to_world'][j];K=h[cn+'/K'][j];depth=h[cn+'/depth_m'][j];rd=robot.render(q,np.linalg.solve(H,C),K)
     vd=render_group(robot,[2,5]);cd=render_group(robot,[3,4,6,7])
     common=binary_erosion((vd>0)&(vd<1)&(cd>0)&(cd<1),iterations=2)
     # Every common projected interior pixel is reported; occluding object pixels are not declared robot truth.
     geometry.append({'camera':cn,'common_interior_pixels':int(common.sum()),'visual_scene_depth_residual_mm':stats((depth-vd)[common]*1000),'collision_scene_depth_residual_mm':stats((depth-cd)[common]*1000),'visual_within3mm':int((common&(abs(depth-vd)<=.003)).sum()),'collision_within3mm':int((common&(abs(depth-cd)<=.003)).sum()),'pixel_identity':'scene depth only; object may occlude robot, no segmentation truth'})
     imagepath=D/'images'/f'c{ci}_f{t}_v{vi}.jpg';Image.fromarray(h[cn+'/rgb'][j]).save(imagepath,quality=94)
     v={'image':f'/图示/五相机可靠性/images/c{ci}_f{t}_v{vi}.jpg','K':K.tolist(),'C':C.tolist(),'name':cn};v['_depth']=depth;v['_robot']=rd;v['_finger']=vd;v['hand_to_camera']=np.linalg.solve(C,H).tolist();views.append(v)
    visual,_=m.geo.surface(robot,[2,5]);ind=independent_mesh(robot,[2,5]);err=max(cKDTree(ind.vertices).query(visual.vertices)[0].max(),cKDTree(visual.vertices).query(ind.vertices)[0].max());maxmesh=max(maxmesh,float(err));assert err<1e-8
    meshes={gid:m.geo.surface(robot,[gid])[0] for gid in [2,3,4,5,6,7]}
    sid=np.flatnonzero(src['supported']&(src['frame']!=t));sid=sid[unique_ids(src['common_xyz'][sid])];cloud=src['common_xyz'][sid]
    assert not np.any(src['frame'][sid]==t)
    np.savez_compressed(D/'data'/f'c{ci}_f{t}_sources.npz',source_ids=sid,source_frames=src['frame'][sid],cloud=cloud,query_cloud=querycloud)
    variants={};vr={}
    for name in cfg['variants']:
     pool=querycloud if name.endswith('query_only') else cloud;ids=[2,5] if name.startswith('visual') else [3,4,6,7]
     pts,report=fit_variant(pool,ref,camera,Q,H,loo,old,meshes,ids)
     if pts:
      pos=np.array([p['hand'] for p in pts]);dv,pv=m.geo.distance(visual,pos);cm,_=m.geo.surface(robot,[3,4,6,7]);dc,pc=m.geo.distance(cm,pos);obs=observation(pos,H,views)
      for i,p in enumerate(pts):
       p['anchor_pixel']=qpix[p['anchor']].tolist();p['visual_distance_mm']=float(dv[i]*1000);p['collision_distance_mm']=float(dc[i]*1000)
       p['observations']=[{kk:vv[i] for kk,vv in o.items()} for o in obs]
       p['robot_occluded']=any(int(o['state'][i])==1 for o in obs)
       p['interval_hand']=[p['gripper']+val*.001*p['normal'] for val in [p['lower'],p['upper']]]
      report['observation_states']=[np.bincount(o['state'],minlength=5).tolist() for o in obs]
      report['robot_occluded_estimates']=sum(p['robot_occluded'] for p in pts)
      report['active_visual_finger_occluded']=sum(any(o['active_visual_finger_occluded'] for o in p['observations']) for p in pts)
      report['zero_compatible_unknown']=sum(p['reason']=='zero_compatible_not_confirmed' for p in pts)
     else:report.update(observation_states=[[0]*5,[0]*5],robot_occluded_estimates=0,zero_compatible_unknown=0,active_visual_finger_occluded=0)
     variants[name]=clean(pts);vr[name]=report
    tracehand=m.apply(Q,ref[anchors]);traceobs=observation(tracehand,H,views)
    for i,tr in enumerate(traces):
     tr['frames'].append(clean({'frame':t,'hand':tracehand[i],'observations':[{kk:vv[i] for kk,vv in o.items()} for o in traceobs]}))
    # Old highlighted estimates are checked against scene depth without relocating them.
    oldpoints=fr['points'];oldobs=observation(np.array([p['hand'] for p in oldpoints]),H,views)
    for i,p in enumerate(oldpoints):p['observations']=clean([{kk:vv[i] for kk,vv in o.items()} for o in oldobs])
    for v in views:v.pop('_depth');v.pop('_robot');v.pop('_finger')
    wf={'frame':t,'views':views,'old_points':oldpoints,'variants':variants,'reports':vr,'geometry':geometry,'object':fr['object'],'object_rgb':fr['object_rgb'],'visual_vertices':visual.vertices.tolist(),'visual_faces':visual.faces.tolist(),'gripper_vertices':fr['gripper_vertices'],'gripper_faces':fr['gripper_faces']}
    web['frames'].append(wf);row={'case':c['title'],'frame':t,'variants':vr,'geometry':geometry};rows.append(row)
    save(D/'summary_partial.json',clean(rows));save(D/'stage_state.json',{'time':m.now(),'status':'running','case':ci,'frame':t})
  cycles=[]
  for tr in traces:
   for vi in range(len(cfg['cameras'])):
    occluded=[f for f in tr['frames'] if f['observations'][vi]['active_visual_finger_occluded']]
    for first in occluded:
     later=[f for f in tr['frames'] if f['frame']>first['frame'] and f['observations'][vi]['state']==0]
     if later:
      cycles.append({'anchor':tr['anchor'],'camera':vi,'occluded_frame':first['frame'],'reappeared_frame':later[0]['frame'],'residual_mm':later[0]['observations'][vi]['depth_residual_mm'],'interpretation':'depth compatible only; pixel identity not proven'});break
  web['traces']=traces;web['cycles']=cycles;web['cycle_definition']='active visual finger occlusion then observed depth compatible; not physical identity truth'
  save(D/'webdata'/f'case{ci}.json',clean(web));cases.append({'case':c['title'],'fixed_anchors':len(anchors),'depth_compatible_occlusion_reappearance':len(cycles),'cycles':cycles})
 robot.close()
 assert all(m.sha(p)==h for p,h in locks.items());assert all(m.sha(p)==h for p,h in allold.items())
 result={'time':m.now(),'rows':rows,'cases':cases,'independent_visual_mesh_max_m':maxmesh,'tested_frame_sources_excluded':True,'old_locked_files_unchanged':len(allold),'no_scoring_training':True,'physical_contact_confirmed':False,'limitations':cfg['holdout_limit']}
 save(D/'summary.json',clean(result));save(D/'stage_state.json',{'time':m.now(),'status':'computed_pending_review'})
 print('reliability_computed',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(D/'failure.json',{'time':m.now(),'type':type(e).__name__,'message':str(e),'stack':[(x.filename,x.lineno) for x in traceback.extract_tb(e.__traceback__)]})
  traceback.print_exc(file=(D/'failure.log').open('w'));save(D/'stage_state.json',{'time':m.now(),'status':'failed','type':type(e).__name__})
