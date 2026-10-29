from pathlib import Path
import os,sys,json,hashlib,ast,traceback,time
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,h5py
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA'); I=R/'experiments/region_pose/iter_036'; E=R/'experiments/region_pose'
sys.path.insert(0,str(R/'src/v3'))
from stage2_probe_motion_fingerpads import cloud
from stage2_robot_only_depth import RobotDepth
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def check(lock):
 d=json.loads(lock.read_text());bad=[]
 for k,v in d.items():
  p=Path(k);p=p if p.is_absolute() else R/p
  if not p.exists() or sha(p)!=v:bad.append(k)
 assert not bad, 'locked input mismatch'
 return len(d)
def stat(x):
 x=np.asarray(x);return {'n':int(x.size),'median_mm':float(np.median(x)*1000) if x.size else None,'p95_mm':float(np.percentile(x,95)*1000) if x.size else None,'max_mm':float(x.max()*1000) if x.size else None,'fraction_le2mm':float((x<=.002).mean()) if x.size else None}
def independently_fit(a,b):
 ca=np.mean(a,axis=0);cb=np.mean(b,axis=0);u,s,v=np.linalg.svd((a-ca).T@(b-cb));D=np.eye(3);D[-1,-1]=np.linalg.det(v.T@u.T);rot=v.T@D@u.T;w=np.eye(4);w[:3,:3]=rot;w[:3,3]=cb-rot@ca;return w
def main():
 pre={'source':check(I/'source_locked.json'),'input':check(I/'input_locked.json'),'old033':check(E/'iter_033/completed_locked.json'),'old035':check(E/'iter_035/completed_locked.json')}
 code=ast.parse((E/'iter_024/src/audit_regions.py').read_text())
 funcs=[n for n in code.body if isinstance(n,ast.FunctionDef) and n.name in ['apply','rigid','in_region','visible']]
 env={'np':np};exec(compile(ast.Module(body=funcs,type_ignores=[]),'<frozen024helpers>','exec'),env)
 apply,rigid,in_region,visible=[env[k] for k in ['apply','rigid','in_region','visible']]
 render=next(n for n in ast.walk(code) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='render' and n.lineno==144)
 rendercode=compile(ast.Expression(body=render),'<frozen024render>','eval')
 asthash={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in funcs}
 stem='white_yellow_mug_demo_6_attempt1_iter035_query36'
 cr=np.load(E/'iter_035/train/correspondences'/ (stem+'.npz'))
 diag=json.loads((E/'iter_035/train/audits'/(stem+'.json')).read_text())
 seed=np.load(E/'iter_035/seeds/white_yellow_mug_demo_6.npz')
 z=np.load(E/'iter_033/cases/white_yellow_mug_demo_6.npz')
 query=cr['query_foreground_xyz'];H=z['hand_to_world'];finger=z['measured_finger_q'];frames=[55,57,60]
 assert diag['query']==36 and all(next(b for b in diag['bridge_results'] if b['frame']==t)['valid'] for t in frames)
 refs=json.loads((E/'iter_035/refs.json').read_text())['white_yellow_mug_demo_6']
 component=np.load(refs['component'])
 tree=cKDTree(query); allrows=[]; raw={k:[] for k in ['source_xyz','source_hand_xyz','mapped_query_xyz','pixel','camera','frame','source_index_033','rgb','source_depth','distance_to_query','query_nearest_index','region_loo_max_shift']}
 arrays={'query_xyz':query}; slices={}; numeric=[]
 with h5py.File(refs['rgbd'],'r') as f:
  def read(g,k,t):
   a=g[k][t];slices[g.name+'/'+k+'/'+str(t)]=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest();return a
  robot=RobotDepth(448)
  # The source cloud is independently rebuilt in the original source pixel order.
  for t in frames:
   Q=np.array(next(b for b in diag['bridge_results'] if b['frame']==t)['query_world_to_anchor_hand'])
   W=H[t]@Q;Wi=np.linalg.inv(W)
   assert np.array_equal(cr['hand_to_world_'+str(t)],H[t])
   ins=cr['inlier_'+str(t)].astype(bool);ids=cr['track_id_'+str(t)][ins];a=cr['query_xyz_'+str(t)][ins];b=cr['bridge_xyz_'+str(t)][ins]
   assert np.max(np.abs(rigid(a,b)-W))<1e-10
   start,end=map(int,z['frame_offsets'][t:t+2]);si=np.arange(start,end)
   src=z['points_world'][si];mapped=apply(src,Wi)
   assert in_region(apply(src,np.linalg.inv(H[t])),finger[t]).all()
   fits=[];deleted=[];reject=[];scaleall=[];diff=[]
   for j in range(len(a)):
    aa=np.delete(a,j,axis=0);bb=np.delete(b,j,axis=0)
    if len(aa)<7:reject.append({'track_id':int(ids[j]),'reason':'remaining_lt7'});continue
    scale=np.linalg.svd(aa-aa.mean(0),compute_uv=False)/np.sqrt(len(aa))
    if scale[1]<.003:reject.append({'track_id':int(ids[j]),'reason':'second_axis_lt3mm'});continue
    walt=rigid(aa,bb);ind=independently_fit(aa,bb);diff.append(float(np.max(np.abs(walt-ind))))
    fits.append(np.linalg.inv(H[t])@walt);deleted.append(int(ids[j]));scaleall.append(scale)
   assert len(fits)>=7,'insufficient eligible leave-one fits'
   fits=np.asarray(fits);deleted=np.asarray(deleted,dtype=np.int64)
   qdelta=np.array([np.linalg.norm(apply(query,qa)-apply(query,Q),axis=1) for qa in fits])
   rdelta=np.array([np.linalg.norm(apply(src,np.linalg.inv(H[t]@qa))-mapped,axis=1) for qa in fits])
   qmax=qdelta.max(0);rmax=rdelta.max(0)
   dist,near=tree.query(mapped);qvis=np.zeros(len(query),bool);camcounts=[];source_rgb=np.zeros((len(si),3),np.uint8);source_depth=np.zeros(len(si))
   for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
    g=f[cam];dep=read(g,'depth_m',t);K=read(g,'K',t);T=read(g,'camera_to_world',t);rgb=read(g,'rgb',t)
    rd=eval(rendercode,{'np':np,'robot':robot,'finger':finger,'t':t,'T':T,'H':H,'K':K})
    world=cloud(dep,K,T);hand=apply(world,np.linalg.inv(H[t]))
    valid=np.isfinite(dep).ravel()&(dep.ravel()>0)&in_region(hand,finger[t])
    full=valid&((rd.ravel()>1)|(dep.ravel()<rd.ravel()-.002))
    strict=valid&(dep.ravel()<rd.ravel()-.002)
    pix=np.column_stack(np.unravel_index(np.flatnonzero(full),dep.shape))[:,::-1]
    local=np.flatnonzero(z['camera'][si]==ci);sp=z['pixel'][si[local]]
    assert np.array_equal(pix,sp),'source pixels differ'
    assert np.array_equal(world[np.flatnonzero(full)],src[local]),'source XYZ differ'
    assert np.array_equal(full,strict),'strict filter membership differs; stop'
    source_rgb[local]=rgb[sp[:,1],sp[:,0]];source_depth[local]=dep[sp[:,1],sp[:,0]]
    qvis|=visible(apply(query,W),dep,rd,K,T)
    camcounts.append({'camera':cam,'source_pixels':len(local),'strict_equal':True})
   arrays.update({f'Q_{t}':Q,f'H_{t}':H[t],f'W_{t}':W,f'finger_{t}':finger[t],f'query_visible_{t}':qvis})
   np.savez_compressed(I/'loo'/f'frame_{t}.npz',Q=Q,H=H[t],fits=fits,deleted_track_id=deleted,deleted_is_grid=seed['source_is_grid'][deleted],remaining_scale=np.asarray(scaleall),query_max_shift=qmax,region_max_shift=rmax,query_delta=qdelta,region_delta=rdelta,source_index_033=si,inlier_track_id=ids,inlier_query_xyz=a,inlier_endpoint_xyz=b)
   vals={'source_xyz':src,'source_hand_xyz':apply(src,np.linalg.inv(H[t])),'mapped_query_xyz':mapped,'pixel':z['pixel'][si],'camera':z['camera'][si],'frame':np.full(len(si),t),'source_index_033':si,'rgb':source_rgb,'source_depth':source_depth,'distance_to_query':dist,'query_nearest_index':near,'region_loo_max_shift':rmax}
   for k,v in vals.items():raw[k].append(v)
   groups={}
   for label,grid in [('original',False),('grid',True)]:
    m=seed['source_is_grid'][deleted]==grid
    groups[label]={'fits':int(m.sum()),'query':stat(qdelta[m].max(0)) if m.any() else stat([]),'region':stat(rdelta[m].max(0)) if m.any() else stat([])}
   allrows.append({'frame':t,'inliers':len(a),'valid_loo':len(fits),'rejected':reject,'source_points':len(si),'cameras':camcounts,'query_shift':stat(qmax),'region_shift':stat(rmax),'source_distance':stat(dist),'near3mm':int((dist<=.003).sum()),'unknown_gt6mm':int((dist>.006).sum()),'query_visible':int(qvis.sum()),'deletion_groups':groups})
   numeric.append({'frame':t,'independent_rigid_max_abs':max(diff),'inverse_roundtrip_m':float(np.linalg.norm(apply(mapped,W)-src,axis=1).max())})
  # Current frame36 query pixel set: saved causal mask, actual measured depth, no targets.
  g=f['agentview'];dep=read(g,'depth_m',36);K=read(g,'K',36);T=read(g,'camera_to_world',36)
  pix=np.column_stack(np.unravel_index(np.flatnonzero(seed['mask']),dep.shape))[:,::-1]
  recon=cloud(dep,K,T)[np.flatnonzero(seed['mask'])]
  assert np.array_equal(recon,query),'current query XYZ mismatch'
  arrays['query_pixel']=pix
 for k,v in raw.items():arrays[k]=np.concatenate(v,axis=0)
 mapped=arrays['mapped_query_xyz'];fids=arrays['frame'];near=arrays['distance_to_query']<=.003
 support=np.zeros(len(mapped),np.int32)
 for t in frames:
  ids=np.flatnonzero((fids==t)&near)
  if len(ids):support+=(cKDTree(mapped[ids]).query(mapped)[0]<=.002)
 supported=near&(support>=2);ids=np.flatnonzero(supported)
 if len(ids):
  _,u=np.unique(np.floor(mapped[ids]/.001).astype(np.int64),axis=0,return_index=True);representatives=ids[u]
 else:representatives=np.array([],np.int64)
 arrays.update({'cross_time_support':support,'supported':supported,'representative_indices':representatives})
 # Independent brute force cross-frame support and representative checks.
 brute=np.array([sum(any(np.linalg.norm(mapped[j]-p)<=.002 for j in np.flatnonzero((fids==t)&near)) for t in frames) for p in mapped])
 assert np.array_equal(brute,support)
 assert len(np.unique(np.floor(mapped[representatives]/.001).astype(np.int64),axis=0))==len(representatives)
 assert np.all(supported[representatives])
 np.savez_compressed(I/'regions.npz',**arrays)
 save(I/'frames.json',allrows);save(I/'slice_hashes.json',slices)
 vis=np.stack([arrays[f'query_visible_{t}'] for t in frames])
 summary={'episode':'white_yellow_mug_demo_6','query':36,'frames':frames,'independent_episodes':1,'raw_source_observations':len(mapped),'near_query_3mm':int(near.sum()),'unknown_gt6mm':int((arrays['distance_to_query']>.006).sum()),'cross_time_observations':int(supported.sum()),'supported_real_1mm_representatives':len(representatives),'query_points':len(query),'query_visible_any':int(vis.any(0).sum()),'query_visible_two':int((vis.sum(0)>=2).sum()),'valid_loo_total':sum(x['valid_loo'] for x in allrows),'per_frame':allrows,'labels_created':False,'physical_identity_verified':False,'network_training':False,'validation_test_access':False}
 save(I/'summary.json',summary)
 post={'source':check(I/'source_locked.json'),'input':check(I/'input_locked.json'),'old033':check(E/'iter_033/completed_locked.json'),'old035':check(E/'iter_035/completed_locked.json')}
 save(I/'audits/completion.json',{'passed':True,'before':pre,'after':post,'method_ast_hash':asthash,'numerical':numeric,'query_reconstruction_exact':True,'source_pixels_xyz_exact':True,'strict_primary_full_source_membership_equal_in_these_frames':True,'independent_support_exact':True,'selected_frames':[55,57,60],'not_continuous_every_frame_verified':True,'slice_hash_timing':'after reading actual slices','no_target_or_training':True})
 save(I/'stage_state.json',{'stage':'computed','utc':time.time()})
if __name__=='__main__':
 try:main()
 except Exception as e:
  (I/'failure_traceback.log').write_text(traceback.format_exc())
  save(I/'failure.json',{'type':type(e).__name__,'message':str(e),'utc':time.time()})
  sys.exit(1)
