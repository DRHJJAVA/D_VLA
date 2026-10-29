from pathlib import Path
import os,sys,json,hashlib,time
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,h5py
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA'); B=Path(__file__).resolve().parents[1]; P=B.parent; OLD=P/'iter_003'; PREV=P/'iter_011'
sys.path.insert(0,str(R/'src/v3'))
from stage2_probe_motion_fingerpads import cloud
from stage2_robot_only_depth import RobotDepth
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def apply(p,T):return p@T[:3,:3].T+T[:3,3]
def rigid(a,b):
 ac=a.mean(0);bc=b.mean(0);u,s,v=np.linalg.svd((a-ac).T@(b-bc));rot=v.T@u.T
 if np.linalg.det(rot)<0:v[-1]*=-1;rot=v.T@u.T
 T=np.eye(4);T[:3,:3]=rot;T[:3,3]=bc-rot@ac;return T
def stats(a):
 return {'count':int(len(a)),'median':float(np.median(a)) if len(a) else None,'p95':float(np.percentile(a,95)) if len(a) else None,'max':float(np.max(a)) if len(a) else None}
def in_region(p,q):
 return (abs(p[:,0])<=.008)&(p[:,2]>=.0854)&(p[:,2]<=.1014)&(p[:,1]>=q[1]+.0005)&(p[:,1]<=q[0]-.0005)
def visible(p,d,rd,K,T):
 cam=apply(p,np.linalg.inv(T));z=cam[:,2];uv=np.rint(cam[:,:2]/np.maximum(z[:,None],1e-9)*np.array([K[0,0],K[1,1]])+K[:2,2]).astype(int)
 ok=(z>0)&(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<d.shape[1])&(uv[:,1]<d.shape[0])
 ids=np.flatnonzero(ok);u,v=uv[ids].T
 out=np.zeros(len(p),bool);out[ids]=(abs(d[v,u]-z[ids])<=.003)&(d[v,u]<rd[v,u]-.002)&np.isfinite(d[v,u])&(d[v,u]>0)
 return out
def main(split):
 for f,h in json.loads((B/'source_locked.json').read_text()).items():assert sha(R/f)==h,(f,'source changed')
 for f,h in json.loads((B/'input_locked.json').read_text()).items():assert sha(R/f)==h,(f,'input changed')
 out=B/split
 if (out/'locked.json').exists():
  for f,h in json.loads((out/'locked.json').read_text()).items():assert sha(out/f)==h
  print('REUSE_SPLIT',split,flush=True);return
 if split=='val':
  freeze=json.loads((B/'pre_validation_freeze.json').read_text())
  assert sha(B/'train/summary.json')==freeze['train_summary_sha256']
  assert sha(B/'source_locked.json')==freeze['source_lock_sha256']
 refs={c['id']:c for c in json.loads((OLD/'evidence/base/protocol.json').read_text())['episodes']}
 allcases=json.loads((PREV/split/'summary.json').read_text())['cases']; cases=[c for c in allcases if c['valid']]; summaries=[]
 for c in cases:
  name=c['event']+'_query'+str(c['query']);cp=out/(name+'.json');npz=out/(name+'.npz');cl=out/(name+'_locked.json')
  if cl.exists():
   for f,h in json.loads(cl.read_text()).items():assert sha(out/f)==h
   summaries.append(json.loads(cp.read_text()));print('REUSE',name,flush=True);continue
  frames=[c['selected_held_frames'][0]]
  for comp in c['time_comparisons']:
   if not comp['valid']:break
   assert comp['frames'][0]==frames[0];frames.append(comp['frames'][1])
  assert len(frames)>=2 and frames==c['selected_held_frames'][:len(frames)]
  ref=refs[c['id']]
  with np.load(PREV/split/'correspondences'/(name+'.npz')) as z:corr={k:z[k] for k in z.files}
  with np.load(OLD/'evidence/motion'/(c['id']+'.npz')) as z:H=z['hand_to_world'];finger=z['measured_finger_q']
  with np.load(ref['component']) as z:pix=z['pixel'].astype(np.int64)
  raw={k:[] for k in ['source_xyz','source_hand_xyz','mapped_query_xyz','rgb','pixel','camera','frame','distance_to_query','query_nearest_index']}
  arrays={}; perframe=[]; visible_frames=[]
  with h5py.File(ref['rgbd'],'r') as f:
   g=f['agentview'];q=c['query'];dq=g['depth_m'][q];Kq=g['K'][q];Tq=g['camera_to_world'][q];robot=RobotDepth(len(dq))
   rdq=robot.render(finger[q],np.linalg.inv(H[q])@Tq,Kq)
   u,v=pix.T
   keep=(abs(dq[v,u]-g['depth_m'][0][v,u])<.005)&(dq[v,u]<rdq[v,u]-.002)&(dq[v,u]>0)&np.isfinite(dq[v,u])
   qp=pix[keep];query=cloud(dq,Kq,Tq)[qp[:,1]*dq.shape[1]+qp[:,0]]
   assert np.array_equal(query,corr['query_foreground_xyz'])
   arrays.update(query_xyz=query,query_pixel=qp,query_rgb=g['rgb'][q][qp[:,1],qp[:,0]],query_K=Kq,query_camera_to_world=Tq)
   tree=cKDTree(query)
   for t in frames:
    diag=next(x for x in c['bridge_results'] if x['frame']==t);assert diag['valid']
    Q=np.array(diag['query_world_to_anchor_hand']);W=H[t]@Q;Wi=np.linalg.inv(W)
    assert np.allclose(corr['hand_to_world_'+str(t)],H[t],atol=1e-12)
    measured=corr['query_xyz_'+str(t)];target=corr['bridge_xyz_'+str(t)];ins=corr['inlier_'+str(t)]
    # Independently confirm matrix convention against actual 3-D inliers.
    err=np.linalg.norm(apply(measured[ins],W)-target[ins],axis=1)
    assert np.median(err)<=.002+1e-10
    arrays['query_to_current_world_'+str(t)]=W;arrays['query_to_current_hand_'+str(t)]=Q
    arrays['hand_to_world_'+str(t)]=H[t];arrays['finger_q_'+str(t)]=finger[t]
    record={'frame':t,'inlier_residual_mm':stats(err*1000),'cameras':[]};queryvis=np.zeros(len(query),bool);framep=[]
    for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
     gg=f[cam];dep=gg['depth_m'][t];K=gg['K'][t];T=gg['camera_to_world'][t];rgb=gg['rgb'][t]
     rd=robot.render(finger[t],np.linalg.inv(H[t])@T,K)
     world=cloud(dep,K,T);hand=apply(world,np.linalg.inv(H[t]))
     ok=(np.isfinite(dep)&(dep>0)&(dep<rd-.002)).reshape(-1)&in_region(hand,finger[t])
     idx=np.flatnonzero(ok);uv=np.c_[idx%dep.shape[1],idx//dep.shape[1]]
     pw=world[idx];mapped=apply(pw,Wi);dist,near=tree.query(mapped)
     assert np.max(abs(apply(mapped,W)-pw),initial=0)<1e-9
     assert np.array_equal(pw,cloud(dep,K,T)[uv[:,1]*dep.shape[1]+uv[:,0]])
     for k,value in [('source_xyz',pw),('source_hand_xyz',hand[idx]),('mapped_query_xyz',mapped),('rgb',rgb.reshape(-1,3)[idx]),('pixel',uv),('camera',np.full(len(idx),ci,dtype=np.int8)),('frame',np.full(len(idx),t,dtype=np.int32)),('distance_to_query',dist),('query_nearest_index',near)]:raw[k].append(value)
     framep.append(pw);queryvis|=visible(apply(query,W),dep,rd,K,T)
     record['cameras'].append({'camera':cam,'raw_region_points':len(idx),'within_query_3mm':int((dist<=.003).sum()),'between_3_and_6mm':int(((dist>.003)&(dist<=.006)).sum()),'unsupported_or_unseen':int((dist>.006).sum())})
    visible_frames.append(queryvis)
    # Leave-one-out transform sensitivity: never used as a label or acceptance override.
    src=measured[ins];dst=target[ins];pv=np.concatenate(framep);base=apply(pv,Wi);delta=[]
    for j in range(len(src)):
     aa=np.delete(src,j,0);bb=np.delete(dst,j,0)
     scale=np.linalg.svd(aa-aa.mean(0),compute_uv=False)/np.sqrt(len(aa))
     if len(aa)<7 or scale[1]<.003:continue
     alt=rigid(aa,bb);delta.append(np.linalg.norm(apply(pv,np.linalg.inv(alt))-base,axis=1))
    record['leave_one_out_fits']=len(delta)
    record['leave_one_out_pointwise_max_shift_mm']=stats(np.max(delta,axis=0)*1000 if len(delta) and len(pv) else np.array([]))
    perframe.append(record)
   robot.close()
  arrays.update({k:np.concatenate(v) for k,v in raw.items()})
  mapped=arrays['mapped_query_xyz'];fids=arrays['frame'];near=arrays['distance_to_query']<=.003
  support=np.zeros(len(mapped),np.int16)
  for t in frames:
   ids=np.flatnonzero(near&(fids==t))
   if len(ids):support+=cKDTree(mapped[ids]).query(mapped)[0]<=.002
  supported=near&(support>=2);ids=np.flatnonzero(supported)
  if len(ids):
   _,representatives=np.unique(np.floor(mapped[ids]/.001).astype(np.int64),axis=0,return_index=True)
   representatives=ids[representatives]
  else:representatives=np.array([],dtype=np.int64)
  arrays.update(distinct_frame_support=support,supported_candidate=supported,voxel_representative_index=representatives,query_visible_at_frames=np.array(visible_frames),selected_frames=np.array(frames))
  assert np.all(arrays['distinct_frame_support'][representatives]>=2)
  np.savez_compressed(npz,**arrays)
  summary={'id':c['id'],'event':c['event'],'split':split,'query':c['query'],'selected_frames':frames,'omitted_later_endpoints':c['selected_held_frames'][len(frames):],'raw_points':len(mapped),'near_query_points':int(near.sum()),'uncertain_3to6mm':int(((arrays['distance_to_query']>.003)&(arrays['distance_to_query']<=.006)).sum()),'unsupported_or_unseen_points':int((arrays['distance_to_query']>.006).sum()),'cross_time_supported_observations':int(supported.sum()),'supported_1mm_voxels':len(representatives),'query_surface_points':len(query),'query_visible_any':int(np.any(visible_frames,axis=0).sum()),'query_visible_two_frames':int((np.sum(visible_frames,axis=0)>=2).sum()),'distance_to_query_mm':stats(arrays['distance_to_query']*1000),'per_frame':perframe,'automatic_training_labels_created':False,'training_performed':False,'exact_source_xyz_verified':True,'causal_query_cloud_exactly_matches_011':True,'inverse_transform_closure_verified':True,'region_is_not_physical_contact_truth':True}
  dump(cp,summary);dump(cl,{cp.name:sha(cp),npz.name:sha(npz)});summaries.append(summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
 summary={'split':split,'cases':summaries,'events':len(summaries),'events_with_supported_regions':sum(c['supported_1mm_voxels']>0 for c in summaries),'official_episode_count':20 if split=='train' else 4,'excluded_from_region_audit':sorted(set(r['id'] for r in refs.values() if r['split']==split)-set(c['id'] for c in summaries)),'not_grasp_success_rate':True,'training_performed':False}
 dump(out/'summary.json',summary)
 dump(out/'locked.json',{str(f.relative_to(out)):sha(f) for f in out.iterdir() if f.is_file() and f.name!='locked.json'})
 print('REGION_AUDIT_COMPLETE',split,summary['events'],flush=True)
if __name__=='__main__':main(sys.argv[1])
