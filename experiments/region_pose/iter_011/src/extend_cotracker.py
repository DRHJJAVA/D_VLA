from pathlib import Path
import os,sys,json,hashlib
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,cv2,h5py,torch
R=Path('/mnt/nvme1/drh/D_VLA');BASE=Path(__file__).resolve().parents[1];SPLIT=sys.argv[1];assert SPLIT in ['train','val'];I=BASE/SPLIT;OLD=BASE.parent/'iter_003'
sys.path[:0]=[str(R/'src/v3'),str(R/'src')]
from stage2_probe_motion_fingerpads import cloud
from stage2_robot_only_depth import RobotDepth
def rigid(a,b):
 ac=a.mean(0);bc=b.mean(0);u,s,v=np.linalg.svd((a-ac).T@(b-bc));rot=v.T@u.T
 if np.linalg.det(rot)<0:v[-1]*=-1;rot=v.T@u.T
 T=np.eye(4);T[:3,:3]=rot;T[:3,3]=bc-rot@ac;return T
def apply(p,T):return p@T[:3,:3].T+T[:3,3]
def estimate(a,b):
 if len(a)<8:return None,np.zeros(len(a),bool),{'valid':False,'reason':'fewer_than_8_endpoint_matches','matches':len(a)}
 rng=np.random.default_rng(2026);best=np.zeros(len(a),bool)
 for _ in range(500):
  ix=rng.choice(len(a),3,replace=False)
  if np.linalg.norm(np.cross(a[ix[1]]-a[ix[0]],a[ix[2]]-a[ix[0]]))<1e-6:continue
  T=rigid(a[ix],b[ix]);inlier=np.linalg.norm(apply(a,T)-b,axis=1)<=.003
  if inlier.sum()>best.sum():best=inlier
 if best.sum()<8:return None,best,{'valid':False,'reason':'fewer_than_8_rigid_inliers','matches':len(a),'inliers':int(best.sum())}
 T=rigid(a[best],b[best]);err=np.linalg.norm(apply(a,T)-b,axis=1);inlier=err<=.003
 if inlier.sum()>=8:T=rigid(a[inlier],b[inlier]);err=np.linalg.norm(apply(a,T)-b,axis=1)
 scale=np.linalg.svd(a[inlier]-a[inlier].mean(0),compute_uv=False)/np.sqrt(max(1,inlier.sum()))
 valid=bool(inlier.sum()>=8 and inlier.mean()>=.6 and len(scale)>1 and scale[1]>=.003 and np.median(err[inlier])<=.002)
 return T,inlier,dict(valid=valid,matches=len(a),inliers=int(inlier.sum()),ratio=float(inlier.mean()),second_axis_scale_mm=float(scale[1]*1000),median_inlier_mm=float(np.median(err[inlier])*1000))
def patchcorr(g0,g1,p0,p1):
 a=cv2.getRectSubPix(g0,(7,7),tuple(map(float,p0))).astype(float);b=cv2.getRectSubPix(g1,(7,7),tuple(map(float,p1))).astype(float)
 if min(a.std(),b.std())<3:return -1.
 a-=a.mean();b-=b.mean();return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
def main():
 cv2.setNumThreads(4);torch.set_num_threads(4);torch.manual_seed(2026)
 torch.cuda.set_per_process_memory_fraction(.15)
 repo=R/'third_party/co_tracker_010';sys.path.insert(0,str(repo))
 from cotracker.predictor import CoTrackerPredictor
 weights=BASE.parent/'iter_010/weights/scaled_offline.pth'
 assert hashlib.sha256(weights.read_bytes()).hexdigest()=='2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834'
 model=CoTrackerPredictor(checkpoint=str(weights),offline=True,v2=False,window_len=60).eval().cuda()
 baseline={x['id']:x for x in json.loads((BASE.parent/'iter_008'/SPLIT/'summary.json').read_text())['cases']}

 refs={r['id']:r for r in json.loads((OLD/'evidence/base/protocol.json').read_text())['episodes']}
 cases=[];excluded=[]
 for f in sorted((OLD/'evidence/fusion').glob('*attempt*.json')):
  info=json.loads(f.read_text());ident=info['id']
  if refs[ident]['split']!=SPLIT:continue
  if not info.get('held'):
   excluded.append({'id':ident,'event':f.stem,'reason':'no_held_evidence_not_failure'});continue
  motion=json.loads((OLD/'evidence/motion'/(ident+'.json')).read_text())
  attempt=motion['attempts'][info['attempt']-1]
  pre=[x for x in info['registration_records'] if x['phase']=='pre' and x['frame']<attempt['start_frame']]
  runs=[x for x in attempt.get('evidence_runs',[]) if len(x)>=5]
  if not pre or not runs:
   excluded.append({'id':ident,'event':f.stem,'reason':'no_strict_pre_or_continuous_support'});continue
  prior=max(pre,key=lambda x:x['frame']);q=prior['frame'];run=min(runs,key=lambda x:x[0]);start=run[0];anchor=info['anchor_frame']
  endpoints=sorted(set([t for t in [start,start+2,start+5,start+10,anchor] if t in run and q<t<=q+64]))
  bridges=[{'frame':t} for t in endpoints]
  cases.append(dict(id=ident,event=f.stem,split=SPLIT,query=q,bridges=bridges,source_fusion=str(f),anchor_frame=anchor,selected_held_frames=endpoints,early_support_start=start,original_failure=prior))
 (I/'excluded_events.json').write_text(json.dumps(excluded,indent=2))
 summary=[]
 for case in cases:
  ident=case['id'];q=case['query'];name=case['event']+'_query'+str(q);bridges=case['bridges'];rec={k:v for k,v in case.items() if k!='bridges'}
  case_lock=I/'case_locks'/(name+'.json')
  if case_lock.exists():
   lock=json.loads(case_lock.read_text())
   assert all(hashlib.sha256((I/f).read_bytes()).hexdigest()==h for f,h in lock.items())
   reused=json.loads((I/'audits'/(name+'.json')).read_text())
   assert reused['query']==q and reused['selected_held_frames']==case['selected_held_frames']
   summary.append(reused);print('REUSE',name,flush=True);continue
  if len(bridges)<2:
   rec.update(valid=False,reason='missing_reference_and_same_run_held_frame_within64');summary.append(rec);continue
  ref=refs[ident];arrays={};result=[]
  with np.load(OLD/'evidence/motion'/(ident+'.npz')) as z:H=z['hand_to_world'];finger=z['measured_finger_q']
  with np.load(ref['component']) as z:pix=z['pixel'].astype(np.int64)
  with h5py.File(ref['rgbd'],'r') as f:
   g=f['agentview'];rgb=g['rgb'][q];dq=g['depth_m'][q];K=g['K'][q];T=g['camera_to_world'][q];d0=g['depth_m'][0]
   robot=RobotDepth(dq.shape[0]);rd=robot.render(finger[q],np.linalg.inv(H[q])@T,K)
   u,v=pix[:,0],pix[:,1];valid=(abs(dq[v,u]-d0[v,u])<.005)&(dq[v,u]<rd[v,u]-.002)&(dq[v,u]>0)&np.isfinite(dq[v,u])
   mask=np.zeros(dq.shape,np.uint8);mask[v[valid],u[valid]]=255;mask=cv2.erode(mask,np.ones((3,3),np.uint8))
   gray0=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);features=cv2.goodFeaturesToTrack(gray0,300,.01,3,mask=mask,blockSize=3)
   if features is None or len(features)<8:
    rec.update(valid=False,reason='fewer_than_8_query_natural_corners',corners=0 if features is None else len(features));summary.append(rec);robot.close();continue
   startuv=features.reshape(-1,2);queryworld=cloud(dq,K,T);rounduv=np.rint(startuv).astype(int);sourcexyz=queryworld[rounduv[:,1]*dq.shape[1]+rounduv[:,0]]
   allquery=queryworld[pix[valid,1]*dq.shape[1]+pix[valid,0]]
   assert len(features)==baseline[ident]['initial_corners'],(ident,len(features),baseline[ident]['initial_corners'])
   assert q==baseline[ident]['query'] and case['selected_held_frames']==baseline[ident]['selected_held_frames']
   endframe=max(b['frame'] for b in bridges)
   video=torch.from_numpy(g['rgb'][q:endframe+1]).permute(0,3,1,2).unsqueeze(0).float().cuda()
   query=torch.from_numpy(np.c_[np.zeros(len(startuv)),startuv].astype(np.float32)).unsqueeze(0).cuda()
   with torch.inference_mode():pred,pvis=model(video,queries=query,backward_tracking=False)
   pred=pred[0].cpu().numpy();pvis=pvis[0].cpu().numpy().astype(bool)
   np.savez_compressed(I/'trajectories'/(name+'.npz'),initial_uv=startuv,frames=np.arange(q,endframe+1),tracks=pred,visible=pvis)
   del video,query;torch.cuda.empty_cache()
   trace=[]
   for frame in range(q+1,endframe+1):
    gray=cv2.cvtColor(g['rgb'][frame],cv2.COLOR_RGB2GRAY)
    xy=pred[frame-q];ok=pvis[frame-q] & np.isfinite(xy).all(1)&(xy[:,0]>=4)&(xy[:,1]>=4)&(xy[:,0]<dq.shape[1]-4)&(xy[:,1]<dq.shape[0]-4)
    alive=np.flatnonzero(ok);tracked=xy[ok].reshape(-1,1,2)
    trace.append({'frame':frame,'model_visible':int(pvis[frame-q].sum()),'surviving_tracks':len(alive)})
    bridge=next((b for b in bridges if b['frame']==frame),None)
    if bridge is None:continue
    dep=g['depth_m'][frame];Kt=g['K'][frame];Tc=g['camera_to_world'][frame];rd=robot.render(finger[frame],np.linalg.inv(H[frame])@Tc,Kt)
    uv=np.rint(tracked.reshape(-1,2)).astype(int)
    endpoint=(dep[uv[:,1],uv[:,0]]>0)&np.isfinite(dep[uv[:,1],uv[:,0]])&(dep[uv[:,1],uv[:,0]]<rd[uv[:,1],uv[:,0]]-.002)
    corr=np.array([patchcorr(gray0,gray,startuv[idx],xy) for idx,xy in zip(alive,tracked.reshape(-1,2))])
    endpoint&=corr>=.65;ids=alive[endpoint];uv=uv[endpoint];p1=cloud(dep,Kt,Tc)[uv[:,1]*dep.shape[1]+uv[:,0]];p0=sourcexyz[ids]
    spans=[]
    for idx,xy in zip(ids,uv):
     sq=np.rint(startuv[idx]).astype(int);b0=dq[sq[1]-1:sq[1]+2,sq[0]-1:sq[0]+2];b1=dep[xy[1]-1:xy[1]+2,xy[0]-1:xy[0]+2]
     spans.append(max(float(np.ptp(b0)),float(np.ptp(b1)))*1000)
    transform,inliers,diag=estimate(p0,p1)
    sp=np.array(spans)
    diag['depth_neighborhood']={k:{'matches':int(mask.sum()),'inliers':int((mask&inliers).sum())} for k,mask in [('smooth_le5mm',sp<=5),('middle', (sp>5)&(sp<=10)),('jump_gt10mm',sp>10)]}
    diag.update(frame=frame,tracked_before_endpoint_filter=len(alive),endpoint_matches=len(ids))
    arrays['track_id_'+str(frame)]=ids;arrays['observed_hand_xyz_'+str(frame)]=apply(p1,np.linalg.inv(H[frame]));arrays['hand_to_world_'+str(frame)]=H[frame];arrays['query_uv_'+str(frame)]=startuv[ids];arrays['bridge_uv_'+str(frame)]=tracked.reshape(-1,2)[endpoint];arrays['query_xyz_'+str(frame)]=p0;arrays['bridge_xyz_'+str(frame)]=p1;arrays['inlier_'+str(frame)]=inliers
    if transform is not None:
     composed=np.linalg.inv(H[frame])@transform
     diag['query_world_to_anchor_hand']=composed.tolist()
    result.append(diag)
   robot.close()
   comparisons=[]
   for later in result[1:]:
    first=result[0];comparison={'frames':[first['frame'],later['frame']],'valid':False,'reason':'endpoint_quality_failed'}
    if first.get('valid') and later.get('valid'):
     d=np.linalg.norm(apply(allquery,np.array(first['query_world_to_anchor_hand']))-apply(allquery,np.array(later['query_world_to_anchor_hand'])),axis=1)
     f0,f1=str(first['frame']),str(later['frame'])
     ids0=arrays['track_id_'+f0][arrays['inlier_'+f0]];ids1=arrays['track_id_'+f1][arrays['inlier_'+f1]]
     common,ix0,ix1=np.intersect1d(ids0,ids1,return_indices=True)
     hp0=arrays['observed_hand_xyz_'+f0][arrays['inlier_'+f0]][ix0];hp1=arrays['observed_hand_xyz_'+f1][arrays['inlier_'+f1]][ix1]
     cd=np.linalg.norm(hp0-hp1,axis=1)
     comparison.update(median_mm=float(np.median(d)*1000),p95_mm=float(np.percentile(d,95)*1000),common_observed_inliers=len(common),observed_median_mm=float(np.median(cd)*1000) if len(cd) else None,observed_p95_mm=float(np.percentile(cd,95)*1000) if len(cd) else None)
     comparison['valid']=bool(comparison['median_mm']<=3 and comparison['p95_mm']<=6 and len(common)>=8 and comparison['observed_median_mm']<=3 and comparison['observed_p95_mm']<=6)
     comparison['reason']='consistent_candidate' if comparison['valid'] else 'hand_frame_inconsistent'
    comparisons.append(comparison)
   comparison=comparisons[0] if comparisons else None
   consistent=bool(comparison and comparison['valid'] and comparison['frames']==case['selected_held_frames'][:2])
   rec.update(valid=bool(consistent),reason='direct_held_surface_consistent_candidate' if consistent else 'insufficient_or_inconsistent_correspondence',initial_corners=len(features),trace=trace,bridge_results=result,dual_bridge_difference=comparison,time_comparisons=comparisons,automatic_target_created=False)
   arrays['query_foreground_xyz']=allquery
   np.savez_compressed(I/'correspondences'/(name+'.npz'),**arrays)
  summary.append(rec);(I/'audits'/(name+'.json')).write_text(json.dumps(rec,indent=2))
  casefiles=[f for f in [I/'audits'/(name+'.json'),I/'correspondences'/(name+'.npz'),I/'trajectories'/(name+'.npz')] if f.exists()]
  case_lock.write_text(json.dumps({str(f.relative_to(I)):hashlib.sha256(f.read_bytes()).hexdigest() for f in casefiles},indent=2))
  print(json.dumps({k:v for k,v in rec.items() if k not in ['trace','original_failure','bridge_results']}),flush=True)
 out={'protocol':'preregistered011_cotracker3_frozen_extension','cases':summary,'candidate_count':sum(r['valid'] for r in summary),'split':SPLIT,'excluded_events':excluded,'training_performed':False,'candidate_not_surface_truth':True}
 (I/'summary.json').write_text(json.dumps(out,indent=2))
 files=[f for f in I.rglob('*') if f.is_file() and f.name not in ['locked.json','run.log'] and '__pycache__' not in str(f)]
 (I/'locked.json').write_text(json.dumps({str(f.relative_to(I)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
 print('DIAGNOSTIC_COMPLETE',len(summary),out['candidate_count'],flush=True)
if __name__=='__main__':main()
