from pathlib import Path
import os,sys,json,hashlib
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,cv2,h5py
R=Path('/mnt/nvme1/drh/D_VLA');I=Path(__file__).resolve().parents[1];OLD=I.parent/'iter_003'
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
 cv2.setNumThreads(4)
 refs={r['id']:r for r in json.loads((OLD/'evidence/base/protocol.json').read_text())['episodes']}
 cases=[]
 for f in sorted((OLD/'evidence/fusion').glob('*attempt*.json')):
  info=json.loads(f.read_text())
  if not info.get('held'):continue
  id=info['id'];event=f.stem
  motion=json.loads((OLD/'evidence/motion'/(id+'.json')).read_text())
  start=motion['attempts'][info['attempt']-1]['start_frame']
  records=info['registration_records']
  for bad in records:
   q=bad['frame']
   if q>=start or bad.get('valid') or bad['phase']!='pre':continue
   bridges=sorted([x for x in records if x.get('valid') and q<x['frame']<=q+32],key=lambda x:x['frame'])[:2]
   cases.append(dict(id=id,event=event,split=refs[id]['split'],query=q,bridges=bridges,source_fusion=str(f),original_failure=bad))
 summary=[]
 for case in cases:
  ident=case['id'];q=case['query'];name=case['event']+'_query'+str(q);bridges=case['bridges'];rec={k:v for k,v in case.items() if k!='bridges'}
  if len(bridges)<2:
   rec.update(valid=False,reason='fewer_than_two_bridge_frames_within32');summary.append(rec);continue
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
   alive=np.arange(len(features));tracked=features;previous=gray0;trace=[]
   for frame in range(q+1,max(b['frame'] for b in bridges)+1):
    gray=cv2.cvtColor(g['rgb'][frame],cv2.COLOR_RGB2GRAY)
    if len(alive)==0:break
    nxt,st,_=cv2.calcOpticalFlowPyrLK(previous,gray,tracked,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    back,bst,_=cv2.calcOpticalFlowPyrLK(gray,previous,nxt,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    xy=nxt.reshape(-1,2);ok=(st[:,0]>0)&(bst[:,0]>0)&(np.linalg.norm(back.reshape(-1,2)-tracked.reshape(-1,2),axis=1)<=1)&(xy[:,0]>=4)&(xy[:,1]>=4)&(xy[:,0]<dq.shape[1]-4)&(xy[:,1]<dq.shape[0]-4)
    alive=alive[ok];tracked=nxt[ok];previous=gray
    trace.append({'frame':frame,'surviving_tracks':len(alive)})
    bridge=next((b for b in bridges if b['frame']==frame),None)
    if bridge is None:continue
    dep=g['depth_m'][frame];Kt=g['K'][frame];Tc=g['camera_to_world'][frame];rd=robot.render(finger[frame],np.linalg.inv(H[frame])@Tc,Kt)
    uv=np.rint(tracked.reshape(-1,2)).astype(int)
    endpoint=(dep[uv[:,1],uv[:,0]]>0)&np.isfinite(dep[uv[:,1],uv[:,0]])&(dep[uv[:,1],uv[:,0]]<rd[uv[:,1],uv[:,0]]-.002)
    corr=np.array([patchcorr(gray0,gray,startuv[idx],xy) for idx,xy in zip(alive,tracked.reshape(-1,2))])
    endpoint&=corr>=.65;ids=alive[endpoint];uv=uv[endpoint];p1=cloud(dep,Kt,Tc)[uv[:,1]*dep.shape[1]+uv[:,0]];p0=sourcexyz[ids]
    transform,inliers,diag=estimate(p0,p1);diag.update(frame=frame,tracked_before_endpoint_filter=len(alive),endpoint_matches=len(ids))
    arrays['query_uv_'+str(frame)]=startuv[ids];arrays['bridge_uv_'+str(frame)]=tracked.reshape(-1,2)[endpoint];arrays['query_xyz_'+str(frame)]=p0;arrays['bridge_xyz_'+str(frame)]=p1;arrays['inlier_'+str(frame)]=inliers
    if transform is not None:
     composed=np.array(bridge['source_world_to_anchor_hand'])@transform
     diag['query_world_to_anchor_hand']=composed.tolist()
    result.append(diag)
   robot.close()
   consistent=False;comparison=None
   if len(result)==2 and all(x.get('valid') for x in result):
    a=apply(allquery,np.array(result[0]['query_world_to_anchor_hand']));b=apply(allquery,np.array(result[1]['query_world_to_anchor_hand']));distance=np.linalg.norm(a-b,axis=1)
    comparison={'median_mm':float(np.median(distance)*1000),'p95_mm':float(np.percentile(distance,95)*1000)}
    consistent=comparison['median_mm']<=3 and comparison['p95_mm']<=6
   rec.update(valid=bool(consistent),reason='dual_bridge_consistent_candidate' if consistent else 'insufficient_or_inconsistent_correspondence',initial_corners=len(features),trace=trace,bridge_results=result,dual_bridge_difference=comparison,automatic_target_created=False)
   arrays['query_foreground_xyz']=allquery
   np.savez_compressed(I/'correspondences'/(name+'.npz'),**arrays)
  summary.append(rec);(I/'audits'/(name+'.json')).write_text(json.dumps(rec,indent=2))
  print(json.dumps({k:v for k,v in rec.items() if k not in ['trace','original_failure','bridge_results']}),flush=True)
 out={'protocol':'preregistered006','cases':summary,'candidate_count':sum(r['valid'] for r in summary),'training_performed':False,'candidate_not_surface_truth':True}
 (I/'summary.json').write_text(json.dumps(out,indent=2))
 files=[f for f in I.rglob('*') if f.is_file() and f.name not in ['locked.json','run.log'] and '__pycache__' not in str(f)]
 (I/'locked.json').write_text(json.dumps({str(f.relative_to(I)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
 print('DIAGNOSTIC_COMPLETE',len(summary),out['candidate_count'],flush=True)
if __name__=='__main__':main()
