from pathlib import Path
import os,sys,json,hashlib,time,fcntl,traceback
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,cv2,h5py,torch
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA');B=Path(__file__).resolve().parents[1];P=B.parent
sys.path[:0]=[str(R/'src/v3'),str(R/'third_party/co_tracker_010')]
from stage2_probe_motion_fingerpads import cloud
from stage2_robot_only_depth import RobotDepth
from tracking_math import rigid,apply,estimate,patchcorr
from cotracker.predictor import CoTrackerPredictor
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2))
def input_cloud(f,component,motion,t,robot):
 g=f['agentview'];dep=g['depth_m'][t];K=g['K'][t];T=g['camera_to_world'][t];rgb=g['rgb'][t]
 H=motion['hand_to_world'][t];finger=motion['measured_finger_q'][t];rd=robot.render(finger,np.linalg.inv(H)@T,K)
 uv=component;u,v=uv.T;ok=(abs(dep[v,u]-g['depth_m'][0][v,u])<.005)&(dep[v,u]<rd[v,u]-.002)&(dep[v,u]>0)&np.isfinite(dep[v,u]);uv=uv[ok]
 xyz=cloud(dep,K,T);points=xyz[uv[:,1]*dep.shape[1]+uv[:,0]]
 return uv,points,rgb,dep,K,T,rd,xyz
def box_score(points,H,q):
 p=apply(points,np.linalg.inv(H));lo=np.array([-.008,q[1]+.0005,.0854]);hi=np.array([.008,q[0]-.0005,.1014])
 return -np.linalg.norm(np.maximum(lo-p,0)+np.maximum(p-hi,0),axis=1)
def metrics(score,y,known):
 ids=np.flatnonzero(known);order=ids[np.argsort(-score[ids],kind='stable')];yy=y[order]
 out={'ap':float(((np.cumsum(yy)/np.arange(1,len(yy)+1))*yy).sum()/yy.sum())}
 for b in [.1,.2,.3]:
  k=int(np.ceil(b*len(ids)));out['coverage'+str(int(b*100))]=float(yy[:k].sum()/yy.sum());out['random'+str(int(b*100))]=k/len(ids)
 return out
def main():
 torch.set_num_threads(4);cv2.setNumThreads(4);torch.manual_seed(2026);torch.cuda.set_per_process_memory_fraction(.15)
 for lock in ['source_locked.json','input_locked.json']:
  for f,h in json.loads((B/lock).read_text()).items():assert sha(R/f)==h,(f,'hash changed')
 model=None
 refs={c['id']:c for c in json.loads((P/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
 cases=json.loads((P/'iter_013/targets/train/summary.json').read_text())['rows'];rows=[]
 for c in cases:
  ident=c['episode'];refq=c['query_frame'];event=c['event']
  motion=np.load(P/'iter_003/evidence/motion'/(ident+'.npz'))
  at=int(event.rsplit('attempt',1)[1]);start=json.loads((P/'iter_003/evidence/motion'/(ident+'.json')).read_text())['attempts'][at-1]['start_frame']
  with np.load(refs[ident]['component']) as z:component=z['pixel'].astype(np.int64)
  ref=np.load(c['target_source']);refworld=ref['query_xyz'];refknown=ref['known'];refy=ref['target'].astype(bool);tree=cKDTree(refworld)
  with h5py.File(refs[ident]['rgbd'],'r') as f:
   g=f['agentview'];robot=RobotDepth(g['depth_m'].shape[1])
   for offset in [15,10,5,1]:
    t=start-offset;name=event+'_query'+str(t);jpath=B/'train'/(name+'.json');zpath=B/'train'/(name+'.npz');lock=B/'train'/(name+'_locked.json')
    if lock.exists():
     for fn,h in json.loads(lock.read_text()).items():assert sha(B/'train'/fn)==h
     rows.append(json.loads(jpath.read_text()));continue
    rec={'episode':ident,'event':event,'query':t,'reference_query':refq,'close_start':start,'offset':offset,'valid':False}
    arrays={}
    if t<0 or t>refq:
     rec['reason']='illegal_fixed_query';dump(jpath,rec);rows.append(rec);continue
    uv,points,rgb,dep,K,T,rd,xyz=input_cloud(f,component,motion,t,robot)
    arrays.update(query_pixel=uv,query_xyz=points,query_rgb=rgb[uv[:,1],uv[:,0]])
    rec['causal_query_points']=len(points)
    if len(points)<128:
     rec['reason']='fewer_than128_causal_points';dump(jpath,rec);np.savez_compressed(zpath,**arrays);dump(lock,{jpath.name:sha(jpath),zpath.name:sha(zpath)});rows.append(rec);continue
    if t==refq:
     assert np.array_equal(points,refworld)
     mapped=points;known=refknown.copy();y=refy.copy();W=np.eye(4);spread=np.zeros(len(points));dist=np.zeros(len(points))
     rec.update(reason='same_frame_reference_reused_not_tracking',correspondence_valid=True,leave_one_out_fits=None)
    else:
     mask=np.zeros(dep.shape,np.uint8);mask[uv[:,1],uv[:,0]]=255;mask=cv2.erode(mask,np.ones((3,3),np.uint8))
     gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);features=cv2.goodFeaturesToTrack(gray,300,.01,3,mask=mask,blockSize=3)
     rec['initial_corners']=0 if features is None else len(features)
     if features is None or len(features)<8:
      rec['reason']='fewer_than8_initial_corners';dump(jpath,rec);np.savez_compressed(zpath,**arrays);dump(lock,{jpath.name:sha(jpath),zpath.name:sha(zpath)});rows.append(rec);continue
     initial=features.reshape(-1,2);p0uv=np.rint(initial).astype(int);allp0=xyz[p0uv[:,1]*dep.shape[1]+p0uv[:,0]]
     if model is None:model=CoTrackerPredictor(checkpoint=str(P/'iter_010/weights/scaled_offline.pth'),offline=True,v2=False,window_len=60).eval().cuda()
     video=torch.from_numpy(g['rgb'][t:refq+1]).permute(0,3,1,2).unsqueeze(0).float().cuda();query=torch.from_numpy(np.c_[np.zeros(len(initial)),initial].astype(np.float32)).unsqueeze(0).cuda()
     with torch.inference_mode():tracks,vis=model(video,queries=query,backward_tracking=False)
     tracks=tracks[0].cpu().numpy();vis=vis[0].cpu().numpy().astype(bool);del video,query
     xy=tracks[-1];ok=vis[-1]&np.isfinite(xy).all(1)&(xy[:,0]>=4)&(xy[:,1]>=4)&(xy[:,0]<dep.shape[1]-4)&(xy[:,1]<dep.shape[0]-4)
     ids=np.flatnonzero(ok);ruv=np.rint(xy[ids]).astype(int)
     dep1=g['depth_m'][refq];K1=g['K'][refq];T1=g['camera_to_world'][refq];rgb1=g['rgb'][refq]
     rd1=robot.render(motion['measured_finger_q'][refq],np.linalg.inv(motion['hand_to_world'][refq])@T1,K1);gray1=cv2.cvtColor(rgb1,cv2.COLOR_RGB2GRAY)
     endpoint=(dep1[ruv[:,1],ruv[:,0]]>0)&np.isfinite(dep1[ruv[:,1],ruv[:,0]])&(dep1[ruv[:,1],ruv[:,0]]<rd1[ruv[:,1],ruv[:,0]]-.002)
     appearance=np.array([patchcorr(gray,gray1,initial[i],xy[i]) for i in ids]);endpoint&=appearance>=.65
     ids=ids[endpoint];ruv=ruv[endpoint];a=allp0[ids];bb=cloud(dep1,K1,T1)[ruv[:,1]*dep1.shape[1]+ruv[:,0]]
     W,ins,diag=estimate(a,bb);rec['correspondence']=diag
     arrays.update(initial_uv=initial,tracks=tracks,track_visible=vis,query_match_uv=initial[ids],reference_match_uv=xy[ids],query_match_xyz=a,reference_match_xyz=bb,inlier=ins,track_ids=ids)
     if not diag['valid']:
      rec['reason']='rigid_correspondence_failed';dump(jpath,rec);np.savez_compressed(zpath,**arrays);dump(lock,{jpath.name:sha(jpath),zpath.name:sha(zpath)});rows.append(rec);print(json.dumps(rec),flush=True);continue
     src=a[ins];dst=bb[ins];alternatives=[]
     for j in range(len(src)):
      aa=np.delete(src,j,0);b=np.delete(dst,j,0);scale=np.linalg.svd(aa-aa.mean(0),compute_uv=False)/np.sqrt(len(aa))
      if len(aa)>=7 and scale[1]>=.003:alternatives.append(rigid(aa,b))
     mapped=apply(points,W);dist,ni=tree.query(mapped);y=refy[ni];known=refknown[ni]&(dist<=.002);spread=np.zeros(len(points))
     for alt in alternatives:
      pp=apply(points,alt);d,idx=tree.query(pp);spread=np.maximum(spread,np.linalg.norm(pp-mapped,axis=1));known&=refknown[idx]&(refy[idx]==y)&(d<=.002)
     known&=spread<=.002
     if len(alternatives)<7:known[:]=False
     rec.update(reason='supported_transfer_diagnostic',correspondence_valid=True,leave_one_out_fits=len(alternatives))
    y=y&known
    pos=int(y.sum());neg=int((known&~y).sum());rec.update(positive=pos,negative=neg,unknown=int((~known).sum()),known_fraction=float(known.mean()))
    H=motion['hand_to_world'][t];Hr=motion['hand_to_world'][refq];q=motion['measured_finger_q'][t];qr=motion['measured_finger_q'][refq]
    early=box_score(points,H,q);late=box_score(mapped,Hr,qr);center=-np.linalg.norm(points-points.mean(0),axis=1)
    rec['hand_translation_to_reference_mm']=float(np.linalg.norm(H[:3,3]-Hr[:3,3])*1000)
    rec['hand_rotation_to_reference_deg']=float(np.degrees(np.arccos(np.clip((np.trace(H[:3,:3].T@Hr[:3,:3])-1)/2,-1,1))))
    if pos>=3 and neg>=16:
     rec.update(valid=True,early_geometry=metrics(early,y,known),reference_geometry_on_same_points=metrics(late,y,known),center=metrics(center,y,known))
    else:rec['metric_reason']='insufficient_transferred_positive_or_negative'
    arrays.update(query_to_reference_world=W,mapped_reference_xyz=mapped,position_sensitivity_m=spread,reference_surface_distance_m=dist,diagnostic_target=y,diagnostic_known=known,early_geometry_score=early,reference_geometry_score=late)
    np.savez_compressed(zpath,**arrays);dump(jpath,rec);dump(lock,{jpath.name:sha(jpath),zpath.name:sha(zpath)});rows.append(rec);print(json.dumps(rec),flush=True)
   robot.close()
 offsets=[]
 for offset in [15,10,5,1]:
  rr=[r for r in rows if r['offset']==offset];valid=[r for r in rr if r['valid']]
  d={'offset':offset,'events':len(rr),'valid_metric_events':len(valid)}
  if len(valid)==6:d.update(early_ap=float(np.mean([r['early_geometry']['ap'] for r in valid])),reference_ap_same_points=float(np.mean([r['reference_geometry_on_same_points']['ap'] for r in valid])))
  offsets.append(d)
 summary={'rows':rows,'offsets':offsets,'training_events':6,'official_train_episodes':20,'validation_read':False,'test_read':False,'training_performed':False,'new_training_samples_created':False}
 dump(B/'train/summary.json',summary)
 dump(B/'train/locked.json',{f.name:sha(f) for f in (B/'train').iterdir() if f.is_file() and f.name!='locked.json'})
 dump(B/'stage_state.json',{'stage':'complete','time':time.time()})
if __name__=='__main__':
 with (B/'pipeline.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   dump(B/'stage_state.json',{'stage':'train_diagnostic','time':time.time()});main()
  except Exception as e:
   dump(B/'blocked.json',{'error':str(e),'traceback':traceback.format_exc()});raise
