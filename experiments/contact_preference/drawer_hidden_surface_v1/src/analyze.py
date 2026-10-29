import os,sys,json,ast,traceback
from pathlib import Path
os.environ['MUJOCO_GL']='egl'
import numpy as np,h5py
from scipy.spatial import cKDTree
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/contact_preference/drawer_hidden_surface_v1';C=R/'experiments/contact_frames/cf017/cases/drawer_demo_0'
sys.path.insert(0,str(R));from src.v3.stage2_robot_only_depth import RobotDepth
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
ns={'np':np}
tree=ast.parse((R/'experiments/contact_preference/stage012_v1/stage2_reuse_localize.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['fit','apply','robust']],type_ignores=[]),'frozen_math','exec'),ns)
fit,apply,robust=[ns[k] for k in ['fit','apply','robust']]
tree=ast.parse((R/'experiments/contact_preference/contact_location_5cam_v1/reliability/src/run.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='render_group'],type_ignores=[]),'frozen_render','exec'),ns)
render_group=ns['render_group']
import importlib.util
p=R/'experiments/contact_preference/occluded_surface_fit_v1/src/fit_surfaces.py'
spec=importlib.util.spec_from_file_location('fit_tools',p);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
def back(dep,K,T,uv):
 u,v=uv.T;z=dep[v,u];P=np.c_[(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z];return P@T[:3,:3].T+T[:3,3]
def proj(P,K,T):
 Q=np.linalg.solve(T[:3,:3],(P-T[:3,3]).T).T
 uv=Q[:,:2]/Q[:,2,None]*np.array([K[0,0],K[1,1]])+np.array([K[0,2],K[1,2]])
 return uv,Q[:,2]
def assign(A,trainX,lab):
 dist,ids=cKDTree(trainX).query(A,k=3);ll=lab[ids]
 return np.where((dist[:,-1]<=.006)&(ll[:,0]==ll[:,1])&(ll[:,0]==ll[:,2]),ll[:,0],-1)
def main():
 seed=np.load(D/'data/source_seed.npz');tr=np.load(D/'data/tracks.npz');mot=np.load(C/'motion.npz');cfg=json.loads((D/'config.json').read_text())
 X=seed['seeds_world'];S=seed['source_world'];held=seed['heldout'];robot=RobotDepth(448);rows=[]
 h=h5py.File(R/'experiments/contact_frames/cf017/rgbd/drawer_demo_0.h5','r')
 for t in cfg['frames']:
  K=h['agentview/K'][t];T=h['agentview/camera_to_world'][t];dep=h['agentview/depth_m'][t];H=mot['hand_to_world'][t];q=mot['measured_finger_q'][t]
  rd=robot.render(q,np.linalg.inv(H)@T,K)
  uv=np.rint(tr['tracks'][t]).astype(int);inside=(uv[:,0]>=0)&(uv[:,0]<448)&(uv[:,1]>=0)&(uv[:,1]<448);uvc=np.clip(uv,0,447);u,v=uvc.T;zz=dep[v,u]
  good=inside&(tr['visibility'][t]>.9)&(zz>0)&np.isfinite(zz)&(zz<rd[v,u]-.002)
  Y=back(dep,K,T,uvc);ids=np.flatnonzero(good&~held);A=X[ids];B=Y[ids]
  labels=np.full(len(ids),-1,int);static=np.linalg.norm(B-A,axis=1)<=.003;labels[static]=0
  models={0:(np.eye(3),np.zeros(3),np.flatnonzero(static))};cluster_rows=[]
  for k in range(1,4):
   rem=np.flatnonzero(labels<0)
   if len(rem)<8:break
   ans=robust(A[rem],B[rem],seed=2026+k)
   if ans is None:break
   rr,tt,im,err=ans;ii=rem[im];ratio=float(np.mean(im));median=float(np.median(err[im])) if im.any() else 1
   if len(ii)<8 or ratio<.6 or median>.002:
    cluster_rows.append({'cluster':k,'accepted_fit':False,'remaining':len(rem),'inliers':len(ii),'ratio':ratio,'median_mm':median*1000});break
   labels[ii]=k;models[k]=(rr,tt,ii)
   cluster_rows.append({'cluster':k,'accepted_fit':True,'remaining':len(rem),'inliers':len(ii),'ratio':ratio,'median_mm':median*1000,'translation_mm':(tt*1000).tolist()})
  sl=assign(S,A,labels);hi=np.flatnonzero(held);hl=assign(X[hi],A,labels)
  mapped=np.full_like(S,np.nan);spread=np.full(len(S),np.nan);supported=np.zeros(len(S),bool)
  modelreport=[]
  for k,(rr,tt,ii) in models.items():
   ss=np.flatnonzero(sl==k);hh=hi[(hl==k)&good[hi]]
   e=np.linalg.norm(apply(X[hh],rr,tt)-Y[hh],axis=1)
   med=float(np.median(e)) if len(e) else None;p95=float(np.percentile(e,95)) if len(e) else None
   valid=len(e)>=3 and med<=.003 and p95<=.006
   mapped[ss]=apply(S[ss],rr,tt);sh=np.zeros(len(ss))
   if k>0 and len(ii)>=8:
    for omit in range(len(ii)):
     jj=np.delete(ii,omit);rj,tj=fit(A[jj],B[jj]);sh=np.maximum(sh,np.linalg.norm(apply(S[ss],rj,tj)-mapped[ss],axis=1))
   spread[ss]=sh;supported[ss]=valid&(sh<=.002)
   modelreport.append({'cluster':k,'train_inliers':len(ii),'source_assigned':len(ss),'heldout_assigned_initial':int((hl==k).sum()),'heldout_available':len(hh),'heldout_median_mm':None if med is None else med*1000,'heldout_p95_mm':None if p95 is None else p95*1000,'heldout_pass':bool(valid),'loo_max_mm':float(sh.max()*1000) if len(sh) else None})
  # Geometry is diagnostic only: never change transforms to make surfaces touch.
  robot.render(q,np.linalg.inv(H)@T,K);mesh,_=M.geo.surface(robot,[2,5])
  mh=np.full_like(mapped,np.nan);finite=np.isfinite(mapped).all(1);mh[finite]=np.linalg.solve(H[:3,:3],(mapped[finite]-H[:3,3]).T).T
  distance=np.full(len(S),np.nan)
  if finite.any():distance[finite]=M.geo.distance(mesh,mh[finite])[0]
  views=[];hidden=np.zeros((2,len(S)),bool);contradiction=np.zeros_like(hidden);visible=np.zeros_like(hidden)
  for ci,cam in enumerate(cfg['cameras']):
   kc=h[cam+'/K'][t];tc=h[cam+'/camera_to_world'][t];dc=h[cam+'/depth_m'][t]
   robot.render(q,np.linalg.inv(H)@tc,kc);fd=render_group(robot,[2,5])
   pp=np.zeros((len(S),2));z=np.zeros(len(S));pp[finite],z[finite]=proj(mapped[finite],kc,tc)
   pix=np.rint(pp).astype(int);ok=finite&(z>0)&(pix[:,0]>=0)&(pix[:,0]<448)&(pix[:,1]>=0)&(pix[:,1]<448)
   pc=np.clip(pix,0,447);cu,cv=pc.T;real=dc[cv,cu];finger=fd[cv,cu]
   # Robot finger is in front, and actual depth agrees with that finger.
   hidden[ci]=ok&(finger>0)&(finger<2)&(z>finger+.001)&(np.abs(real-finger)<=.003)
   visible[ci]=ok&(real>0)&(np.abs(real-z)<=.003)
   contradiction[ci]=ok&(real>z+.006)&(real>0)
   views.append({'camera':cam,'uv':pp.tolist(),'in_view':ok.tolist()})
  candidate=supported&(sl>0)&hidden.any(0)&~contradiction.any(0)&(distance<=.003)
  # Separate supported hidden surface localization from contact-compatible distance.
  localized=supported&(sl>0)&hidden.any(0)&~contradiction.any(0)
  np.savez_compressed(D/'data'/('frame_%03d.npz'%t),source_world=S,source_uv=seed['source_uv'],mapped_world=mapped,mapped_hand=mh,source_cluster=sl,supported=supported,loo_spread=spread,distance_visual_finger=distance,hidden_by_finger=hidden,visible=visible,free_space_conflict=contradiction,localized=localized,contact_compatible=candidate,train_ids=ids,train_cluster=labels,heldout_ids=hi,heldout_cluster=hl)
  row={'frame':t,'available_tracks':int(good.sum()),'train_tracks':len(ids),'static_train':int(static.sum()),'clusters':cluster_rows,'models':modelreport,'source_total':len(S),'source_unassigned':int((sl<0).sum()),'supported_moving':int((supported&(sl>0)).sum()),'finger_occluded_localized':int(localized.sum()),'contact_compatible_estimates':int(candidate.sum()),'free_space_conflicts':int((contradiction.any(0)&supported&(sl>0)).sum()),'views':views,'localized_ids':np.flatnonzero(localized).tolist(),'candidate_ids':np.flatnonzero(candidate).tolist(),'unknown_ids':np.flatnonzero(~candidate).tolist(),'source_uv':seed['source_uv'].tolist()}
  save(D/'data'/('frame_%03d.json'%t),row);rows.append(row)
  print(json.dumps({k:row[k] for k in ['frame','available_tracks','supported_moving','finger_occluded_localized','contact_compatible_estimates','source_unassigned']}),flush=True)
 h.close();robot.close()
 save(D/'summary.json',{'time':now(),'case':'drawer_demo_0','cameras':cfg['cameras'],'source_frame':0,'source_points':319,'rows':rows,'physical_contact_truth':False,'scoring_training':False})
 save(D/'stage_state.json',{'time':now(),'stage':'analyzed'})
if __name__=='__main__':
 try:main()
 except Exception as e:
  (D/'analyze_failure.log').write_text(traceback.format_exc());save(D/'failure.json',{'time':now(),'type':type(e).__name__,'message':str(e),'stack':[{'file':x.filename,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(e.__traceback__)]});print('FAILED '+type(e).__name__);sys.exit(1)
