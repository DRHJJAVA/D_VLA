import os,sys,json,traceback,importlib.util
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ['MUJOCO_GL']='egl';R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/contact_preference/contact_persistence_three_v1'
sys.path.insert(0,str(R))
import numpy as np,h5py
from scipy.spatial import cKDTree
def loadmod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
B=loadmod('prior_near',R/'experiments/contact_preference/drawer_precontact_multiview_v1/src/analyze.py');A=B.A
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def normals(P):
 _,ix=cKDTree(P).query(P,k=min(12,len(P)));V=P[ix]-P[ix].mean(1)[:,None];cov=np.einsum('nki,nkj->nij',V,V);ev,evec=np.linalg.eigh(cov)
 n=evec[:,:,0];ok=(ev[:,1]>1e-9)&(ev[:,0]/np.maximum(ev.sum(1),1e-12)<.1)
 return n,ok
def observations(c,h,m,s,tracks,t,rd):
 X=[];Y=[];valid=[];ends=[]
 for ci,cam in enumerate(c['cameras']):
  tr=tracks[ci];uv=np.rint(tr['tracks'][t]).astype(int);v=(tr['visible'][t]>.9)&(uv[:,0]>=0)&(uv[:,0]<448)&(uv[:,1]>=0)&(uv[:,1]<448)
  uv=np.clip(uv,0,447);dep=h[cam+'/depth_m'][t];K=h[cam+'/K'][t];T=h[cam+'/camera_to_world'][t]
  dd=dep[uv[:,1],uv[:,0]];v&=np.isfinite(dd)&(dd>0)&(dd<rd[ci][t,uv[:,1],uv[:,0]]-.002)
  yy=A.back(dep,K,T,uv);X.append(s[ci]['seeds_world']);Y.append(yy);valid.append(v);ends.append(uv)
 return np.concatenate(X),np.concatenate(Y),np.concatenate(valid),ends
def fit_checked(X,Y,v,trainmask,heldmask):
 inds=np.flatnonzero(v&trainmask)
 if len(inds)<8:return None
 z=A.robust(X[inds],Y[inds],2026)
 if z is None:return None
 rr,tt,im,err=z;ii=inds[im]
 if len(ii)<8 or np.median(err[im])>.002:return None
 hh=np.flatnonzero(v&heldmask)
 if len(hh)<3:return None
 he=np.linalg.norm(A.apply(X[hh],rr,tt)-Y[hh],axis=1)
 if np.median(he)>.003 or np.percentile(he,95)>.006:return None
 # every accepted inlier is left out once; conservative global corner displacement.
 lo=[]
 for j in range(len(ii)):
  jj=np.delete(ii,j)
  if len(jj)>=7:lo.append(A.fit(X[jj],Y[jj]))
 return dict(R=rr,t=tt,ii=ii,held=hh,held_errors=he,loo=lo)
def part_discovery(c,h,m,s,tracks,rd,robot,sf):
 lengths=[len(z['seeds_world']) for z in s];allheld=np.concatenate([z['heldout'] for z in s]);best=None;reports=[]
 for t in sorted(set(list(range(1,c['frames'],4))+[c['frames']-1])):
  if t==sf:continue
  X,Y,v,_=observations(c,h,m,s,tracks,t,rd);train=v&~allheld
  static=np.linalg.norm(Y-X,axis=1)<=.003;remaining=train&~static
  for kk in range(3):
   ii=np.flatnonzero(remaining)
   if len(ii)<8:break
   z=A.robust(X[ii],Y[ii],2026+kk)
   if z is None:break
   rr,tt,im,err=z;ids=ii[im];remaining[ids]=False
   if len(ids)<8 or np.median(err[im])>.002:continue
   # Membership uses only the source image/depth, never heldout future residual.
   labs=np.full(len(X),-1,int);labs[ids]=1;labs[train&static]=0;labs[remaining]=0
   hmask=np.zeros(len(X),bool);masks=[];offset=0
   for ci,ss in enumerate(s):
    nt=lengths[ci];trids=np.flatnonzero(~ss['heldout']);full={'uv':ss['full_uv'],'world':ss['full_world']}
    lab=labs[offset:offset+nt][trids]
    if len(trids)<3:
     masks.append(np.zeros(len(ss['world']),bool));offset+=nt;continue
    pl=B.membership(ss['world'],ss['uv'],ss['seeds_world'][trids],ss['seeds_uv'][trids],lab,full)==1
    hh=B.membership(ss['seeds_world'],ss['seeds_uv'],ss['seeds_world'][trids],ss['seeds_uv'][trids],lab,full)==1
    hmask[offset:offset+nt]=hh&ss['heldout'];masks.append(pl);offset+=nt
   hi=np.flatnonzero(hmask&v)
   if len(hi)<3:continue
   he=np.linalg.norm(A.apply(X[hi],rr,tt)-Y[hi],axis=1)
   if np.median(he)>.003 or np.percentile(he,95)>.006:continue
   P=np.concatenate([ss['world'][mm] for ss,mm in zip(s,masks)])
   if not len(P):continue
   P=A.apply(P,rr,tt);ph=A.apply(P,*inverse(m['hand_to_world'][t]))
   robot.render(m['measured_finger_q'][t],np.linalg.inv(m['hand_to_world'][t])@h['agentview/camera_to_world'][t],h['agentview/K'][t])
   mesh,_=A.M.geo.surface(robot,[2,5]);close=((ph>mesh.bounds[0]-.015)&(ph<mesh.bounds[1]+.015)).all(1)
   near=0;mingap=None
   if close.any():
    dd,_=A.M.geo.distance(mesh,ph[close]);near=int((dd<.01).sum());mingap=float(dd.min()*1000)
   reports.append(dict(frame=t,cluster=kk,inliers=len(ids),heldout=len(hi),heldout_p95_mm=float(np.percentile(he,95)*1000),near_seeded_surface=near,minimum_gap_mm=mingap))
   score=near
   if near and (best is None or score>best['score']):
    tm=np.zeros(len(X),bool);tm[ids]=True
    best=dict(score=score,frame=t,train=tm,held=hmask,masks=masks,inliers=ids,report=reports[-1])
 return best,reports
def inverse(H):
 r=np.linalg.inv(H[:3,:3]);return r,-r@H[:3,3]
def views(P,h,rd,finger,t):
 hidden=np.zeros(len(P),bool);visible=np.zeros(len(P),bool);conflict=np.zeros(len(P),bool);uvs=[]
 for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
  uv,z=A.proj(P,h[cam+'/K'][t],h[cam+'/camera_to_world'][t]);uvs.append(uv)
  good=np.isfinite(uv).all(1)&(z>0)&(uv[:,0]>=0)&(uv[:,0]<448)&(uv[:,1]>=0)&(uv[:,1]<448)
  p=np.clip(np.rint(np.nan_to_num(uv)).astype(int),0,447);dep=h[cam+'/depth_m'][t][p[:,1],p[:,0]];fd=finger[ci][p[:,1],p[:,0]]
  hidden|=good&np.isfinite(fd)&(fd<z-.001)&(np.abs(dep-fd)<.004)
  visible|=good&(np.abs(dep-z)<=.003)
  conflict|=good&(dep>z+.006)&(fd>z+.006)
 return hidden,visible,conflict,uvs
def main():
 cases=json.loads((D/'cases.json').read_text());robot=A.RobotDepth(448);allsummary=[]
 for c in cases:
  cid=c['id'];out=D/'data'/cid;outp=out/'estimates';outp.mkdir(exist_ok=True);h=h5py.File(c['rgbd'],'r');m=np.load(out/'motion.npz')
  rd=[np.load(out/f'{cam}_robot_depth.npz')['depth'] for cam in c['cameras']]
  models=[];discover=[]
  for sf in c['source_frames']:
   save(D/'stage_state.json',dict(stage='part_discovery',case=cid,source_frame=sf,time=now()))
   s=[dict(np.load(out/f'{cam}_{sf}_source.npz')) for cam in c['cameras']]
   tracks=[dict(np.load(out/f'{cam}_{sf}_tracks.npz')) if (out/f'{cam}_{sf}_tracks.npz').exists() else dict(tracks=np.zeros((c['frames'],len(ss['seeds_uv']),2)),visible=np.zeros((c['frames'],len(ss['seeds_uv'])))) for cam,ss in zip(c['cameras'],s)]
   best,reports=part_discovery(c,h,m,s,tracks,rd,robot,sf);discover.append(dict(source_frame=sf,models=reports,selected=best['report'] if best else None))
   if best is None:continue
   P=np.concatenate([ss['world'][mask] for ss,mask in zip(s,best['masks'])]);rgb=np.concatenate([ss['rgb'][mask] for ss,mask in zip(s,best['masks'])]);uv=np.concatenate([ss['uv'][mask] for ss,mask in zip(s,best['masks'])])
   camid=np.concatenate([np.full(int(mask.sum()),ci) for ci,mask in enumerate(best['masks'])]);nn,nok=normals(P)
   models.append(dict(sf=sf,s=s,tracks=tracks,best=best,P=P,rgb=rgb,uv=uv,cam=camid,normals=nn,normal_ok=nok,active=np.zeros(len(P),bool),anchor=np.zeros_like(P),anchor_q=np.zeros((len(P),2)),anchor_spread=np.zeros(len(P)),previous_fit=None))
   np.savez_compressed(outp/f'source_{sf}.npz',world=P,rgb=rgb,uv=uv,camera=camid,normal=nn,normal_ok=nok,train=best['train'],held=best['held'])
  save(outp/'discovery.json',discover);frames=[]
  for t in range(c['frames']):
   save(D/'stage_state.json',dict(stage='frame_estimation',case=cid,frame=t,total=c['frames'],time=now()))
   if t==0:
    frame=dict(frame=0,counts=dict(direct=0,continued=0,unknown=0,separated=0),reason='t0 robot state unknown',models=[],points=[],cloud=[],finger=[],views=[]);frames.append(frame);save(outp/'frame_000.json',frame);continue
   H=m['hand_to_world'][t];q=m['measured_finger_q'][t];finger=[]
   for cam in c['cameras']:
    robot.render(q,np.linalg.inv(H)@h[cam+'/camera_to_world'][t],h[cam+'/K'][t]);finger.append(A.render_group(robot,[2,5]))
   mesh,_=A.M.geo.surface(robot,[2,5]);ftree=cKDTree(mesh.triangles_center)
   pts=[];cloud=[];stats=[];all_np=[];counts=dict(direct=0,continued=0,unknown=0,separated=0)
   for mi,mod in enumerate(models):
    X,Y,v,_=observations(c,h,m,mod['s'],mod['tracks'],t,rd)
    f=fit_checked(X,Y,v,mod['best']['train'],mod['best']['held'])
    P0=mod['P'];n=len(P0);rec=dict(source_frame=mod['sf'],available=int(v.sum()),localized=bool(f),source_points=n)
    if f is None:
     rec['reason']='insufficient independent same-part mapping; no hand-only propagation';stats.append(rec);continue
    P=A.apply(P0,f['R'],f['t']);ph=A.apply(P,*inverse(H));sp=np.zeros(n)
    for rr,tt in f['loo']:sp=np.maximum(sp,np.linalg.norm(A.apply(P0,rr,tt)-P,axis=1))
    dist=np.full(n,np.inf);closest=np.zeros_like(ph);cos=np.zeros(n)
    near=((ph>mesh.bounds[0]-.01)&(ph<mesh.bounds[1]+.01)).all(1)
    if near.any():
     dist[near],closest[near]=A.M.geo.distance(mesh,ph[near]);fi=ftree.query(closest[near])[1]
     nh=(mod['normals']@f['R'].T)@np.linalg.inv(H[:3,:3]).T
     cos[near]=np.abs(np.sum(nh[near]*mesh.face_normals[fi],axis=1))
    hidden,visible,conflict,uvs=views(P,h,rd,finger,t)
    stable=sp<=.002;geometric=stable&mod['normal_ok']&(cos>=.7)&((dist+sp)<=.003)&~conflict&(hidden|visible)
    delta=np.linalg.norm(ph-mod['anchor'],axis=1);dq=np.max(np.abs(q[None]-mod['anchor_q']),axis=1)
    separated=(dist-sp>.006)|conflict
    # Relative pose from independently estimated object surface, not copied from hand.
    persistence=mod['active']&stable&(delta<=.002+sp+mod['anchor_spread'])&(dq<=.001)&~separated
    direct=geometric&~mod['active'];continued=persistence&~direct
    # Established support can also be refreshed by new geometry after relative changes.
    refresh=geometric&~persistence;direct|=refresh;continued&=~direct
    state=np.zeros(n,np.int8);state[(dist<.01)&~(direct|continued)]=3;state[continued]=2;state[direct]=1
    expired=mod['active']&separated;state[expired]=4
    mod['active'][separated]=False
    mod['active'][direct|continued]=True
    # Do not chain drifting anchors: only re-anchor on a direct observation-supported estimate.
    mod['anchor'][direct]=ph[direct];mod['anchor_q'][direct]=q;mod['anchor_spread'][direct]=sp[direct]
    for name,mask in [('direct',direct),('continued',continued),('unknown',state==3),('separated',expired)]:counts[name]+=int(mask.sum())
    rec.update(direct=int(direct.sum()),continued=int(continued.sum()),unknown=int((state==3).sum()),separated=int(expired.sum()),inliers=len(f['ii']),heldout=len(f['held']),heldout_p95_mm=float(np.percentile(f['held_errors'],95)*1000),loo_p95_mm=float(np.percentile(sp,95)*1000),wrist_train=int((f['ii']>=len(mod['s'][0]['seeds_world'])).sum()))
    stats.append(rec);keep=np.flatnonzero(state>0)
    np.savez_compressed(outp/f'{mod["sf"]}_{t:03d}.npz',world=P,hand=ph,state=state,distance=dist,spread=sp,normal_cos=cos,hidden=hidden,visible=visible,conflict=conflict,relative_delta=delta,opening_delta=dq,fit_R=f['R'],fit_t=f['t'],inlier_ids=f['ii'],heldout_ids=f['held'],heldout_errors=f['held_errors'])
    # Geometry outputs retain every classified source point; display coalesces 1mm cells.
    for j in keep:
     if state[j]==4:continue
     pts.append(dict(p=P[j].tolist(),state=int(state[j]),hidden=bool(hidden[j]),sf=mod['sf'],cam=int(mod['cam'][j]),uv=mod['uv'][j].tolist(),id=int(j),gap_mm=float(dist[j]*1000) if np.isfinite(dist[j]) else None,spread_mm=float(sp[j]*1000)))
    ii=np.linspace(0,n-1,min(n,800)).astype(int)
    cloud.extend(np.c_[P[ii],mod['rgb'][ii]].tolist())
   # duplicates across source frames/views are fused for display, never treated as independent evidence.
   unique={}
   for p in pts:
    k=tuple(np.floor(np.array(p['p'])/.001).astype(int))
    priority={1:3,2:2,3:1}[p['state']]
    if k not in unique or priority>{1:3,2:2,3:1}[unique[k]['state']]:unique[k]=p
   pts=list(unique.values());displaycounts={str(k):sum(p['state']==k for p in pts) for k in [1,2,3]}
   vertices=A.apply(mesh.vertices,H[:3,:3],H[:3,3]);edges=mesh.edges_unique[::max(1,len(mesh.edges_unique)//600)]
   fv=[[vertices[i].tolist(),vertices[j].tolist()] for i,j in edges]
   vv=[dict(camera=cam,K=h[cam+'/K'][t].tolist(),T=h[cam+'/camera_to_world'][t].tolist()) for cam in c['cameras']]
   frame=dict(frame=t,counts=counts,display_counts=displaycounts,models=stats,points=pts,cloud=cloud,finger=fv,views=vv)
   save(outp/f'frame_{t:03d}.json',frame);frames.append({k:frame[k] for k in ['frame','counts','display_counts','models']})
   if t%10==0:print(json.dumps(dict(case=cid,frame=t,counts=counts)),flush=True)
  save(outp/'summary.json',dict(case=cid,frames=frames,source_models=len(models),contact_truth=False))
  allsummary.append(dict(case=cid,frames=c['frames'],localized_sources=len(models),frames_with_support=sum(x['counts']['direct']+x['counts']['continued']>0 for x in frames),frames_with_continuation=sum(x['counts']['continued']>0 for x in frames),t0_unknown=True))
  h.close()
 robot.close();save(D/'summary.json',dict(time=now(),cases=allsummary,scoring_training=False,contact_truth=False));save(D/'stage_state.json',dict(stage='analysis_complete',time=now()))
if __name__=='__main__':
 try:main()
 except Exception as e:
  (D/'analyze_traceback.log').write_text(traceback.format_exc());save(D/'analysis_failure.json',dict(type=type(e).__name__,message=str(e),stack=[dict(file=x.filename,line=x.lineno) for x in traceback.extract_tb(e.__traceback__)]));save(D/'stage_state.json',dict(stage='analysis_failed',time=now()));sys.exit(1)
