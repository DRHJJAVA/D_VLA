import os,sys,json,traceback,importlib.util
from pathlib import Path
os.environ['MUJOCO_GL']='egl'
import numpy as np,h5py
from scipy.spatial import Delaunay,cKDTree
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/contact_preference/drawer_precontact_multiview_v1';C=R/'experiments/contact_frames/cf017/cases/drawer_demo_0'
sys.path.insert(0,str(R))
sp=importlib.util.spec_from_file_location('old_analysis',R/'experiments/contact_preference/drawer_hidden_surface_v1/src/analyze_v2.py');A=importlib.util.module_from_spec(sp);sp.loader.exec_module(A)
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def membership(Q,QUV, X,XUV, labels, full):
 out=np.full(len(Q),-1,int)
 if len(X)<3:return out
 tri=Delaunay(XUV);si=tri.find_simplex(QUV);vix=tri.simplices
 lv=labels[vix];good=(lv[:,0]>=0)&(lv[:,0]==lv[:,1])&(lv[:,0]==lv[:,2])
 lookup=np.full((448,448,3),np.nan);uv=full['uv'].astype(int);lookup[uv[:,1],uv[:,0]]=full['world'];edgeok={}
 for ti in np.flatnonzero(good):
  ii=vix[ti]
  for aa,bb in [(ii[0],ii[1]),(ii[1],ii[2]),(ii[2],ii[0])]:
   edge=tuple(sorted([int(aa),int(bb)]))
   if edge not in edgeok:
    n=int(np.ceil(np.max(np.abs(XUV[aa]-XUV[bb]))))+1;w=np.linspace(0,1,n);pix=np.rint(XUV[aa][None]*(1-w[:,None])+XUV[bb][None]*w[:,None]).astype(int)
    real=lookup[pix[:,1],pix[:,0]];expected=X[aa][None]*(1-w[:,None])+X[bb][None]*w[:,None]
    edgeok[edge]=bool(np.isfinite(real).all() and np.max(np.linalg.norm(real-expected,axis=1))<=.002)
   if not edgeok[edge]:good[ti]=False;break
 ix=np.flatnonzero(si>=0);ix=ix[good[si[ix]]]
 if len(ix):
  tt=tri.transform[si[ix]];b=np.einsum('ijk,ik->ij',tt[:,:2],QUV[ix]-tt[:,2]);b=np.c_[b,1-b.sum(1)]
  expect=np.sum(X[vix[si[ix]]]*b[:,:,None],axis=1);ok=np.linalg.norm(expect-Q[ix],axis=1)<=.002
  out[ix[ok]]=lv[si[ix[ok]],0]
 return out
def main():
 cfg=json.loads((D/'config.json').read_text());cases=json.loads((D/'cases.json').read_text());mot=np.load(C/'motion.npz');robot=A.RobotDepth(448)
 h=h5py.File(R/'experiments/contact_frames/cf017/rgbd/drawer_demo_0.h5','r');results=[]
 for sf in cfg['source_frames']:
  cc=[c for c in cases if c['source_frame']==sf];sd=[np.load(D/'data'/(c['key']+'_source.npz')) for c in cc];tracks=[np.load(D/'data'/(c['key']+'_tracks.npz')) for c in cc]
  for t in cfg['target_frames']:
   save(D/'stage_state.json',{'time':now(),'stage':'analyzing','source_frame':sf,'target_frame':t})
   H=mot['hand_to_world'][t];q=mot['measured_finger_q'][t];X=[];Y=[];train_source=[];valids=[];ends=[]
   for ci,(c,s,tr) in enumerate(zip(cc,sd,tracks)):
    cam=c['camera'];K=h[cam+'/K'][t];T=h[cam+'/camera_to_world'][t];dep=h[cam+'/depth_m'][t];rd=robot.render(q,np.linalg.inv(H)@T,K)
    uv=np.rint(tr['tracks'][t-sf]).astype(int);inside=(uv[:,0]>=0)&(uv[:,0]<448)&(uv[:,1]>=0)&(uv[:,1]<448);uvc=np.clip(uv,0,447);u,v=uvc.T
    z=dep[v,u];good=inside&(tr['visibility'][t-sf]>.9)&(z>0)&np.isfinite(z)&(z<rd[v,u]-.002)
    yy=A.back(dep,K,T,uvc);ii=np.flatnonzero(good&~s['heldout'])
    X.append(s['seeds_world'][ii]);Y.append(yy[ii]);train_source.extend([(ci,int(i)) for i in ii]);valids.append(good);ends.append(yy)
   X=np.concatenate(X);Y=np.concatenate(Y);train_source=np.array(train_source);lab=np.full(len(X),-1,int)
   static=np.linalg.norm(Y-X,axis=1)<=.003;lab[static]=0;models={0:(np.eye(3),np.zeros(3),np.flatnonzero(static))}
   for k in range(1,4):
    rem=np.flatnonzero(lab<0)
    if len(rem)<8:break
    ans=A.robust(X[rem],Y[rem],2026+k)
    if ans is None:break
    rr,tt,im,err=ans;ii=rem[im]
    if len(ii)<8 or np.median(err[im])>.002:break
    lab[ii]=k;models[k]=(rr,tt,ii)
   # All memberships are source-only; neither heldout endpoint nor its residual selects membership.
   sl=[];hl=[];his=[]
   for ci,s in enumerate(sd):
    pos=np.flatnonzero(train_source[:,0]==ci);ii=train_source[pos,1];hi=np.flatnonzero(s['heldout'])
    sl.append(membership(s['world'],s['uv'],s['seeds_world'][ii],s['seeds_uv'][ii],lab[pos],s))
    hl.append(membership(s['seeds_world'][hi],s['seeds_uv'][hi],s['seeds_world'][ii],s['seeds_uv'][ii],lab[pos],s));his.append(hi)
   reports={};loofits={}
   for k,(rr,tt,ii) in models.items():
    errors=[];viewchecks=[]
    for ci,s in enumerate(sd):
     hi=his[ci];hh=hi[(hl[ci]==k)&valids[ci][hi]];e=np.linalg.norm(A.apply(s['seeds_world'][hh],rr,tt)-ends[ci][hh],axis=1);errors.extend(e.tolist())
     viewchecks.append({'camera':cc[ci]['camera'],'assigned':int((hl[ci]==k).sum()),'available':len(hh),'median_mm':float(np.median(e)*1000) if len(e) else None,'p95_mm':float(np.percentile(e,95)*1000) if len(e) else None})
    e=np.array(errors);valid=len(e)>=3 and np.median(e)<=.003 and np.percentile(e,95)<=.006
    reports[k]={'cluster':k,'train_inliers':len(ii),'heldout_available':len(e),'heldout_median_mm':float(np.median(e)*1000) if len(e) else None,'heldout_p95_mm':float(np.percentile(e,95)*1000) if len(e) else None,'pass':bool(valid),'by_camera':viewchecks}
    loofits[k]=[]
    if k>0:
     for omit in range(len(ii)):
      jj=np.delete(ii,omit);loofits[k].append(A.fit(X[jj],Y[jj]))
   # Current finger geometry and two camera depth observations.
   viewdata=[]
   for cam in cfg['cameras']:
    K=h[cam+'/K'][t];T=h[cam+'/camera_to_world'][t];dep=h[cam+'/depth_m'][t]
    robot.render(q,np.linalg.inv(H)@T,K);fd=A.render_group(robot,[2,5]);viewdata.append((K,T,dep,fd))
   mesh,_=A.M.geo.surface(robot,[2,5]);mtree=cKDTree(np.asarray(mesh.vertices))
   for ci,(c,s) in enumerate(zip(cc,sd)):
    S=s['world'];n=len(S);mp=np.full_like(S,np.nan);spread=np.full(n,np.nan);supported=np.zeros(n,bool)
    for k,(rr,tt,ii) in models.items():
     ix=np.flatnonzero(sl[ci]==k);mp[ix]=A.apply(S[ix],rr,tt);sh=np.zeros(len(ix))
     for rj,tj in loofits[k]:sh=np.maximum(sh,np.linalg.norm(A.apply(S[ix],rj,tj)-mp[ix],axis=1))
     spread[ix]=sh;supported[ix]=reports[k]['pass']&(sh<=.002)
    finite=np.isfinite(mp).all(1);ph=np.full_like(mp,np.nan);ph[finite]=np.linalg.solve(H[:3,:3],(mp[finite]-H[:3,3]).T).T;dist=np.full(n,np.nan)
    # Exact distance only near finger mesh bbox; farther points cannot qualify.
    if finite.any():
     lo=np.min(mesh.vertices,axis=0)-.01;hi=np.max(mesh.vertices,axis=0)+.01;close=finite&np.all((ph>=lo)&(ph<=hi),axis=1)
     dist[finite]=np.inf
     if close.any():dist[close]=A.M.geo.distance(mesh,ph[close])[0]
    hidden=np.zeros((2,n),bool);visible=np.zeros_like(hidden);conflict=np.zeros_like(hidden);uvs=[]
    for vi,(K,T,dep,fd) in enumerate(viewdata):
     pp=np.zeros((n,2));zz=np.zeros(n);pp[finite],zz[finite]=A.proj(mp[finite],K,T);pi=np.rint(pp).astype(int)
     ok=finite&(zz>0)&(pi[:,0]>=0)&(pi[:,0]<448)&(pi[:,1]>=0)&(pi[:,1]<448);pc=np.clip(pi,0,447);u,v=pc.T;real=dep[v,u];finger=fd[v,u]
     hidden[vi]=ok&(finger>0)&(finger<2)&(zz>finger+.001)&(np.abs(real-finger)<=.003)
     visible[vi]=ok&(real>0)&(np.abs(real-zz)<=.003)
     conflict[vi]=ok&(real>zz+.006)&(real>0);uvs.append(pp)
    localized=supported&hidden.any(0)&~conflict.any(0);compatible=localized&(dist<=.003)
    # A diagnostic mask has no authority to become contact supervision.
    hypothesis=finite&hidden.any(0)&(dist<=.003)
    key=c['key']+'_to_'+str(t)
    np.savez_compressed(D/'data'/(key+'.npz'),source_world=S,source_uv=s['uv'],mapped_world=mp,mapped_hand=ph,source_cluster=sl[ci],supported=supported,loo_spread=spread,distance_visual_finger=dist,hidden_by_finger=hidden,visible=visible,free_space_conflict=conflict,localized=localized,contact_compatible=compatible,train_source=train_source,train_cluster=lab,heldout_ids=his[ci],heldout_cluster=hl[ci],models_R=np.stack([x[0] for x in models.values()]),models_t=np.stack([x[1] for x in models.values()]))
    keep=np.flatnonzero(localized|hypothesis);show=np.flatnonzero(compatible)
    row={'key':key,'source_frame':sf,'source_camera':c['camera'],'target_frame':t,'source_points':n,'assigned':int(finite.sum()),'supported':int(supported.sum()),'localized_hidden':int(localized.sum()),'contact_compatible':int(compatible.sum()),'hypothesis_hidden_near':int(hypothesis.sum()),'models':list(reports.values()),'keep_ids':keep.tolist(),'contact_ids':show.tolist(),'source_uv':s['uv'][keep].tolist(),'target_uv':[p[keep].tolist() for p in uvs],'keep_contact':compatible[keep].tolist(),'keep_supported':localized[keep].tolist(),'keep_gap_mm':np.where(np.isfinite(dist[keep]),dist[keep]*1000,-1).tolist(),'all_source_uv_sample':s['uv'][np.linspace(0,n-1,min(2000,n)).astype(int)].tolist()}
    save(D/'data'/(key+'.json'),row);results.append(row)
    print(json.dumps({k:row[k] for k in ['key','source_points','assigned','supported','localized_hidden','contact_compatible','hypothesis_hidden_near']}),flush=True)
 h.close();robot.close();save(D/'summary.json',{'time':now(),'cameras':cfg['cameras'],'source_frames':cfg['source_frames'],'target_frames':cfg['target_frames'],'rows':results,'scoring_training':False,'contact_truth':False});save(D/'stage_state.json',{'time':now(),'stage':'analyzed','rows':len(results)})
if __name__=='__main__':
 try:main()
 except Exception as e:
  (D/'analysis_failure.log').write_text(traceback.format_exc());save(D/'failure.json',{'time':now(),'type':type(e).__name__,'message':str(e),'stack':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]});sys.exit(1)
