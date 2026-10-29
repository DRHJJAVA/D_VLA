import os,sys,json,traceback,importlib.util
from pathlib import Path
os.environ['MUJOCO_GL']='egl'
import numpy as np,h5py
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/contact_preference/drawer_precontact_multiview_v1';C=R/'experiments/contact_frames/cf017/cases/drawer_demo_0';sys.path.insert(0,str(R))
sp=importlib.util.spec_from_file_location('near_analysis',D/'src/analyze.py');B=importlib.util.module_from_spec(sp);sp.loader.exec_module(B);A=B.A
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 cfg=json.loads((D/'config.json').read_text());cases=json.loads((D/'cases.json').read_text());h=h5py.File(R/'experiments/contact_frames/cf017/rgbd/drawer_demo_0.h5');mot=np.load(C/'motion.npz');robot=A.RobotDepth(448);rows=[]
 for sf in cfg['source_frames']:
  cc=[c for c in cases if c['source_frame']==sf];sd=[np.load(D/'data'/(c['key']+'_source.npz')) for c in cc];tr=[np.load(D/'data'/(c['key']+'_tracks.npz')) for c in cc];owners=[[],[]];owned=[np.zeros(len(s['world']),bool) for s in sd]
  for t in cfg['target_frames']:
   H=mot['hand_to_world'][t];q=mot['measured_finger_q'][t];yy=[]
   for ci,c in enumerate(cc):
    cam=c['camera'];uv=np.clip(np.rint(tr[ci]['tracks'][t-sf]).astype(int),0,447);yy.append(A.back(h[cam+'/depth_m'][t],h[cam+'/K'][t],h[cam+'/camera_to_world'][t],uv))
   common=np.load(D/'data'/(cc[0]['key']+'_to_'+str(t)+'.npz'));train=common['train_source'];lab=common['train_cluster']
   X=np.array([sd[ci]['seeds_world'][idx] for ci,idx in train]);Y=np.array([yy[ci][idx] for ci,idx in train])
   models={i:(r,tt) for i,(r,tt) in enumerate(zip(common['models_R'],common['models_t']))}
   report=json.loads((D/'data'/(cc[0]['key']+'_to_'+str(t)+'.json')).read_text())['models'];passing={m['cluster'] for m in report if m['pass']}
   loofits={}
   for k in passing:
    loofits[k]=[]
    if k>0:
     ids=np.flatnonzero(lab==k)
     for omit in range(len(ids)):
      ii=np.delete(ids,omit);loofits[k].append(A.fit(X[ii],Y[ii]))
   vd=[]
   for cam in cfg['cameras']:
    K=h[cam+'/K'][t];T=h[cam+'/camera_to_world'][t];dep=h[cam+'/depth_m'][t];robot.render(q,np.linalg.inv(H)@T,K);fd=A.render_group(robot,[2,5]);vd.append((K,T,dep,fd))
   mesh,_=A.M.geo.surface(robot,[2,5])
   for ci,(c,s) in enumerate(zip(cc,sd)):
    key=c['key']+'_to_'+str(t);base=np.load(D/'data'/(key+'.npz'));S=s['world'];n=len(S)
    cl=base['source_cluster'].copy();mp=base['mapped_world'].copy();spread=base['loo_spread'].copy();sup=base['supported'].copy();transferred=np.zeros(n,bool);conflicting=np.zeros(n,bool);transfer_reports=[]
    for own in owners[ci]:
     targetlabels=[int(l) for pair,l in zip(train,lab) if tuple(pair) in own['tracks'] and int(l) in passing]
     labs,cnts=np.unique(targetlabels,return_counts=True)
     accept=len(labs)==1 and labs[0]>0 and cnts[0]>=8
     transfer_reports.append({'owner_frame':own['frame'],'owner_cluster':own['cluster'],'matched_passed_clusters':labs.tolist(),'counts':cnts.tolist(),'accepted_identity':bool(accept)})
     if not accept:continue
     k=int(labs[0]);ix=own['ids'];bad=(cl[ix]>=0)&(cl[ix]!=k);conflicting[ix[bad]]=True
     ix=ix[~bad];rr,tt=models[k];mp[ix]=A.apply(S[ix],rr,tt);sh=np.zeros(len(ix))
     for rj,tj in loofits[k]:sh=np.maximum(sh,np.linalg.norm(A.apply(S[ix],rj,tj)-mp[ix],axis=1))
     cl[ix]=k;spread[ix]=sh;sup[ix]=sh<=.002;transferred[ix]=True
    sup[conflicting]=False
    finite=np.isfinite(mp).all(1);ph=np.full_like(mp,np.nan);ph[finite]=np.linalg.solve(H[:3,:3],(mp[finite]-H[:3,3]).T).T;dist=np.full(n,np.nan)
    lo=np.min(mesh.vertices,axis=0)-.01;hi=np.max(mesh.vertices,axis=0)+.01;close=finite&np.all((ph>=lo)&(ph<=hi),axis=1);dist[finite]=np.inf
    if close.any():dist[close]=A.M.geo.distance(mesh,ph[close])[0]
    hidden=np.zeros((2,n),bool);visible=np.zeros_like(hidden);free=np.zeros_like(hidden);uvs=[]
    for vi,(K,T,dep,fd) in enumerate(vd):
     pp=np.zeros((n,2));zz=np.zeros(n);pp[finite],zz[finite]=A.proj(mp[finite],K,T);pix=np.rint(pp).astype(int);ok=finite&(zz>0)&(pix[:,0]>=0)&(pix[:,0]<448)&(pix[:,1]>=0)&(pix[:,1]<448);pc=np.clip(pix,0,447);u,v=pc.T;real=dep[v,u];finger=fd[v,u]
     hidden[vi]=ok&(finger>0)&(finger<2)&(zz>finger+.001)&(np.abs(real-finger)<=.003);visible[vi]=ok&(real>0)&(np.abs(real-zz)<=.003);free[vi]=ok&(real>zz+.006)&(real>0);uvs.append(pp)
    localized=sup&hidden.any(0)&~free.any(0);contact=localized&(dist<=.003);hyp=finite&hidden.any(0)&(dist<=.003)
    np.savez_compressed(D/'identity_v2'/(key+'.npz'),source_world=S,source_uv=s['uv'],mapped_world=mp,mapped_hand=ph,source_cluster=cl,supported=sup,loo_spread=spread,distance_visual_finger=dist,hidden_by_finger=hidden,visible=visible,free_space_conflict=free,localized=localized,contact_compatible=contact,identity_transferred=transferred,identity_conflict=conflicting)
    keep=np.flatnonzero(localized|hyp);row={'key':key,'source_frame':sf,'source_camera':c['camera'],'target_frame':t,'source_points':n,'assigned':int(finite.sum()),'supported':int(sup.sum()),'localized_hidden':int(localized.sum()),'contact_compatible':int(contact.sum()),'hypothesis_hidden_near':int(hyp.sum()),'transferred_points':int(transferred.sum()),'identity_conflict':int(conflicting.sum()),'transfer_reports':transfer_reports,'models':report,'keep_ids':keep.tolist(),'contact_ids':np.flatnonzero(contact).tolist(),'source_uv':s['uv'][keep].tolist(),'target_uv':[p[keep].tolist() for p in uvs],'keep_contact':contact[keep].tolist(),'keep_supported':localized[keep].tolist(),'keep_gap_mm':np.where(np.isfinite(dist[keep]),dist[keep]*1000,-1).tolist(),'all_source_uv_sample':s['uv'][np.linspace(0,n-1,min(2000,n)).astype(int)].tolist()}
    save(D/'identity_v2'/(key+'.json'),row);rows.append(row);print(json.dumps({k:row[k] for k in ['key','localized_hidden','contact_compatible','transferred_points','identity_conflict']}),flush=True)
    # Establish new ownership only from direct current-source memberships, not propagated ones.
    for k in sorted(passing):
     if k==0:continue
     ix=np.flatnonzero((base['source_cluster']==k)&base['supported']&~owned[ci])
     if not len(ix) or np.median(np.linalg.norm(base['mapped_world'][ix]-S[ix],axis=1))<=.005:continue
     ownertracks=set(map(tuple,train[lab==k]));owners[ci].append({'frame':t,'cluster':k,'ids':ix,'tracks':ownertracks});owned[ci][ix]=True
 h.close();robot.close();save(D/'summary_identity_v2.json',{'time':now(),'rows':rows,'scoring_training':False});save(D/'stage_state.json',{'time':now(),'stage':'identity_analyzed','rows':len(rows)})
if __name__=='__main__':
 try:main()
 except Exception as e:
  (D/'identity_failure.log').write_text(traceback.format_exc());save(D/'failure.json',{'time':now(),'type':type(e).__name__,'message':str(e),'stack':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]});sys.exit(1)
