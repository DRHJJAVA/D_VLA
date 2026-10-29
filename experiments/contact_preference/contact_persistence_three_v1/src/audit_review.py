import os,sys,json,hashlib,traceback,importlib.util
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ['MUJOCO_GL']='egl'
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/contact_preference/contact_persistence_three_v1';sys.path.insert(0,str(R))
import numpy as np,h5py
from scipy.spatial import Delaunay
s=importlib.util.spec_from_file_location('current_analysis',D/'src/analyze_v3.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M);A=M.A
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def runs(x):
 out=[]
 for t in x:
  if not out or t!=out[-1][-1]+1:out.append([t,t])
  else:out[-1][-1]=t
 return out
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 cases=json.loads((D/'cases.json').read_text());robot=A.RobotDepth(448);report=[];coord=[];patchcount=0;projection=0.;source_error=0.;transform_error=0.;invalid_support=0;fingerrows=[]
 for c in cases:
  out=D/'data'/c['id'];ed=out/'estimates_v2';h=h5py.File(c['rgbd'],'r');mot=np.load(out/'motion.npz');N=c['frames']
  source={}
  for sf in c['source_frames']:
   pp=ed/f'source_{sf}.npz'
   if not pp.exists():continue
   ss=dict(np.load(pp));tri=[]
   for cam in [0,1]:
    ix=np.flatnonzero(ss['camera']==cam)
    if len(ix)>=3:
     try:
      tt=ix[Delaunay(ss['uv'][ix]).simplices]
      d=np.max(np.stack([np.linalg.norm(ss['world'][tt[:,a]]-ss['world'][tt[:,b]],axis=1) for a,b in [(0,1),(1,2),(2,0)]]),axis=0)
      tt=tt[d<=.004];tri.extend(tt.tolist())
     except Exception:pass
    if len(ix):
     uv=ss['uv'][ix];cc=c['cameras'][cam];P=A.back(h[cc+'/depth_m'][sf],h[cc+'/K'][sf],h[cc+'/camera_to_world'][sf],uv);source_error=max(source_error,float(np.max(np.linalg.norm(P-ss['world'][ix],axis=1))))
     puv,pz=A.proj(P,h[cc+'/K'][sf],h[cc+'/camera_to_world'][sf]);projection=max(projection,float(np.max(np.abs(puv-uv))))
   ss['tri']=np.array(tri,int).reshape(-1,3);source[sf]=ss
  intervals={};sumrows=[];allcontacts=[];geomframes=[];continuedframes=[];hiddenframes=[];maxpoints=0;totalsourcepointframes=0;frdiag=[]
  for t in range(N):
   p=ed/f'frame_{t:03d}.json';j=json.loads(p.read_text());tris=[];basecount=0;framehidden=0;fullcounts={1:0,2:0};auditmodels=[]
   for sf,ss in source.items():
    pp=ed/f'{sf}_{t:03d}.npz'
    if not pp.exists():continue
    z=np.load(pp);P=ss['world']@z['fit_R']+z['fit_t'];H=mot['hand_to_world'][t];hand=np.linalg.solve(H[:3,:3],(P-H[:3,3]).T).T
    transform_error=max(transform_error,float(np.max(np.linalg.norm(P-z['world'],axis=1))),float(np.max(np.linalg.norm(hand-z['hand'],axis=1))))
    state=z['state'];active=(state==1)|(state==2);g=z['geometric'];basecount+=int(g.sum());framehidden+=int((active&z['hidden']).sum())
    invalid_support+=int((active&((z['spread']>.002)|z['conflict'])).sum())
    # Check continuation directly against saved anchor-relative deltas and gripper opening.
    bad=(state==2)&((z['opening_delta']>.001)|(z['distance']-z['spread']>.006))
    invalid_support+=int(bad.sum())
    for idx in np.flatnonzero(active):intervals.setdefault(f'{sf}:{idx}',[]).append(t)
    for k in [1,2]:fullcounts[k]+=int((state==k).sum())
    tt=ss['tri']
    if len(tt):
     keep=np.all(active[tt],axis=1)
     for a,b,cc in tt[keep]:
      # All three vertices are actual observed same-source points; no bridging missing patches.
      st=2 if np.any(state[[a,b,cc]]==2) else 1
      tris.append({'p':P[[a,b,cc]].tolist(),'state':st,'geometric':bool(g[[a,b,cc]].all()),'sf':sf})
   j['triangles']=tris;j['baseline_geometric_source_count']=basecount;j['hidden_supported_source_count']=framehidden
   patchcount+=len(tris)
   # Current observed context in gray only, robot removed; not a contact/identity label.
   context=[];diag=[]
   if t:
    H=mot['hand_to_world'][t];q=mot['measured_finger_q'][t]
    for cam in c['cameras']:
     dep=h[cam+'/depth_m'][t];K=h[cam+'/K'][t];T=h[cam+'/camera_to_world'][t];rd=robot.render(q,np.linalg.inv(H)@T,K)
     vv,uu=np.where((dep>0)&np.isfinite(dep)&(dep<rd-.002));uv=np.c_[uu,vv];P=A.back(dep,K,T,uv);ph=np.linalg.solve(H[:3,:3],(P-H[:3,3]).T).T
     mask=np.linalg.norm(ph-np.array([0,0,.09]),axis=1)<.12;ii=np.flatnonzero(mask)
     if len(ii)>600:ii=ii[np.linspace(0,len(ii)-1,600).astype(int)]
     rgb=h[cam+'/rgb'][t];context.extend(np.c_[P[ii],rgb[vv[ii],uu[ii]]].tolist())
     vals={}
     for name,ids in [('finger',[2,5]),('gripper_body',[0])]:
      mesh,_=A.M.geo.surface(robot,ids);near=((ph>mesh.bounds[0]-.025)&(ph<mesh.bounds[1]+.025)).all(1)
      if near.any():
       dd,_=A.M.geo.distance(mesh,ph[near]);vals[name]={'near3mm':int((dd<.003).sum()),'min_mm':float(dd.min()*1000)}
      else:vals[name]={'near3mm':0,'min_mm':None}
     diag.append({'camera':cam,**vals})
   j['context_cloud']=context;j['current_depth_diagnostic']=diag
   p.write_text(json.dumps(j,ensure_ascii=False,separators=(',',':')))
   if fullcounts[1]+fullcounts[2]:allcontacts.append(t)
   if fullcounts[2]:continuedframes.append(t)
   if basecount:geomframes.append(t)
   if framehidden:hiddenframes.append(t)
   totalsourcepointframes+=fullcounts[1]+fullcounts[2];maxpoints=max(maxpoints,len(j['points']))
   frdiag.append({'frame':t,'views':diag})
   sumrows.append({k:j[k] for k in ['frame','counts','display_counts','models'] if k in j}|{'baseline_geometric_source_count':basecount,'hidden_supported_source_count':framehidden})
  save(ed/'summary.json',dict(case=c['id'],frames=sumrows,source_models=len(source),contact_truth=False))
  duration=[dict(source_id=k,supported_frames=len(v),supported_seconds=len(v)/20,ranges=runs(v)) for k,v in intervals.items()]
  save(ed/'source_contact_durations.json',duration)
  rr=dict(case=c['id'],title=c['title'],frames=N,source_models=len(source),support_ranges=runs(allcontacts),geometry_ranges=runs(geomframes),continuation_ranges=runs(continuedframes),hidden_support_ranges=runs(hiddenframes),support_frames=len(allcontacts),geometry_frames=len(geomframes),continuation_frames=len(continuedframes),max_same_source_supported_seconds=max([x['supported_seconds'] for x in duration],default=0),counting='source point time is per-point; not summed across views/sources; gaps unknown not counted',max_display_points=maxpoints)
  report.append(rr);fingerrows.append(dict(case=c['id'],frames=frdiag));h.close()
 robot.close()
 locks=json.loads((D/'input_locked.json').read_text());mismatch=[p for p,h in locks.items() if sha(p)!=h]
 assert not mismatch and invalid_support==0
 assert transform_error<1e-6 and source_error<1e-6 and projection<1e-5
 save(D/'audits/current_depth_finger_vs_body.json',fingerrows)
 save(D/'audits/completion.json',dict(time=now(),cases=report,input_files=len(locks),hash_mismatches=mismatch,source_world_error_m=source_error,source_projection_error_px=projection,independent_transform_error_m=transform_error,invalid_support=invalid_support,supported_surface_triangles=patchcount,total_frames=sum(c['frames'] for c in cases),image_count=len(list((D/'images').rglob('*.jpg'))),scoring_training=False,human_labels_used=False,reserved_test_access=False,contact_truth=False))
 save(D/'summary_review.json',dict(time=now(),cases=report));save(D/'stage_state.json',dict(stage='numeric_audit_complete',time=now()))
if __name__=='__main__':
 try:main()
 except Exception as e:
  (D/'audit_traceback.log').write_text(traceback.format_exc());save(D/'audit_failure.json',dict(type=type(e).__name__,message=str(e),stack=[dict(file=x.filename,line=x.lineno) for x in traceback.extract_tb(e.__traceback__)]));sys.exit(1)
