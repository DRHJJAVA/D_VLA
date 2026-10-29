from pathlib import Path
import os,sys,json,hashlib,importlib.util,traceback
os.environ['MUJOCO_GL']='egl'
import numpy as np,h5py
from scipy.stats import norm
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/stage3_temporal_support_v3';B=R/'experiments/contact_preference/stage3_pilot_v2'
sys.path.insert(0,str(R))
from src.v3.stage2_robot_only_depth import RobotDepth
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def apply(T,p):return p@T[:3,:3].T+T[:3,3]
def category(p,n,u):
 s=np.stack([p,n,u]);v=np.argmax(s,axis=0);v[(s==s.max(0)).sum(0)>1]=2;return v
def evidence(z,models):
 ps=[];ns=[]
 for m in models:
  sig=z['sigma_base_m'] if m['error_model']=='base' else z['sigma_proxy_m']
  lc=norm.logpdf(z['signed_distance_m'],0,sig)
  ls=b.logsep(z['signed_distance_m'],sig,m['mu'],m['family'])
  d=np.tanh((lc-ls)/2);ps.append(np.maximum(d,0));ns.append(np.maximum(-d,0))
 return np.min(ps,0),np.min(ns,0)
def visibility(xyz,H,q,h,j,robot):
 out=[];records=[]
 for cam in ['agentview','robot0_eye_in_hand']:
  g=h[cam];K=g['K'][j].astype(float);C=g['camera_to_world'][j].astype(float);dep=g['depth_m'][j].astype(float)
  rd=robot.render(q,np.linalg.solve(H,C),K)
  p=apply(np.linalg.inv(C),xyz);uv=(p@K.T);uv=uv[:,:2]/uv[:,2:3]
  finite=np.isfinite(uv).all(1)&(p[:,2]>0)
  ij=np.rint(np.where(np.isfinite(uv),uv,-100)).astype(int)
  valid=finite&(ij[:,0]>=0)&(ij[:,0]<448)&(ij[:,1]>=0)&(ij[:,1]<448)
  dd=np.full(len(p),np.nan);rr=dd.copy();dd[valid]=dep[ij[valid,1],ij[valid,0]];rr[valid]=rd[ij[valid,1],ij[valid,0]]
  valid&=np.isfinite(dd)&(dd>0)&(dd<5)
  delta=p[:,2]-dd;cl=np.full(len(p),4,np.int8)
  cl[valid&(abs(delta)<=.003)]=0
  behind=valid&(delta>.003);robot_occ=behind&np.isfinite(rr)&(rr>0)&(rr<5)&(abs(rr-dd)<=.003)&(rr<p[:,2]-.003)
  cl[behind]=2;cl[robot_occ]=1;cl[valid&(delta<-.003)]=3
  out.append(cl);records.append((K,C,dd,rr,delta))
 out=np.array(out)
 combined=np.full(len(xyz),4,np.int8)
 combined[(out==2).any(0)]=2
 combined[(out==1).any(0)&~(out==2).any(0)]=1
 combined[(out==0).any(0)]=0
 combined[(out==3).any(0)]=3
 return out,combined,records
def main():
 global b
 sp=importlib.util.spec_from_file_location('base_support',B/'src/build_supports.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
 cfg=json.loads((O/'config.json').read_text());cases=cfg['cases'];models=json.loads((B/'fitted_models.json').read_text())
 inputs=[B/'fitted_models.json',B/'src/build_supports.py',R/'src/v3/stage2_robot_only_depth.py']
 for c in cases:
  inputs += [B/'points_evidence'/f"{c['key']}_frame{t}.npz" for t in c['frames']]
  inputs += [R/'data/v3/trainval_rgbd_448'/c['asset']/(c['demo']+'.h5'),R/'experiments/region_pose/iter_003/evidence/motion'/f"{c['asset']}_{c['demo']}.npz"]
 lock={str(p):sha(p) for p in inputs};save(O/'input_locked.json',lock)
 save(O/'source_locked.json',{str(O/'src/run.py'):sha(O/'src/run.py'),str(O/'config.json'):sha(O/'config.json')})
 save(O/'stage_state.json',{'time':now(),'status':'running'})
 robot=RobotDepth(448);summ=[]
 for ci,c in enumerate(cases):
  mm=[m for m in models if m['target_episode']==ci]
  assert all(ci not in m['training_episode_ids'] for m in mm)
  rows=[];prev=None;prevP=None;framesout=[]
  with h5py.File(R/'data/v3/trainval_rgbd_448'/c['asset']/(c['demo']+'.h5'),'r') as h:
   indexes=np.asarray(h['frame_indices']);motion=dict(np.load(R/'experiments/region_pose/iter_003/evidence/motion'/f"{c['asset']}_{c['demo']}.npz"))
   for t in c['frames']:
    z=dict(np.load(B/'points_evidence'/f"{c['key']}_frame{t}.npz"));n=len(z['query_xyz'])
    if prev is not None:assert np.array_equal(prev['query_xyz'],z['query_xyz']) and np.array_equal(prev['query_pixel'],z['query_pixel'])
    j=int(np.where(indexes==t)[0][0]);H=z['hand_to_world'];assert np.allclose(H,motion['hand_to_world'][t],atol=1e-6)
    vc,vis,details=visibility(z['mapped_world_xyz'],H,motion['measured_finger_q'][t],h,j,robot)
    gp,gn=evidence(z,mm)
    scope=z['excluded_mounting_nearest'];penetration=z['deep_penetration'];opposing=z['opposing_normals']
    # Unknown is not evidence of contact at another place.
    bad=scope|penetration|np.isin(vis,[2,3,4])
    posbad=bad|~opposing
    gp[posbad]=0;gn[bad]=0
    motion_support=np.zeros(n);continuity=np.zeros(n);eh=np.full(n,np.nan);e0=eh.copy()
    if prev is not None:
     observed=z['mapped_world_xyz'];transport=apply(H@np.linalg.inv(prev['hand_to_world']),prev['mapped_world_xyz'])
     eh=np.sum((observed-transport)**2,1);e0=np.sum((observed-prev['mapped_world_xyz'])**2,1)
     drift=np.sum((z['mapped_hand_xyz']-prev['mapped_hand_xyz'])**2,1)
     ms=[];cs=[]
     for key in ['sigma_base_m','sigma_proxy_m']:
      vv=z[key]**2+prev[key]**2
      ms.append(np.maximum(np.tanh((e0-eh)/(4*vv)),0))
      cs.append(np.exp(-drift/(2*vv)))
     motion_support=np.min(ms,0);continuity=np.min(cs,0)
    seed=np.minimum(gp,motion_support)
    carry=np.zeros(n) if prev is None else np.minimum(np.minimum(prevP,gp),continuity)
    carry[posbad]=0;seed[posbad]=0
    P=np.maximum(seed,carry);N=gn;U=1-P-N
    assert np.min(U)>-1e-10
    U=np.maximum(U,0);cat=category(P,N,U);cat[scope]=3
    Gcat=category(gp,gn,1-gp-gn);Gcat[scope]=3
    basecat=category(z['positive'],z['separation'],z['unknown']);basecat[scope]=3
    oc=(vis==1);green=(cat==0)
    unchanged=np.zeros(n,bool) if prevP is None else (prevP>0)&oc&(carry>0)
    checks={'mass_sum_max_error':float(np.max(abs(P+N+U-1))),'scope_positive':int((P[scope]>0).sum()),'penetration_positive':int((P[penetration]>0).sum()),'front_conflict_positive':int((P[vis==3]>0).sum()),'other_occlusion_positive':int((P[vis==2]>0).sum()),'nonopposing_positive':int((P[~opposing]>0).sum())}
    assert all(v==0 for k,v in checks.items() if k!='mass_sum_max_error')
    row={'frame':t,'n':n,'baseline_counts':np.bincount(basecat,minlength=4).tolist(),'instant_geometry_counts':np.bincount(Gcat,minlength=4).tolist(),'new_counts':np.bincount(cat,minlength=4).tolist(),'visibility_counts':np.bincount(vis,minlength=5).tolist(),'green_robot_occluded':int((green&oc).sum()),'green_visible':int((green&(vis==0)).sum()),'baseline_green_robot_occluded':int(((basecat==0)&oc).sum()),'carry_robot_occluded_nonzero':int(unchanged.sum()),'carry_dominates_seed':int((carry>seed+1e-12).sum()),'positive_max':float(P.max()),'motion_support_median':float(np.median(motion_support)),'checks':checks}
    out=dict(query_xyz=z['query_xyz'],query_pixel=z['query_pixel'],query_rgb=z['query_rgb'],query_source_index=z['query_source_index'],mapped_hand_xyz=z['mapped_hand_xyz'],mapped_world_xyz=z['mapped_world_xyz'],Q=z['query_to_current_hand'],H=H,positive=P,separation=N,unknown=U,category=cat,baseline_positive=z['positive'],baseline_separation=z['separation'],baseline_unknown=z['unknown'],baseline_category=basecat,geometry_positive=gp,geometry_separation=gn,geometry_category=Gcat,motion_support=motion_support,continuity=continuity,seed=seed,carry=carry,visibility=vis,camera_visibility=vc,hand_model_squared_error=eh,static_model_squared_error=e0,excluded_scope=scope,penetration=penetration,opposing=opposing,distance_m=z['distance_m'],sigma_base_m=z['sigma_base_m'],sigma_proxy_m=z['sigma_proxy_m'])
    np.savez_compressed(O/'points'/f"{c['key']}_frame{t}.npz",**out)
    framesout.append(out);rows.append(row);prev=z;prevP=P
  ps=np.array([a['positive'] for a in framesout]);ns=np.array([a['separation'] for a in framesout]);us=np.array([a['unknown'] for a in framesout])
  np.savez_compressed(O/'points'/f"event_c{ci}.npz",frames=c['frames'],positive=ps,separation=ns,unknown=us,mean_positive=ps.mean(0),mean_separation=ns.mean(0),mean_unknown=us.mean(0),observed_positive_endpoints=np.sum([a['category']==0 for a in framesout],0))
  summ.append(dict(**c,rows=rows));save(O/'summary_partial.json',summ)
 robot.close()
 mismatches=[p for p,h in lock.items() if sha(p)!=h];assert not mismatches
 save(O/'summary.json',{'time':now(),'cases':summ,'training':False,'model_fit':False,'sparse_endpoints_only':True,'causal_among_held_endpoints':True,'physical_contact_truth':False})
 save(O/'audits/input_integrity.json',{'time':now(),'files':len(lock),'mismatches':mismatches})
 save(O/'stage_state.json',{'time':now(),'status':'computed_pending_audit'})
 print('completed')
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(O/'failure.json',{'time':now(),'type':type(e).__name__,'message':str(e),'stack':[(x.filename,x.lineno) for x in traceback.extract_tb(e.__traceback__)]})
  traceback.print_exc(file=(O/'failure.log').open('w'))
