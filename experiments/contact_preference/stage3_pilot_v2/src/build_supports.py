import os
os.environ["MUJOCO_GL"]="egl"
from pathlib import Path
import json,hashlib,importlib.util,traceback,sys
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from scipy.spatial import cKDTree,ConvexHull
from scipy.special import expit,logsumexp
from scipy.stats import norm,exponnorm,skewnorm
from scipy.optimize import minimize
R=Path("/mnt/nvme1/drh/D_VLA");O=R/"experiments/contact_preference/stage3_pilot_v2";OLD=R/"experiments/contact_preference/stage3_pilot_v1"
def now():return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
sp=importlib.util.spec_from_file_location("old",OLD/"src/build_supports_v2.py");m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def normal_est(p,cam,k):
 ix=cKDTree(p).query(p,k=k)[1];v=p[ix]-p[ix].mean(1,keepdims=True);cov=np.einsum("nki,nkj->nij",v,v)/k
 ev,evec=np.linalg.eigh(cov);n=evec[:,:,0];n[(n*(cam-p)).sum(1)<0]*=-1
 valid=ev[:,1]>np.finfo(float).eps
 return n,valid
def distances(mesh,p):
 ds=[];pts=[];ns=[]
 center=np.asarray(mesh.vertices).mean(0)
 for i in range(0,len(p),48):
  q,d,tri=m.trimesh.proximity.closest_point_naive(mesh,p[i:i+48])
  n=mesh.face_normals[tri].copy();n[(n*(q-center)).sum(1)<0]*=-1
  ds.append(d);pts.append(q);ns.append(n)
 return np.concatenate(ds),np.concatenate(pts),np.concatenate(ns)
def logsep(d,s,mu,family):
 if family=="exp":return exponnorm.logpdf(d,mu/s,loc=0,scale=s)
 return skewnorm.logpdf(d,mu/s,loc=0,scale=np.sqrt(s*s+mu*mu))
def fitmodel(ds,ss,ww,family):
 d=np.concatenate(ds);s=np.concatenate(ss);w=np.concatenate(ww);w=w/w.sum()
 lc=norm.logpdf(d,0,s)
 def obj(x):
  pi=expit(x[0]);mu=np.exp(x[1]);ls=logsep(d,s,mu,family)
  ll=np.logaddexp(np.log(pi)+lc,np.log1p(-pi)+ls)
  if not np.isfinite(ll).all():return 1e100
  return -float(np.sum(w*ll))
 mu0=max(float(np.sum(w*np.maximum(d,0))),float(np.median(s)))
 results=[]
 for pi in [.01,.2,.8]:
  z=minimize(obj,[np.log(pi/(1-pi)),np.log(mu0)],method="L-BFGS-B",bounds=[(-12,12),(-13.8,2.3)])
  results.append(z)
 z=min(results,key=lambda a:a.fun)
 return dict(pi=float(expit(z.x[0])),mu=float(np.exp(z.x[1])),family=family,objective=float(z.fun),optimizer_success=bool(z.success))
def posterior(d,s,model):
 pi=model["pi"];lc=norm.logpdf(d,0,s)+np.log(pi);ls=logsep(d,s,model["mu"],model["family"])+np.log1p(-pi)
 return np.exp(lc-np.logaddexp(lc,ls))
def train_groups(features,field="B"):
 allout=[];allmodels=[]
 for target in range(3):
  train=[i for i in range(3) if i!=target];scores=[];mods=[]
  for group in [[train[0]],train,[train[1]]]:
   for err in ["base","proxy"]:
    for family in ["exp","half"]:
     ds=[];ss=[];ww=[]
     for j in group:
      f=features[j][field];ix=f["fit_index"]
      ds.append(f["d"][ix]);ss.append(f[err][ix]);ww.append(np.ones(len(ix))/len(ix)/len(group))
     model=fitmodel(ds,ss,ww,family)
     model.update(target_episode=target,training_episode_ids=group,error_model=err,variant=field)
     vals=posterior(features[target][field]["d"],features[target][field][err],model)
     scores.append(vals);mods.append(model)
  a=np.stack(scores);allout.append((a.min(0),a.max(0),a));allmodels+=mods
 return allout,allmodels
def main():
 cfg=json.loads((O/"config.json").read_text());S=json.loads((OLD/"summary.json").read_text())
 inputs=[O/"config.json",OLD/"src/build_supports_v2.py",OLD/"completed_locked.json",OLD/"robot_assets_postread_locked.json"]
 for ca in S["cases"]:
  inputs += [R/"experiments/region_pose/iter_024/train"/(ca["key"]+".npz")]
  inputs += [OLD/a["data"] for a in ca["rows"]]
 for p in json.loads((OLD/"robot_assets_postread_locked.json").read_text()):inputs.append(Path(p))
 locked={str(p):sha(p) for p in inputs};save(O/"input_locked.json",locked)
 save(O/"stage_state.json",dict(stage="geometry",time=now(),pid=os.getpid()))
 sp=importlib.util.spec_from_file_location("robot",R/"src/v3/stage2_robot_only_depth.py");rob=importlib.util.module_from_spec(sp);sp.loader.exec_module(rob)
 import mujoco
 robot=rob.RobotDepth(32);features=[];geometry=[]
 try:
  for ci,ca in enumerate(S["cases"]):
   z0=np.load(R/"experiments/region_pose/iter_024/train"/(ca["key"]+".npz"));xyz=z0["query_xyz"];cam=z0["query_camera_to_world"][:3,3]
   nn12,valid12=normal_est(xyz,cam,12);nn24,valid24=normal_est(xyz,cam,24)
   _,unique=np.unique(np.floor(xyz/.001).astype(np.int64),axis=0,return_index=True)
   frames=[]
   for row in ca["rows"]:
    t=row["frame"];old=np.load(OLD/row["data"]);hp=old["mapped_hand_xyz"];Q=old["query_to_current_hand"]
    robot.data.qpos[:]=z0["finger_q_"+str(t)];mujoco.mj_forward(robot.model,robot.data)
    dd=[];cp=[];normals=[];ins=[]
    for g in [3,4,6,7]:
     mesh,_=m.surface(robot,[g]);d,q,n=distances(mesh,hp);dd.append(d);cp.append(q);normals.append(n)
     h=ConvexHull(mesh.vertices).equations;ins.append(np.all(hp@h[:,:3].T+h[:,3]<=0,axis=1))
    ds=np.stack(dd);nearest=np.argmin(ds,axis=0);ar=np.arange(len(xyz));d=ds[nearest,ar];closest=np.stack(cp)[nearest,ar];ng=np.stack(normals)[nearest,ar]
    within=np.any(ins,axis=0);signed=np.where(within,-d,d)
    excluded,_=m.surface(robot,[1]);db,_=m.distance(excluded,hp)
    visual,_=m.surface(robot,[2,5]);dv,_=m.distance(visual,hp)
    sigma=old["sigma_base_m"];sigma_proxy=sigma+np.abs(d-dv)
    mount=db<d
    n12=nn12@Q[:3,:3].T;n12/=np.linalg.norm(n12,axis=1)[:,None]
    n24=nn24@Q[:3,:3].T;n24/=np.linalg.norm(n24,axis=1)[:,None]
    dot12=(n12*ng).sum(1);dot24=(n24*ng).sum(1)
    opposing=(dot12<0)&(dot24<0)&valid12&valid24
    deep=signed < -3*sigma_proxy
    mesh,_=m.surface(robot,[3,4,6,7])
    frames.append(dict(t=t,oldpath=row["data"],signed=signed,d=d,base=sigma,proxy=sigma_proxy,mount=mount,deep=deep,
      opposing=opposing,dot12=dot12,dot24=dot24,nearest_geom=np.array([3,4,6,7])[nearest],closest=closest,
      robot_vertices=mesh.vertices,robot_faces=mesh.faces))
   # Shared-frame uncertainty is not divided by sqrt(T).
   b=np.stack([f["signed"] for f in frames]);mean=b.mean(0);spread=b.std(0)
   sb=np.sqrt(np.mean(np.stack([f["base"] for f in frames])**2,axis=0)+spread**2)
   sx=np.sqrt(np.mean(np.stack([f["proxy"] for f in frames])**2,axis=0)+spread**2)
   eligible=~np.any([f["mount"]|f["deep"] for f in frames],axis=0)
   fitix=unique[eligible[unique]]
   assert len(fitix)>16
   oldz=[np.load(OLD/a["data"]) for a in ca["rows"]]
   aa=np.stack([z["distance_m"] for z in oldz]);amean=aa.mean(0);asp=aa.std(0)
   abase=np.sqrt(np.mean(np.stack([z["sigma_base_m"] for z in oldz])**2,axis=0)+asp**2)
   aproxy=np.sqrt(np.mean(np.stack([z["sigma_proxy_m"] for z in oldz])**2,axis=0)+asp**2)
   features.append(dict(B=dict(d=mean,base=sb,proxy=sx,fit_index=fitix),A=dict(d=amean,base=abase,proxy=aproxy,fit_index=unique)))
   geometry.append(frames)
   np.savez_compressed(O/"points"/("event_features_c"+str(ci)+".npz"),mean_signed_gap=mean,base=sb,proxy=sx,temporal_std=spread,
      fit_index=fitix,source_unique_1mm=unique,normals_query_k12=nn12,normals_query_k24=nn24,query_xyz=xyz)
 finally:robot.close()
 save(O/"stage_state.json",dict(stage="fitting_evidence_mixture",time=now(),pid=os.getpid()))
 output,models=train_groups(features,"B");ablation,modelsA=train_groups(features,"A")
 save(O/"fitted_models.json",models);save(O/"ablation_models.json",modelsA)
 rowsout=[]
 for ci,(ca,fs) in enumerate(zip(S["cases"],geometry)):
  rawL,rawU,post=output[ci];rows=[]
  for fi,(row,f) in enumerate(zip(ca["rows"],fs)):
   old=np.load(OLD/row["data"]);L=rawL.copy();U=rawU.copy()
   L[~f["opposing"]]=0
   # A point is only positive where this instant also agrees with the event's physical-contact model.
   inst=np.stack([posterior(f["signed"],f[model["error_model"]],model) for model in models if model["target_episode"]==ci])
   L=np.minimum(L,inst.min(0));U=np.maximum(U,inst.max(0))
   scope=f["mount"];conflict=f["deep"]
   L[scope|conflict]=0;U[scope|conflict]=1
   supports=np.column_stack([L,1-U,U-L]);cat=np.argmax(supports,axis=1)
   ties=(np.isclose(supports,supports.max(1)[:,None],atol=1e-12,rtol=0).sum(1)>1);cat[ties]=2
   display=cat.copy();display[scope]=3;display[conflict&~scope]=4
   fn=ca["key"]+"_frame"+str(f["t"])+".npz"
   np.savez_compressed(O/"points"/fn,query_xyz=old["query_xyz"],query_pixel=old["query_pixel"],query_rgb=old["query_rgb"],
     query_source_index=old["query_source_index"],mapped_hand_xyz=old["mapped_hand_xyz"],mapped_world_xyz=old["mapped_world_xyz"],
     query_to_current_hand=old["query_to_current_hand"],hand_to_world=old["hand_to_world"],currently_depth_consistent=old["currently_depth_consistent"],
     positive=L,separation=1-U,unknown=U-L,display_class=display,support_class=cat,
     posterior_sensitivity=post,instant_posterior_sensitivity=inst,raw_lower=rawL,raw_upper=rawU,
     distance_m=f["d"],signed_distance_m=f["signed"],sigma_base_m=f["base"],sigma_proxy_m=f["proxy"],
     closest_gripper_hand=f["closest"],nearest_gripper_geom=f["nearest_geom"],excluded_mounting_nearest=scope,
     deep_penetration=conflict,opposing_normals=f["opposing"],normal_dot_k12=f["dot12"],normal_dot_k24=f["dot24"],
     gripper_vertices=f["robot_vertices"],gripper_faces=f["robot_faces"],
     old_positive=old["positive"],old_display_class=old["display_class"],
     ablation_statistical_lower=ablation[ci][0],ablation_statistical_upper=ablation[ci][1])
   assert np.isfinite(supports).all() and np.all(supports>=0) and np.allclose(supports.sum(1),1)
   assert np.all(L[scope|conflict|~f["opposing"]]==0)
   rows.append(dict(frame=f["t"],query_frame=row["query_frame"],n=len(L),
    count={str(i):int((display==i).sum()) for i in range(5)},positive_max=float(L.max()),
    positive_mean=float(L.mean()),old_positive_max=float(old["positive"].max()),
    statistics_only_max=float(ablation[ci][0].max()),
    old_unknown_nearest_mount=int(((old["display_class"]==2)&scope).sum()),
    data="points/"+fn,current_depth_consistent=int(old["currently_depth_consistent"].sum()),
    occluded_or_depth_inconsistent=int((~old["currently_depth_consistent"]).sum()),
    current_visible_green=int(((display==0)&old["currently_depth_consistent"]).sum()),
    occluded_green=int(((display==0)&~old["currently_depth_consistent"]).sum())))
  rowsout.append(dict(**{k:v for k,v in ca.items() if k!="rows"},rows=rows,
    fit_1mm_sources=int(len(features[ci]["B"]["fit_index"]))))
 # Heldout synthetic tests: not a physical accuracy benchmark.
 tests=[]
 for ci in range(3):
  ms=[z for z in models if z["target_episode"]==ci];s0=float(np.median(features[ci]["B"]["base"]))
  d=np.array([0,1,3,10])*s0
  pp=np.stack([posterior(d,np.ones(4)*s0,z) for z in ms])
  tests.append(dict(case=ci,sigma_m=s0,gap_sigma=[0,1,3,10],lower=pp.min(0).tolist(),upper=pp.max(0).tolist()))
  assert np.all(pp[:,0]>=pp[:,-1])
 after={p:sha(Path(p)) for p in locked};assert after==locked
 save(O/"summary.json",dict(time=now(),cases=rowsout,synthetic=tests,model_count=len(models),
  no_scoring_training=True,prior_learned_from_other_episode=True,calibrated_contact_probability=False))
 save(O/"stage_state.json",dict(stage="computed",time=now()))
 print(json.dumps(dict(stage="computed",cases=[dict(key=c["key"],counts=[r["count"] for r in c["rows"]]) for c in rowsout])))
if __name__=="__main__":
 try:main()
 except Exception as e:
  (O/"failure.log").write_text(traceback.format_exc())
  save(O/"failure.json",dict(type=type(e).__name__,message=str(e),time=now(),
   locations=[dict(file=f.filename,line=f.lineno,name=f.name) for f in traceback.extract_tb(e.__traceback__)]))
  save(O/"stage_state.json",dict(stage="failed",type=type(e).__name__,time=now()))
  print(json.dumps(dict(error=type(e).__name__)));sys.exit(1)
