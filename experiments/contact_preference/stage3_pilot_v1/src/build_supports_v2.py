import os
os.environ["MUJOCO_GL"]="egl"
import json,hashlib,traceback,sys,importlib.util
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import h5py
import trimesh
from scipy.spatial import ConvexHull
from scipy.special import log_ndtr
R=Path("/mnt/nvme1/drh/D_VLA")
O=R/"experiments/contact_preference/stage3_pilot_v1"
def now():return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def apply(p,T):return p@T[:3,:3].T+T[:3,3]
def rigid(a,b):
 ac=a.mean(0);bc=b.mean(0);u,s,v=np.linalg.svd((a-ac).T@(b-bc));rot=v.T@u.T
 if np.linalg.det(rot)<0:v[-1]*=-1;rot=v.T@u.T
 T=np.eye(4);T[:3,:3]=rot;T[:3,3]=bc-rot@ac;return T
def fit(x,y,lam):
 mu=x.mean(0);sd=x.std(0);sd[sd<1e-9]=1
 X=np.column_stack([np.ones(len(x)),(x-mu)/sd])
 pen=np.eye(X.shape[1])*lam;pen[0,0]=0
 b=np.linalg.solve(X.T@X+pen,X.T@y)
 return mu,sd,b
def pred(m,x):mu,sd,b=m;return np.column_stack([np.ones(len(x)),(x-mu)/sd])@b
def blocks(case):
 z=np.load(R/"experiments/region_pose/iter_022/train/correspondences"/(case["key"]+".npz"));ts=case["frames"];out=[]
 for i in range(2,len(ts)):
  a,b,c=ts[i-2:i+1];ids=[z["track_id_"+str(t)] for t in (a,b,c)]
  common=np.intersect1d(np.intersect1d(ids[0],ids[1]),ids[2])
  if len(common)<4:continue
  ix=[np.array([np.flatnonzero(s==k)[0] for k in common]) for s in ids]
  p=[z["bridge_xyz_"+str(t)][ind] for t,ind in zip((a,b,c),ix)]
  # All jointly observed filtered tracks, without conditioning on RANSAC inliers.
  v=(p[1]-p[0])/(b-a);y=(p[2]-p[1])/(c-b)
  H=z["hand_to_world_"+str(c)]@np.linalg.inv(z["hand_to_world_"+str(b)])
  out.append(dict(v=v,y=y,p=p[1],H=H,dt=c-b,frames=[a,b,c],n=len(common)))
 return out
def features(bs,control=None):
 X0=[];X1=[];Y=[]
 order=list(range(len(bs)))
 if control=="roll":order=np.roll(order,1).tolist()
 if control=="reverse":order=order[::-1]
 for i,b in enumerate(bs):
  other=bs[order[i]]
  h=(apply(b["p"],other["H"])-b["p"])/other["dt"]
  X0.append(b["v"]);X1.append(np.column_stack([b["v"],h]));Y.append(b["y"])
 return np.concatenate(X0),np.concatenate(X1),np.concatenate(Y)
def motion(cases,lams):
 bs=[blocks(c) for c in cases];data=[features(b) for b in bs];out=[]
 for k,c in enumerate(cases):
  train=[j for j in range(len(cases)) if j!=k];chosen=[];errs=[]
  for mi in [0,1]:
   scores=[]
   for lam in lams:
    es=[]
    for j in train:
     jj=[v for v in train if v!=j]
     m=fit(np.concatenate([data[v][mi] for v in jj]),np.concatenate([data[v][2] for v in jj]),lam)
     es.append(float(np.mean((pred(m,data[j][mi])-data[j][2])**2)))
    scores.append(np.mean(es))
   lam=lams[int(np.argmin(scores))];chosen.append(lam)
   m=fit(np.concatenate([data[v][mi] for v in train]),np.concatenate([data[v][2] for v in train]),lam)
   errs.append(float(np.mean((pred(m,data[k][mi])-data[k][2])**2)))
   if mi==1:
    controls={s:float(np.mean((pred(m,features(bs[k],s)[1])-data[k][2])**2)) for s in ["roll","reverse"]}
  basis=bool(len(bs[k])>1 and errs[1]<min(errs[0],*controls.values()))
  out.append(dict(key=c["key"],n_temporal_blocks=len(bs[k]),n_track_transitions=len(data[k][2]),ridge=chosen,
    mse_m2_per_frame2=dict(M0=errs[0],M1=errs[1],**controls),positive_basis=basis,
    note="heldout predictive association only; controls unidentifiable with one temporal block",
    blocks=[dict(frames=b["frames"],n=b["n"]) for b in bs[k]]))
 return out
def surface(robot,ids):
 meshes=[];hulls=[]
 m=robot.model;d=robot.data
 for g in ids:
  T=np.eye(4);T[:3,:3]=d.geom_xmat[g].reshape(3,3);T[:3,3]=d.geom_xpos[g]
  if int(m.geom_type[g])==6:
   mesh=trimesh.creation.box(extents=2*m.geom_size[g])
  elif int(m.geom_type[g])==7:
   ix=int(m.geom_dataid[g]);v0=int(m.mesh_vertadr[ix]);nv=int(m.mesh_vertnum[ix]);f0=int(m.mesh_faceadr[ix]);nf=int(m.mesh_facenum[ix])
   mesh=trimesh.Trimesh(vertices=m.mesh_vert[v0:v0+nv].copy(),faces=m.mesh_face[f0:f0+nf].copy(),process=False)
  else:raise ValueError("unexpected robot geom type")
  mesh.apply_transform(T);meshes.append(mesh)
  # Collision hulls are convex; visual hulls are NOT used as inside tests.
  if ids==[1,3,4,6,7]:hulls.append(ConvexHull(mesh.vertices).equations)
 return trimesh.util.concatenate(meshes),hulls
def distance(mesh,p):
 ds=[];cp=[]
 for j in range(0,len(p),48):
  q,d,_=trimesh.proximity.closest_point_naive(mesh,p[j:j+48]);ds.append(d);cp.append(q)
 return np.concatenate(ds),np.concatenate(cp)
def inside(hulls,p):return np.any(np.stack([np.all(p@h[:,:3].T+h[:,3]<=0,axis=1) for h in hulls]),axis=0)
def post(d,s,w,prior):
 # Stable integral of Gaussian likelihood over positive uniform gap.
 a=d/s;b=(d-w)/s
 # Use survival CDF for positive tail to avoid catastrophic cancellation.
 hi=np.where(b>0,log_ndtr(-b),log_ndtr(a));lo=np.where(b>0,log_ndtr(-a),log_ndtr(b))
 delta=np.minimum(lo-hi,-1e-15)
 logsep=hi+np.log(-np.expm1(delta))-np.log(w)
 logcon=-0.5*(d/s)**2-np.log(s)-0.5*np.log(2*np.pi)
 l=logcon-logsep+np.log(prior/(1-prior))
 return 1/(1+np.exp(-np.clip(l,-700,700)))
def main():
 c=json.loads((O/"config.json").read_text());cases=c["cases"]
 paths=[O/"config.json",R/"src/v3/stage2_robot_only_depth.py"]
 for ca in cases:
  key=ca["key"]
  paths += [R/"experiments/region_pose/iter_024/train"/(key+".npz"),R/"experiments/region_pose/iter_022/train/correspondences"/(key+".npz"),
    R/"data/v3/trainval_rgbd_448"/ca["asset"]/(ca["demo"]+".h5")]
  paths += [R/"experiments/region_pose/iter_024/train_loo"/(key+"_frame"+str(t)+".npz") for t in ca["frames"]]
 before={str(p):sha(p) for p in paths};save(O/"input_locked.json",before)
 save(O/"stage_state.json",dict(stage="computing",time=now(),pid=os.getpid()))
 mot=motion(cases,c["motion_ridge"]);save(O/"motion_comparison.json",mot)
 spec=importlib.util.spec_from_file_location("robot_depth_stage3",R/"src/v3/stage2_robot_only_depth.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 import mujoco
 robot=mod.RobotDepth(32)
 summary=[];projection_max=0;rgb_bad=0;tests={}
 try:
  for ca,mo in zip(cases,mot):
   key=ca["key"];z=np.load(R/"experiments/region_pose/iter_024/train"/(key+".npz"))
   corr=np.load(R/"experiments/region_pose/iter_022/train/correspondences"/(key+".npz"))
   xyz=z["query_xyz"];extent=float(np.linalg.norm(np.ptp(xyz,axis=0)));rows=[]
   hf=h5py.File(R/"data/v3/trainval_rgbd_448"/ca["asset"]/(ca["demo"]+".h5"),"r")
   qframe=int(key.rsplit("query",1)[1])
   for fi,t in enumerate(ca["frames"]):
    suf=str(t);loo=np.load(R/"experiments/region_pose/iter_024/train_loo"/(key+"_frame"+suf+".npz"))
    Q=z["query_to_current_hand_"+suf];H=z["hand_to_world_"+suf];hp=apply(xyz,Q);wp=apply(hp,H)
    robot.data.qpos[:]=z["finger_q_"+suf];mujoco.mj_forward(robot.model,robot.data)
    mesh,hulls=surface(robot,[1,3,4,6,7]);visual,_=surface(robot,[0,2,5])
    d,closest=distance(mesh,hp);dv,_=distance(visual,hp)
    penetrated=inside(hulls,hp)
    # Collision union is a known geometric proxy. Convexifying visual mesh could falsely fill cavities.
    ix=np.flatnonzero(corr["inlier_"+suf]);a=corr["query_xyz_"+suf][ix];b=corr["bridge_xyz_"+suf][ix]
    residual=[]
    for ii in range(len(ix)):
     keep=np.arange(len(ix))!=ii
     if keep.sum()<3:continue
     T=rigid(a[keep],b[keep]);residual.append(np.linalg.norm(apply(a[ii:ii+1],T)[0]-b[ii]))
    resid=float(np.sqrt(np.mean(np.square(residual))))
    cam=hf["agentview"];Tcam=np.asarray(cam["camera_to_world"][qframe] if cam["camera_to_world"].ndim==3 else cam["camera_to_world"])
    K=np.asarray(cam["K"][qframe] if cam["K"].ndim==3 else cam["K"])
    pc=apply(xyz,np.linalg.inv(Tcam.astype(np.float64)));pixel=np.abs(pc[:,2])*np.sqrt(1/K[0,0]**2+1/K[1,1]**2)
    shift=loo["query_max_shift_m"]
    sigma=np.sqrt(resid**2+pixel**2+shift**2)
    proxy=np.abs(d-dv);sigma2=sigma+proxy
    values=[]
    for sig in [sigma,sigma2]:
     for W in [sig,np.full(len(sig),extent)]:
      for prior in c["prior_contact_sensitivity"]:values.append(post(d,sig,W,prior))
    mat=np.stack(values);rawL=mat.min(0);rawU=mat.max(0);L=rawL.copy();U=rawU.copy()
    if not mo["positive_basis"]:L[:]=0
    # Geometric penetration or invalid numerical mapping supplies neither sign.
    invalid=penetrated|~np.isfinite(sigma)|(sigma<=0)
    L[invalid]=0;U[invalid]=1
    mass=np.column_stack([L,1-U,U-L])
    cat=np.argmax(mass,axis=1);ties=(np.isclose(mass,mass.max(1)[:,None],atol=1e-12,rtol=0).sum(1)>1);cat[ties]=2
    # Directly visible = old depth consistency evidence, not ground-truth visibility/contact.
    vis=z["query_visible_at_frames"][fi]
    sid=z["query_pixel"];rgb=z["query_rgb"]
    qrgb=np.asarray(cam["rgb"][qframe])
    # Existing pixel schema is u,v.
    rgb_bad+=int(np.any(qrgb[sid[:,1],sid[:,0]]!=rgb,axis=1).sum())
    reproj=pc@K.T;reproj=reproj[:,:2]/reproj[:,2:3]
    projection_max=max(projection_max,float(np.max(np.abs(reproj-sid))))
    # Save registered continuous supports and provenance without class-only loss labels.
    out=O/"points"/(key+"_frame"+suf+".npz")
    np.savez_compressed(out,query_xyz=xyz,query_pixel=sid,query_rgb=rgb,query_source_index=np.arange(len(xyz)),
      mapped_hand_xyz=hp,mapped_world_xyz=wp,closest_gripper_hand=closest,
      distance_m=d,visual_distance_m=dv,proxy_discrepancy_m=proxy,penetrated=penetrated,
      sigma_base_m=sigma,sigma_proxy_m=sigma2,loo_shift_m=shift,pixel_scale_m=pixel,
      posterior_sensitivity=mat,raw_lower=rawL,raw_upper=rawU,lower=L,upper=U,
      positive=mass[:,0],separation=mass[:,1],unknown=mass[:,2],display_class=cat,
      currently_depth_consistent=vis,gripper_vertices=mesh.vertices,gripper_faces=mesh.faces,
      hand_to_world=H,query_to_current_hand=Q)
    rows.append(dict(frame=t,query_frame=qframe,n=len(xyz),current_depth_consistent=int(vis.sum()),
      occluded_or_depth_inconsistent=int((~vis).sum()),motion_positive_basis=mo["positive_basis"],
      mean_support=dict(positive=float(L.mean()),separation=float((1-U).mean()),unknown=float((U-L).mean())),
      dominant_count=dict(positive=int((cat==0).sum()),separation=int((cat==1).sum()),unknown=int((cat==2).sum())),
      positive_max=float(L.max()),inside_proxy_unknown=int(penetrated.sum()),
      distance_mm_percentiles=np.percentile(d*1000,[0,10,50,90,100]).tolist(),
      sigma_base_mm_percentiles=np.percentile(sigma*1000,[0,50,100]).tolist(),
      proxy_difference_mm_percentiles=np.percentile(proxy*1000,[0,50,100]).tolist(),
      heldout_inlier_rms_mm=resid*1000,n_loo=len(residual),data=str(out.relative_to(O))))
    assert np.isfinite(mass).all() and np.all(mass>=0) and np.allclose(mass.sum(1),1)
    assert np.all(L<=U) and np.all(L[invalid]==0) and np.all(U[invalid]==1)
   hf.close();summary.append(dict(**ca,rows=rows))
 finally:robot.close()
 # Counterexamples are deterministic numerical checks, not empirical physical accuracy.
 sig=np.array([.001]);near=post(np.array([0.]),sig,np.array([.1]),.5)[0]
 far=post(np.array([.02]),sig,np.array([.1]),.5)[0]
 uncertain=post(np.array([.02]),np.array([.02]),np.array([.1]),.5)[0]
 assert far<near and uncertain>far
 tests.update(near_contact_model=float(near),far_contact_model=float(far),far_with_large_error=float(uncertain),
  support_sum_and_unknown_gates=True,query_rgb_mismatch=rgb_bad,query_reprojection_max_px=projection_max,
  finite_all_points=True,missing_motion_lower_zero=True)
 assert rgb_bad==0 and projection_max<1e-6
 after={str(p):sha(p) for p in paths};assert before==after
 save(O/"summary.json",dict(time=now(),cases=summary,motion=mot,scope="3 old training episodes / 12 endpoints, no score network, uncalibrated sensitivity supports"))
 save(O/"audit.json",dict(time=now(),checks=tests,input_hash_unchanged=True,input_count=len(paths),
  negative_semantics="separation at this observed time and pose, not never contactable",
  limitations=["no physical contact ground truth","collision surface approximation","old inlier-conditioned correspondence residual",
    "three episodes insufficient for calibration","all currently sampled episodes are transport","no temporal duration target computed",
    "visual-mesh penetration not tested; visual hull would fill cavities"]))
 save(O/"stage_state.json",dict(stage="computed",time=now()))
 print(json.dumps(dict(stage="computed",cases=[dict(key=s["key"],frames=len(s["rows"])) for s in summary]),ensure_ascii=True))
if __name__=="__main__":
 try:main()
 except Exception as e:
  (O/"failure.log").write_text(traceback.format_exc())
  save(O/"failure.json",dict(time=now(),type=type(e).__name__,message=str(e),locations=[dict(file=f.filename,line=f.lineno,name=f.name) for f in traceback.extract_tb(e.__traceback__)]))
  save(O/"stage_state.json",dict(stage="failed",time=now(),type=type(e).__name__))
  print(json.dumps(dict(stage="failed",type=type(e).__name__),ensure_ascii=True))
  sys.exit(1)
