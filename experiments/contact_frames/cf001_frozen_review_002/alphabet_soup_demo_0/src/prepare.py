import os,sys,ast,json,gc
os.environ["MUJOCO_GL"]="egl"
os.environ["OMP_NUM_THREADS"]="4"
from common import *
import numpy as np,cv2,h5py,torch,trimesh,mujoco
from scipy.spatial import cKDTree
sys.path.insert(0,str(ROOT/"src/v3"))
from stage2_probe_motion_fingerpads import cloud as flat_cloud
def cloud(d,K,T):
 return flat_cloud(d,K,T).reshape(d.shape[0],d.shape[1],3)
from stage2_robot_only_depth import RobotDepth
def apply(p,T):return p@T[:3,:3].T+T[:3,3]
def rigid(a,b):
 ca=a.mean(0);cb=b.mean(0);u,s,v=np.linalg.svd((a-ca).T@(b-cb));R=v.T@u.T
 if np.linalg.det(R)<0:v[-1]*=-1;R=v.T@u.T
 T=np.eye(4);T[:3,:3]=R;T[:3,3]=cb-R@ca
 return T
def fit(a,b,rng):
 if len(a)<8:return None,None
 best=np.zeros(len(a),bool)
 for _ in range(256):
  j=rng.choice(len(a),3,replace=False)
  if np.linalg.norm(np.cross(a[j[1]]-a[j[0]],a[j[2]]-a[j[0]]))<1e-7:continue
  T=rigid(a[j],b[j]);inside=np.linalg.norm(apply(a,T)-b,axis=1)<.003
  if inside.sum()>best.sum():best=inside
 if best.sum()<8 or best.mean()<.6:return None,best
 T=rigid(a[best],b[best]);er=np.linalg.norm(apply(a[best],T)-b[best],axis=1)
 singular=np.linalg.svd(a[best]-a[best].mean(0),compute_uv=False)
 if np.median(er)>.002 or singular[1]<.001:return None,best
 return T,best
def finger_mesh(robot):
 m=robot.model;d=robot.data;meshes=[]
 for i in range(m.ngeom):
  name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or ""
  if not (name.startswith("finger") and "collision" in name):continue
  if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
   k=int(m.geom_dataid[i]);a=int(m.mesh_vertadr[k]);n=int(m.mesh_vertnum[k]);b=int(m.mesh_faceadr[k]);nf=int(m.mesh_facenum[k])
   mesh=trimesh.Trimesh(vertices=m.mesh_vert[a:a+n].copy(),faces=m.mesh_face[b:b+nf].copy(),process=False)
  elif m.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX:mesh=trimesh.creation.box(extents=2*m.geom_size[i])
  else:continue
  v=np.asarray(mesh.vertices)@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]
  meshes.append(trimesh.Trimesh(vertices=v,faces=mesh.faces,process=False))
 return trimesh.util.concatenate(meshes)
def distance(mesh,p):
 out=[]
 for j in range(0,len(p),64):out.append(trimesh.proximity.closest_point_naive(mesh,p[j:j+64])[1])
 return np.concatenate(out)
def main():
 cfg=json.loads((RUN/"config.json").read_text());rng=np.random.default_rng(cfg["seed"]);ident=cfg["episode"]
 rgbpath=ROOT/cfg["rgbd"];comppath=ROOT/cfg["component"];motionpath=ROOT/cfg["motion"];weightpath=ROOT/cfg["frozen_tracker"]
 refs=[json.loads(l) for l in (ROOT/"data/v3/manifests/original_sequences.jsonl").read_text().splitlines()]
 ref=next(x for x in refs if x["id"]==ident)
 stage("locking_inputs",episode=ident)
 files=[rgbpath,comppath,motionpath,weightpath,ROOT/"src/v3/stage2_robot_only_depth.py",ROOT/"src/v3/stage2_probe_motion_fingerpads.py"]
 lock={str(p):sha(p) for p in files}
 assert lock[str(weightpath)]=="2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"
 source={str(p):sha(p) for p in sorted((RUN/"src").glob("*.py"))};source[str(RUN/"config.json")]=sha(RUN/"config.json")
 save(RUN/"source_locked.json",source)
 with np.load(motionpath) as z:H=z["hand_to_world"].copy();q=z["measured_finger_q"].copy()
 with np.load(comppath) as z:pix=z["pixel"].astype(np.int64);reference=int(z["query_frame"])
 assert reference==0
 with h5py.File(ref["source_path"],"r") as f:
  e=f["data"][ref["source_episode"]]
  # Explicit allowed fields only: no simulator object pose/reward/contact/success.
  raw={k:np.asarray(e[k]) for k in ["actions","obs/ee_pos","obs/ee_ori","obs/gripper_states","obs/joint_states"]}
  rawlocks={k:{"shape":list(v.shape),"sha256":hashlib.sha256(v.tobytes()).hexdigest()} for k,v in raw.items()}
 save(RUN/"input_locked.json",{"files":lock,"official_selected_episode":ref["source_episode"],"official_allowed_datasets":rawlocks})
 with h5py.File(rgbpath,"r") as f:
  g=f["agentview"];video=np.asarray(g["rgb"]);D=np.asarray(g["depth_m"]);K=np.asarray(g["K"]);C=np.asarray(g["camera_to_world"])
 Tn=len(video);assert len(H)==len(q)==Tn
 # Observe initial foreground without hand-dependent transform.
 mask=np.zeros(D[0].shape,np.uint8);u,v=pix.T
 valid=np.isfinite(D[0,v,u])&(D[0,v,u]>.01);mask[v[valid],u[valid]]=1
 fulluv=np.column_stack(np.nonzero(mask)[::-1]).astype(np.int64)
 surf0=cloud(D[0],K[0],C[0])[fulluv[:,1],fulluv[:,0]]
 # 1mm real-source representatives, never averaged or invented.
 _,idx=np.unique(np.floor(surf0/.001).astype(np.int64),axis=0,return_index=True)
 idx=np.sort(idx);surf0=surf0[idx];fulluv=fulluv[idx]
 eroded=cv2.erode(mask,np.ones((3,3),np.uint8))
 candidates=np.column_stack(np.nonzero(eroded)[::-1])
 if len(candidates)<8:candidates=np.column_stack(np.nonzero(mask)[::-1])
 # Spatial farthest-point pixels; foreground only, no contact targets.
 chosen=[0];ds=np.full(len(candidates),np.inf)
 for _ in range(min(cfg["max_tracks"],len(candidates))-1):
  ds=np.minimum(ds,((candidates-candidates[chosen[-1]])**2).sum(1));chosen.append(int(np.argmax(ds)))
 uv0=candidates[chosen].astype(np.float32);ni=len(uv0);assert ni>=8
 sourcexyz=cloud(D[0],K[0],C[0])[uv0[:,1].astype(int),uv0[:,0].astype(int)]
 sourcecolor=video[0,uv0[:,1].astype(int),uv0[:,0].astype(int)]/255.
 heldout=np.zeros(ni,bool);heldout[np.random.default_rng(2026).permutation(ni)[:max(4,ni//4)]]=True
 np.savez_compressed(RUN/"artifacts/reference.npz",surface_xyz=surf0,surface_uv=fulluv,track_uv=uv0,track_xyz=sourcexyz,track_color=sourcecolor,heldout=heldout)
 stage("tracking_full_video",frames=Tn,tracks=ni,reference_sources=len(surf0))
 sys.path.insert(0,str(ROOT/"third_party/co_tracker_010"))
 from cotracker.predictor import CoTrackerPredictor
 torch.manual_seed(2026);torch.cuda.set_per_process_memory_fraction(.18)
 tracker=CoTrackerPredictor(checkpoint=str(weightpath),offline=True,v2=False,window_len=60).eval().cuda()
 vid=torch.from_numpy(video).permute(0,3,1,2).unsqueeze(0).float().cuda()
 queries=torch.from_numpy(np.column_stack([np.zeros(ni),uv0]).astype(np.float32)).unsqueeze(0).cuda()
 with torch.inference_mode():pred,vis=tracker(vid,queries=queries,backward_tracking=False)
 tracks=pred[0].cpu().numpy();visibility=vis[0].cpu().numpy()
 np.savez_compressed(RUN/"artifacts/tracks.npz",tracks=tracks,visibility=visibility)
 del tracker,vid,queries,pred,vis;gc.collect();torch.cuda.empty_cache()
 stage("building_independent_visual_evidence")
 robot=RobotDepth(D.shape[1]);points=np.zeros((Tn,ni,10),np.float32);images=np.zeros((Tn,4,96,96),np.float32)
 transforms=np.full((Tn,4,4),np.nan);y=np.zeros(Tn,np.float32);weight=np.zeros(Tn,np.float32);quality=np.zeros(Tn,np.float32);dmin=np.full(Tn,np.nan);sigma=np.full(Tn,np.nan)
 rows=[];geometric=np.zeros(Tn,bool);counts=np.zeros(Tn,int)
 states=[];statevalid=np.zeros(Tn,bool)
 for t in range(Tn):
  images[t,:3]=cv2.resize(video[t],(96,96)).transpose(2,0,1)/255.
  depth=cv2.resize(np.nan_to_num(D[t]),(96,96),interpolation=cv2.INTER_NEAREST)
  images[t,3]=np.clip(depth,0,2)/2.
  # t aligns to obs t-1; missing frame0 is masked, not fabricated truth.
  rt=max(0,t-1);hp=np.nan_to_num(H[t,:3,3]);rot=np.nan_to_num(H[t,:3,:2]).reshape(-1)
  actual=raw["actions"][rt];joints=raw["obs/joint_states"][rt]
  robotstate=np.r_[hp,rot,np.nan_to_num(q[t]),joints,actual]
  delta=robotstate-states[-1][:len(robotstate)] if t>1 else np.zeros_like(robotstate)
  states.append(np.r_[robotstate,delta].astype(np.float32))
  if t==0 or not np.isfinite(H[t]).all() or not np.isfinite(q[t]).all():
   rows.append({"frame":t,"reason":"unknown_hand_alignment"});continue
  statevalid[t]=True
  rd=robot.render(q[t],np.linalg.inv(H[t])@C[t],K[t])
  uv=np.rint(tracks[t]).astype(np.int64)
  inbounds=(uv[:,0]>=0)&(uv[:,0]<D.shape[2])&(uv[:,1]>=0)&(uv[:,1]<D.shape[1])
  safe=np.clip(uv,[0,0],[D.shape[2]-1,D.shape[1]-1]);uu,vv=safe.T
  dep=D[t,vv,uu];robd=rd[vv,uu]
  ok=inbounds&(visibility[t]>.9)&np.isfinite(dep)&(dep>.01)&((robd>1)|(dep<robd-.002))
  world=cloud(D[t],K[t],C[t])[vv,uu];ok&=np.isfinite(world).all(1)
  hand=apply(np.nan_to_num(world),np.linalg.inv(H[t]))
  points[t,:,:3]=np.where(ok[:,None],hand/.1,0)
  points[t,:,3:6]=sourcecolor
  points[t,:,6:9]=(sourcexyz-sourcexyz.mean(0))/.1
  points[t,:,9]=ok.astype(float)
  fitids=np.flatnonzero(ok&~heldout);testids=np.flatnonzero(ok&heldout);counts[t]=ok.sum()
  rec={"frame":t,"visible_tracks":int(ok.sum()),"fit_support":len(fitids),"heldout_support":len(testids)}
  M,inliers=fit(sourcexyz[fitids],world[fitids],rng)
  if M is None:rec["reason"]="independent_visual_fit_unresolved";rows.append(rec);continue
  ins=fitids[inliers];transforms[t]=M
  err=np.linalg.norm(apply(sourcexyz[testids],M)-world[testids],axis=1)
  rec["inliers"]=len(ins);rec["heldout_median_mm"]=float(np.median(err)*1000) if len(err) else None
  if len(testids)<4:rec["reason"]="heldout_support_missing";rows.append(rec);continue
  rec["heldout_p95_mm"]=float(np.quantile(err,.95)*1000)
  if np.quantile(err,.95)>.010:rec["reason"]="heldout_correspondence_inconsistent";rows.append(rec);continue
  predsurf=apply(surf0,M);spread=[]
  for k in range(min(32,len(ins))):
   sub=np.delete(ins,k);Mk=rigid(sourcexyz[sub],world[sub]);spread.append(np.linalg.norm(apply(surf0,Mk)-predsurf,axis=1))
  sp=float(np.quantile(np.max(spread,axis=0),.95))
  sig=float(np.sqrt(.002**2+np.median(err)**2+sp**2));sigma[t]=sig
  quality[t]=float(inliers.mean()*np.exp(-np.median(err)/.003)*np.exp(-sp/.003))
  mesh=finger_mesh(robot)
  hp=apply(predsurf,np.linalg.inv(H[t]));dist=distance(mesh,hp)
  # Already unique reference 1mm source cells.
  nearest=np.argsort(dist)[:3];dd=float(np.median(dist[nearest]));dmin[t]=dd
  y[t]=float(np.exp(-dd**2/(2*sig**2)));weight[t]=quality[t]
  geometric[t]=quality[t]>=.25 and sig<=.006
  rec.update(reason="geometric_evidence_not_contact_truth",distance_mm=dd*1000,sigma_mm=sig*1000,quality=float(quality[t]),weak_target=float(y[t]),nearest_reference_indices=nearest.tolist())
  rows.append(rec)
  if t%20==0:stage("building_independent_visual_evidence",frame=t,total=Tn)
 robot.close()
 states=np.stack(states);states[~statevalid]=0
 centers=np.full((Tn,3),np.nan)
 for t in range(Tn):
  if np.isfinite(transforms[t]).all():centers[t]=apply(surf0.mean(0,keepdims=True),transforms[t])[0]
 dy=np.zeros((Tn,5),np.float32);dynvalid=np.zeros(Tn,bool)
 for t in range(1,Tn-1):
  if geometric[t] and geometric[t+1]:
   dy[t]=np.r_[centers[t+1]-centers[t],q[t+1]-q[t]]/.01;dynvalid[t]=True
 block=np.minimum(np.arange(Tn)*5//Tn,4)
 train=np.isin(block,[0,2,3]);audit=~train
 train &= statevalid
 for t in range(Tn-1):dynvalid[t]&=train[t] and train[t+1] and block[t]==block[t+1]
 np.savez_compressed(RUN/"artifacts/dataset.npz",images=images,points=points,states=states,target=y,weight=weight,quality=quality,distance=dmin,sigma=sigma,geometry_reliable=geometric,state_valid=statevalid,visual_transforms=transforms,dy=dy,dynvalid=dynvalid,train=train,audit=audit,block=block,visible_counts=counts,frame_time=np.asarray(ref["timestamps_seconds"]))
 save(RUN/"artifacts/frame_evidence.json",rows)
 report={"episode":ident,"frames":Tn,"tracks":ni,"reference_real_surface_sources":len(surf0),"state_dim":states.shape[-1],"fit_frames":int(np.isfinite(transforms).all((1,2)).sum()),"weighted_frames":int((weight>0).sum()),"reliable_frames":int(geometric.sum()),"weak_high_frames":np.flatnonzero((y>=.8)&geometric).tolist(),"weak_low_frames":np.flatnonzero((y<=.2)&geometric).tolist(),"training_frames":int(train.sum()),"audit_frames":int(audit.sum()),"auxiliary_train_frames":int(dynvalid.sum()),"no_true_contact_labels":True,"reference_frame":0,"t0_hand_unknown":True}
 save(RUN/"artifacts/preparation_summary.json",report)
 assert all(sha(p)==h for p,h in lock.items())
 save(RUN/"artifacts/input_integrity.json",{"all_input_hashes_unchanged":True})
 stage("prepared",**report)
if __name__=="__main__":
 try:main()
 except Exception as e:fail(e);raise SystemExit(1)
