import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
import sys,json,time,pathlib,traceback,concurrent.futures,fcntl
import numpy as np
import h5py
import torch
ROOT=pathlib.Path("/mnt/nvme1/drh/D_VLA")
RUN=ROOT/"experiments/lcjp003/r5/joint/v001"
OLD=ROOT/"experiments/lcjp003/r4/joint/v001"
sys.path.insert(0,str(OLD/"src"))
import data as old_data
geo,legacy_supervision=old_data.legacy()
torch.set_num_threads(2)
def atomic(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(".tmp");t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def episode(e,requested=None):
 eid=e["id"];out=RUN/"teacher"/eid;out.mkdir(parents=True,exist_ok=True)
 if (out/"complete.json").exists():return json.loads((out/"complete.json").read_text())
 fi,world,uv,good,views,_=legacy_supervision.episode_tracks(eid)
 world=np.nan_to_num(world).astype(np.float32);good=np.asarray(good,np.float32)
 frames=sorted(int(p.stem) for p in (OLD/"cache"/eid).glob("*.pt"))
 with h5py.File(ROOT/"experiments/contact_joint_object/cache/rgbd"/(eid+".h5"),"r") as h:
  rawfi=np.array(h["frame_indices"]);meta=json.loads(h.attrs["robot_metadata"])
  assert np.array_equal(rawfi,np.arange(e["frames"])), "rgbd_frame_index_mismatch"
  poses=np.array(h["robot/ee_position"],np.float32)
  rots=np.array(h["robot/geom_rotation"][:,meta["geom_names"].index("gripper0_hand_collision")],np.float32)
 with h5py.File(ROOT/e["raw"],"r") as h:
  grips=(np.array(h["data/"+e["episode"]+"/actions"])[:,-1]>0).astype(int)
 assert len(grips)==e["frames"] and len(frames)>0, "registered_episode_length"
 for t in frames[:2]:
  q=torch.load(OLD/"cache"/eid/(str(t)+".pt"),weights_only=False,map_location="cpu")
  assert np.allclose(poses[t],q["inputs"]["ee"].numpy(),atol=1e-6)
  assert np.allclose(rots[t],q["inputs"]["rotation"].numpy(),atol=1e-6)
  assert int(q["labels"]["grip"][0])==int(grips[t]), "grip_convention"
 segment=np.cumsum(np.r_[False,grips[1:]!=grips[:-1]])
 # Independent point-motion response, never the old proximity/max teacher.
 response=np.zeros_like(good);quality=np.zeros_like(good)
 for i in range(1,len(fi)-1):
  a,b,c=map(int,fi[i-1:i+2]);dt=max(b-a,1);dn=max(c-b,1)
  observed=world[i+1];free=world[i]+(world[i]-world[i-1])*(dn/dt)
  rot=rots[c]@rots[b].T
  follow=(world[i]-poses[b])@rot.T+poses[c]
  ef=np.linalg.norm(observed-free,axis=-1);eg=np.linalg.norm(observed-follow,axis=-1)
  ok=good[i-1]*good[i]*good[i+1]
  residual=np.minimum(ef,eg)[ok>0]
  sigma=max(float(np.median(residual)) if len(residual) else 0.,1e-4)
  delta=np.linalg.norm(follow-world[i],axis=-1)
  response[i]=1/(1+np.exp(-np.clip((ef-eg)/sigma,-30,30)))
  quality[i]=ok*delta/(delta+sigma)
 totals={"known_frames":0,"instances":0,"target_response_mass":0.}
 for t in (frames if requested is None else requested):
  p=out/(str(t)+".pt")
  if p.exists():continue
  objs,_=geo.load_objects(e,t)
  base=torch.load(ROOT/"experiments/lcjp003/r2/v2/v001/cache"/eid/(str(t)+".pt"),map_location="cpu",weights_only=False)
  sem=np.zeros((len(objs),2),np.float32)
  cids=np.array(base["candidate_ids"]);v=base["views"].numpy();prior=base["priors"].numpy()
  for j,o in enumerate(objs):
   matches=np.isin(cids,np.array(o["ids"]))
   if matches.any():sem[j]=prior[matches].mean(0)
  ti=np.flatnonzero(fi==t)
  val=np.zeros(world.shape[1],np.float32);conf=val.copy();current=np.zeros_like(val)
  if len(ti):
   k=int(ti[0]);future=np.flatnonzero((fi>=t)&(fi<t+16)&(segment[np.clip(fi,0,len(segment)-1)]==segment[t]))
   if len(future):
    w=quality[future];den=w.sum(0)
    val=(response[future]*w).sum(0)/np.maximum(den,1e-8)
    conf=den/len(future);current=good[k]
   xyz=world[k];pixels=np.rint(np.nan_to_num(uv[k])).astype(int)
  else:
   xyz=np.zeros((world.shape[1],3),np.float32);pixels=np.zeros((world.shape[1],2),int)
  object_mass=np.zeros(len(objs),np.float32)
  for j,o in enumerate(objs):
   member=np.zeros(len(val),bool)
   for camera in range(2):
    ma=o["masks"][camera]
    if ma is None:continue
    inside=(views==camera)&(pixels[:,0]>=0)&(pixels[:,0]<256)&(pixels[:,1]>=0)&(pixels[:,1]<256)
    ix=np.flatnonzero(inside);member[ix]=ma[pixels[ix,1],pixels[ix,0]]
   w=conf*current*member
   object_mass[j]=float((val*w).sum()/max(float(member.sum()),1.))
  if not len(objs):raise RuntimeError("no_teacher_instances")
  # These are teacher distributions; the student must predict its own target.
  logits_primary=8*(sem[:,0]-sem[:,0].mean())+np.log(object_mass+1e-4)
  logits_dest=8*(sem[:,1]-sem[:,1].mean())
  winners=[int(logits_primary.argmax()),int(logits_dest.argmax())]
  area=np.array([sum(int(m.sum()) for m in o["masks"] if m is not None) for o in objs])
  chosen=list(dict.fromkeys(winners+np.argsort(-area).tolist()))[:16]
  masks=np.zeros((2,16,256,256),bool);valid=np.zeros(16,bool)
  for j,oi in enumerate(chosen):
   valid[j]=True
   for c in range(2):
    if objs[oi]["masks"][c] is not None:masks[c,j]=objs[oi]["masks"][c]
  role=np.zeros((2,17),np.float32)
  for roleidx,logits in enumerate([logits_primary,logits_dest]):
   z=logits[chosen];prob=np.exp(z-z.max());prob/=prob.sum();role[roleidx,:len(chosen)]=prob
  confidence=np.array([min(float(object_mass.max()*4),1.),float(np.ptp(sem[:,1]))],np.float32)
  target_membership=np.zeros(len(val),np.float32)
  for j in range(len(chosen)):
   for c in range(2):
    ix=np.flatnonzero((views==c)&(pixels[:,0]>=0)&(pixels[:,0]<256)&(pixels[:,1]>=0)&(pixels[:,1]<256))
    target_membership[ix]+=masks[c,j,pixels[ix,1],pixels[ix,0]]*role[0,j]
  teacher={"teacher_version":"r5_response_mean_v1","instance_masks":torch.from_numpy(masks),
   "instance_valid":torch.from_numpy(valid),"role_distribution":torch.from_numpy(role),
   "role_weight":torch.from_numpy(confidence),"track_xyz":torch.from_numpy(xyz),
   "track_response":torch.from_numpy(val),"track_confidence":torch.from_numpy(conf*current),
   "track_target":torch.from_numpy(np.minimum(target_membership,1)),
   "history_indices":base["history_indices"],"future_in_inputs":False,
   "episode":eid,"frame":t,"segment":int(segment[t]),"split":e["split"]}
  tmp=p.with_suffix(".tmp");torch.save(teacher,tmp);tmp.replace(p)
  totals["known_frames"]+=int(conf.max()>0);totals["instances"]+=len(chosen);totals["target_response_mass"]+=float(object_mass.max())
 result={"episode":eid,"frames":len(frames),"teacher_version":"r5_response_mean_v1",**totals,"completed":time.time()}
 if requested is None:atomic(out/"complete.json",result)
 return result
def main():
 lock=open(RUN/"teacher_prepare.lock","w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 es=json.loads((RUN/"episodes.json").read_text())
 assert all(int(e["episode"].split("_")[-1]) in list(range(8))+[45,46] for e in es)
 atomic(RUN/"prepare_state.json",{"status":"preparing_response_teacher","episodes":len(es),"pid":os.getpid(),"time":time.time()})
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
  priorities=json.loads((RUN/"priority_windows.json").read_text())
  pf={pool.submit(episode,e,priorities[e["id"]]):e["id"] for e in es if e["id"] in priorities}
  count=0
  for f in concurrent.futures.as_completed(pf):
   f.result();count+=1
   atomic(RUN/"prepare_state.json",{"status":"priority_windows","completed_priority_episodes":count,"total_priority_episodes":len(pf),"pid":os.getpid(),"time":time.time()})
  atomic(RUN/"priority_ready.json",{"ready":True,"windows":sum(map(len,priorities.values())),"time":time.time()})
  fut={pool.submit(episode,e):e["id"] for e in es}
  done=[]
  for f in concurrent.futures.as_completed(fut):
   done.append(f.result());atomic(RUN/"prepare_state.json",{"status":"preparing_response_teacher","completed_episodes":len(done),"total_episodes":len(es),"frames":sum(x["frames"] for x in done),"pid":os.getpid(),"time":time.time()})
 atomic(RUN/"teacher_complete.json",{"version":"r5_response_mean_v1","episodes":done,"time":time.time()})
if __name__=="__main__":
 try:main()
 except BaseException as ex:
  atomic(RUN/"prepare_failure.json",{"type":type(ex).__name__,"frames":[{"file":f.filename,"line":f.lineno} for f in traceback.extract_tb(ex.__traceback__)],"time":time.time()})
  raise SystemExit(1)
