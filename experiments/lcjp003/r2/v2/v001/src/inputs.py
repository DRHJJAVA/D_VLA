from common import *
import h5py,collections
from PIL import Image,ImageDraw
class Inputs:
 def __init__(self,expert):
  self.mu,self.sd=setup_normalization();self.expert=expert;self.sam=None;self.prior=None;self.words={}
  self.actions={}
  for e in ROWS:
   with h5py.File(ROOT/e["raw"],"r") as f:self.actions[e["id"]]=f["data/"+e["episode"]+"/actions"][:]
  self.old=ROOT/"experiments/contact_joint_object"
 def instruction(self,e):return e["task"].removesuffix("_demo").replace("_"," ")+"."
 def phrases(self,e):
  s=self.instruction(e).removeprefix("pick up the ").split(" and place it in the ")
  assert len(s)==2
  return s[0],s[1].rstrip(".")
 def word_embedding(self,e):
  if e["task"] not in self.words:
   p=RUN/"cache"/(e["task"]+"_words.pt")
   if p.exists():self.words[e["task"]]=torch.load(p,map_location="cpu",weights_only=False)
   else:
    self.words[e["task"]]=text_words(self.expert,self.instruction(e));atomic_torch(p.relative_to(RUN),self.words[e["task"]])
  return self.words[e["task"]]
 def load_sam(self):
  if self.sam is not None:return
  g=load_defs(V1/"segmentation_coverage.py")
  t=ast.parse((V1/"segmentation_coverage.py").read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=="main")
  setup=next(n for n in f.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=="model" for v in n.targets))
  exec(compile(ast.Module(body=[setup],type_ignores=[]),"<sam32-existing-config>","exec"),g)
  self.sam=g["SAM2AutomaticMaskGenerator"](g["model"],points_per_side=32,points_per_batch=64)
 def load_prior(self):
  if self.prior is not None:return
  g=load_defs(V1/"v1_check.py")
  t=ast.parse((V1/"v1_check.py").read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=="main")
  nodes=[n for n in f.body if 30<=n.lineno<=39]
  exec(compile(ast.Module(body=nodes,type_ignores=[]),"<frozen-region-language-prior>","exec"),g)
  self.prior=g["query"]
 def cache_frame(self,e,t):
  rng=rng_state()
  try:return self._cache_frame(e,t)
  finally:restore_rng(rng)
 def _cache_frame(self,e,t):
  eid=e["id"];d=RUN/"cache"/eid;d.mkdir(exist_ok=True)
  p=d/(str(t)+".pt")
  if p.exists():return torch.load(p,map_location="cpu",weights_only=False)
  self.load_sam();started=__import__("time").perf_counter()
  masks=[];meta=[];images=[]
  with h5py.File(self.old/"cache/rgbd"/(eid+".h5"),"r") as f:
   for vi,cam in enumerate(["agentview","robot0_eye_in_hand"]):
    rgb=f[cam+"/rgb"][t];images.append(rgb)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    prev=torch.backends.cudnn.allow_tf32;torch.backends.cudnn.allow_tf32=True
    try:
     with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):result=self.sam.generate(rgb)
    finally:torch.backends.cudnn.allow_tf32=prev
    # Retain baseline proposals, without semantic filtering or fallback boxes.
    for a in result:
     masks.append(a["segmentation"].astype(bool));meta.append({"view":vi,"area":int(a["area"]),"iou":float(a["predicted_iou"]),"stability":float(a["stability_score"])})
  obs=d/"observations.h5"
  original_file=h5py.File
  with original_file(self.old/"cache/observations"/(eid+".h5"),"r") as old,original_file(self.old/"cache/rgbd"/(eid+".h5"),"r") as rgb,original_file(obs,"a") as out:
   name="frames/"+str(t)
   if name in out:del out[name]
   f=out.create_group(name)
   for v in range(2):f.create_dataset("features_"+str(v),data=old[name+"/features_"+str(v)][:])
   from data import backproject
   for k,(mask,m) in enumerate(zip(masks,meta)):
    view=m["view"];cam=["agentview","robot0_eye_in_hand"][view];uv=np.argwhere(mask)[:,::-1]
    dep=rgb[cam+"/depth_m"][t];K=rgb[cam+"/K"][t];C=rgb[cam+"/camera_to_world"][t]
    # Surface extraction itself is delegated to the corrected V0 geometry adapter.
    xyz=backproject(dep,K,C,uv)
    o=f.create_group("object_"+str(k));o.create_dataset("mask_"+str(view),data=np.packbits(mask))
    o.create_dataset("uv",data=uv);o.create_dataset("camera",data=np.full(len(uv),view,dtype=np.int8))
    o.create_dataset("xyz",data=xyz);o.create_dataset("normal",data=np.zeros_like(xyz));o.create_dataset("desc",data=np.zeros(384,np.float32))
  import data as p0
  ds=p0.P0Dataset.__new__(p0.P0Dataset)
  ds.root=ROOT;ds.old=self.old;ds.rows={eid:e};ds.actions={eid:self.actions[eid]};ds.read_log=[];ds._cache=collections.OrderedDict()
  ds.instructions={eid:self.instruction(e)};ds.action_mean=self.mu;ds.action_std=self.sd
  ds.input_protocol_version="p0_surface_visibility_v3";ds.adapter_sha256=sha(ROOT/"experiments/lcjp003/src/data.py")
  factory=p0.P0Dataset
  expected=str(self.old/"cache/observations"/(eid+".h5"))
  def redirected(path,*a,**kw):
   if str(path)==expected:path=obs
   assert not a or a[0]=="r","raw files are read-only"
   return original_file(path,*a,**kw)
  p0.P0Dataset=lambda *a,**kw:ds;h5py.File=redirected
  try:obj,h,target,grip,idx=legacy()["make_data"](e,t,self.mu,self.sd)
  finally:p0.P0Dataset=factory;h5py.File=original_file
  assert len(obj)>0,"No valid physical candidate; preserve failure and do not synthesize an object"
  x=np.stack([a[1] for a in obj]);valid=np.stack([a[2] for a in obj]);encoded_ids=[int(a[0]) for a in obj];ids=[k//2 for k in encoded_ids]
  assert all(k%2==meta[k//2]["view"] for k in encoded_ids)
  # Real previous actions only: the label at t is never part of the input.
  previous=np.zeros((16,7),np.float32)
  for j,k in enumerate(idx):
   if k>0:previous[j]=self.actions[eid][int(k)-1]
  h=np.concatenate([h,previous],-1)
  self.load_prior();entries=[]
  for k in ids:
   m=masks[k];vi=meta[k]["view"];full=Image.fromarray(images[vi]);ys,xs=np.where(m)
   box=(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1))
   marked=full.copy();ImageDraw.Draw(marked).rectangle(box,outline=(255,0,0),width=2)
   crop=full.crop(box)
   for phrase in self.phrases(e):entries.append((marked,crop,phrase))
  with torch.no_grad():prior=np.asarray(self.prior(entries),np.float32).reshape(len(ids),2)
  future,fvalid=self.future_targets(e,t,x,valid,ids,masks,meta)
  dct={"x":torch.from_numpy(x),"valid":torch.from_numpy(valid),"h":torch.from_numpy(h).float(),"words":self.word_embedding(e),"target":torch.tensor(target,dtype=torch.float32),"grip":torch.tensor(grip,dtype=torch.long),"priors":torch.tensor(prior),"future":torch.tensor(future),"future_valid":torch.tensor(fvalid),"candidate_ids":ids,"views":torch.tensor([meta[k]["view"] for k in ids]),"episode":eid,"frame":t,"history_indices":np.asarray(idx).tolist(),"input_protocol":"R2_grid32_current_plus_causal_history","cache_seconds":__import__("time").perf_counter()-started}
  atomic_torch(p.relative_to(RUN),dct)
  return dct
 def future_targets(self,e,t,x,valid,ids,masks,meta):
  from data import backproject
  C,P=x.shape[:2];out=np.zeros((C,P,3,387),np.float32);ok=np.zeros((C,P,3),bool)
  eid=e["id"];cams=["agentview","robot0_eye_in_hand"]
  centers=(x[...,384:387]*valid[...,None]).sum(2)/valid.sum(2).clip(1)[...,None]
  with h5py.File(self.old/"cache/observations"/(eid+".h5"),"r") as obs,h5py.File(self.old/"cache/rgbd"/(eid+".h5"),"r") as rgb:
   names=json.loads(rgb.attrs["robot_metadata"])["body_names"];ei=names.index("gripper0_eef")
   origin=rgb["robot/body_position"][t,ei];rot=rgb["robot/body_rotation"][t,ei]
   T=len(rgb["frame_indices"])
   for ci,k in enumerate(ids):
    v=meta[k]["view"];cam=cams[v];track=obs["tracks/"+str(v)]
    uv=track["uv"][t];u=np.rint(uv[:,0]).astype(int).clip(0,255);vv=np.rint(uv[:,1]).astype(int).clip(0,255)
    eligible=track["valid"][t]&track["visible"][t]&masks[k][vv,u]&np.isfinite(uv).all(-1)
    ix=np.where(eligible)[0]
    if not len(ix):continue
    current=backproject(rgb[cam+"/depth_m"][t],rgb[cam+"/K"][t],rgb[cam+"/camera_to_world"][t],np.stack([u[ix],vv[ix]],-1))
    local=(current-origin)@rot
    nearest=np.linalg.norm(centers[ci,:,None]-local[None],axis=-1).argmin(-1)
    # Targets come from observed correspondence, never object pose/contact truth.
    for oi,dt in enumerate(CONFIG["future_offsets"]):
     end=t+dt
     if end>=T:continue
     uv2=track["uv"][end,ix];u2=np.rint(uv2[:,0]).astype(int).clip(0,255);v2=np.rint(uv2[:,1]).astype(int).clip(0,255)
     good=track["valid"][end,ix]&track["visible"][end,ix]&np.isfinite(uv2).all(-1)
     later=backproject(rgb[cam+"/depth_m"][end],rgb[cam+"/K"][end],rgb[cam+"/camera_to_world"][end],np.stack([u2,v2],-1))
     desc=obs["frames/"+str(end)+"/features_"+str(v)][:].astype(np.float32)[v2//8,u2//8]
     desc0=obs["frames/"+str(t)+"/features_"+str(v)][:].astype(np.float32)[vv[ix]//8,u[ix]//8]
     target=np.concatenate([((later-current)@rot)/.05,(desc-desc0)],-1)
     finite=np.isfinite(target).all(-1)
     for pi,ni in enumerate(nearest):
      if valid[ci,pi].any() and good[ni] and finite[ni]:
       out[ci,pi,oi]=target[ni];ok[ci,pi,oi]=True
  return out,ok
