"""Stage 1 only: state restoration, contact candidates, geometry; no training."""
import os,sys,json,argparse,hashlib
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("LIBERO_CONFIG_PATH","/mnt/nvme1/drh/D_VLA/configs/libero")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,h5py
from restore_audit import OffScreenRenderEnv,libero_utils
from asset_paths import localize_assets
ROOT=Path("/mnt/nvme1/drh/D_VLA")
LIB=Path("/home/drh/project/ECV-VLA/vendor/LIBERO")
OUT=ROOT/"data/v2/stage1"
TASKS=[
("alphabet_soup","libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5",8),
("tomato_sauce","libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5",8),
("porcelain_mug","libero_90/STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy_demo.hdf5",7),
("white_yellow_mug","libero_90/LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5",7)]
def save(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(path)
def resolve(d):
 b=Path(d.attrs["bddl_file_name"]).name
 found=list((LIB/"libero/libero/bddl_files").rglob(b))
 if len(found)!=1:raise ValueError((b,found))
 return found[0]
def restore(env,ep):
 xml,count=localize_assets(libero_utils.postprocess_model_xml(ep.attrs["model_file"],{}),LIB)
 env.reset();env.reset_from_xml_string(xml);env.sim.reset()
 return count
def state(env,ep,t):
 original=np.asarray(ep["states"][t],dtype=np.float64)
 env.sim.set_state_from_flattened(original);env.sim.forward()
 return float(np.max(np.abs(env.sim.get_state().flatten()-original)))
def runs(flags):
 start=None
 for t,v in enumerate(list(flags)+[False]):
  if v and start is None:start=t
  if not v and start is not None:yield start,t-1;start=None
def ids(env,obj):
 m=env.sim.model
 targets=[i for i in range(m.ngeom) if (m.geom_id2name(i) or "").startswith(obj+"_1_")]
 pads=[m.geom_name2id("gripper0_finger"+str(i)+"_pad_collision") for i in (1,2)]
 body=m.body_name2id(obj+"_1_main")
 return targets,pads,body
def mesh_points(env,targets,body,n=80000):
 m=env.sim.model;d=env.sim.data;vertices=[];triangles=[];offset=0
 for g in targets:
  if m.geom_group[g]!=1:continue
  mid=int(m.geom_dataid[g])
  if int(m.geom_type[g])!=7:raise ValueError("Expected visual mesh")
  va=int(m.mesh_vertadr[mid]);vn=int(m.mesh_vertnum[mid]);fa=int(m.mesh_faceadr[mid]);fn=int(m.mesh_facenum[mid])
  v=np.asarray(m.mesh_vert[va:va+vn],float)@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
  v=(v-d.body_xpos[body])@d.body_xmat[body].reshape(3,3)
  faces=np.asarray(m.mesh_face[fa:fa+fn],int)
  vertices.append(v);triangles.append(faces+offset);offset+=len(v)
 v=np.concatenate(vertices);f=np.concatenate(triangles);tri=v[f]
 area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
 rng=np.random.default_rng(20261023);chosen=rng.choice(len(tri),n,p=area/area.sum())
 uv=rng.random((n,2));uv[uv.sum(axis=1)>1]=1-uv[uv.sum(axis=1)>1]
 pts=tri[chosen,0]+uv[:,0,None]*(tri[chosen,1]-tri[chosen,0])+uv[:,1,None]*(tri[chosen,2]-tri[chosen,0])
 return pts,v,f
def inspect_episode(env,ep,obj):
 targets,pads,b=ids(env,obj);m=env.sim.model;d=env.sim.data
 site=m.site_name2id("gripper0_grip_site");rows=[];errors=[]
 for t in range(len(ep["states"])):
  errors.append(state(env,ep,t));op=d.body_xpos[b].copy();oR=d.body_xmat[b].reshape(3,3).copy()
  gp=d.site_xpos[site].copy();gR=d.site_xmat[site].reshape(3,3).copy()
  contacts=[]
  for ci in range(d.ncon):
   c=d.contact[ci];pair=(int(c.geom1),int(c.geom2))
   for side,pad in enumerate(pads):
    if pad in pair and any(g in targets for g in pair):
     contacts.append({"side":side,"world":c.pos.tolist(),"object":((c.pos-op)@oR).tolist(),"distance_m":float(c.dist),"geoms":[m.geom_id2name(g) for g in pair]})
  rows.append({"frame":t,"object_position":op.tolist(),"object_rotation":oR.tolist(),"gripper_position":gp.tolist(),"gripper_rotation":gR.tolist(),
   "relative_position":((op-gp)@gR).tolist(),"relative_rotation":(gR.T@oR).tolist(),
   "pads":[{"position":d.geom_xpos[g].tolist(),"rotation":d.geom_xmat[g].reshape(3,3).tolist(),"halfsize":m.geom_size[g].tolist()} for g in pads],"contacts":contacts})
 candidates=[]
 flags=[{x["side"] for x in r["contacts"]}=={0,1} for r in rows]
 for a,z in runs(flags):
  subset=rows[a:z+1];p=np.array([r["object_position"] for r in subset]);rel=np.array([r["relative_position"] for r in subset])
  lift=float(p[:,2].max()-p[0,2]);drift=float(np.linalg.norm(rel-rel[0],axis=1).max())
  Rs=np.array([r["relative_rotation"] for r in subset]);delta=Rs@Rs[0].T
  rot=float(np.degrees(np.arccos(np.clip((np.trace(delta,axis1=1,axis2=2)-1)/2,-1,1))).max())
  passes=z-a+1>=3 and lift>=.05 and drift<=.03
  lifted=[i for i in range(a,z+1) if rows[i]["object_position"][2]-rows[a]["object_position"][2]>=.05]
  center=lifted[0] if lifted else a
  w=list(range(center,min(center+5,z+1)))
  if len(w)<3:w=list(range(max(a,z-4),z+1))
  candidates.append({"start":a,"end":z,"lift_m":lift,"drift_m":drift,"relative_rotation_max_deg":rot,"numeric_pass":passes,"window":w,"manual_review":"pending","mode":"pending_visual_review"})
 return rows,candidates,max(errors)
def collect(obj,raw,nwanted):
 dest=OUT/obj;dest.mkdir(parents=True,exist_ok=True);manifest=dest/"manifest.json"
 done=json.loads(manifest.read_text()) if manifest.exists() else {"object":obj,"source":str(raw),"episodes":[],"events":[],"rejected":[],"status":"collecting"}
 if len(done["events"])>=nwanted:return
 with h5py.File(raw,"r") as f:
  d=f["data"];env=OffScreenRenderEnv(bddl_file_name=str(resolve(d)),use_camera_obs=False)
  try:
   for name in sorted(d,key=lambda x:int(x.split("_")[-1])):
    if name in {e["episode"] for e in done["episodes"]}:continue
    ep=d[name];restore(env,ep);rows,candidates,error=inspect_episode(env,ep,obj)
    save(dest/(name+"_states_contacts.json"),{"episode":name,"rows":rows,"candidates":candidates})
    done["episodes"].append({"episode":name,"frames":len(rows),"state_roundtrip_error":error,"candidate_count":sum(c["numeric_pass"] for c in candidates)})
    for c in candidates:
     item={"id":obj+"_"+name+"_event"+str(c["start"]),"episode":name,**c}
     (done["events"] if c["numeric_pass"] else done["rejected"]).append(item)
    if not (dest/"surface.npz").exists():
     state(env,ep,0);tar,pad,b=ids(env,obj);p,v,faces=mesh_points(env,tar,b)
     np.savez_compressed(dest/"surface.npz",points_object=p,vertices=v,faces=faces)
     done["visual_bbox_size_m"]=(v.max(axis=0)-v.min(axis=0)).tolist()
    save(manifest,done);print(obj,name,"accepted",len(done["events"]),"roundtrip",error,flush=True)
    if len(done["events"])>=nwanted:break
  finally:env.close()
 done["status"]="awaiting_labels_and_manual_review";save(manifest,done)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--object",default="all");args=parser.parse_args()
 for obj,path,n in TASKS:
  if args.object not in ("all",obj):continue
  raw=ROOT/"data/raw"/path
  if not raw.exists():print("MISSING",raw,flush=True);continue
  collect(obj,raw,n)
