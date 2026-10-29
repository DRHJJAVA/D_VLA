"""Full synchronized RGB-D replay for the stage-1 review episodes; resume by files."""
import sys,os,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage1_collect import ROOT,OUT,TASKS,save,resolve,restore,state
from restore_audit import OffScreenRenderEnv
import h5py,numpy as np
from robosuite.utils.camera_utils import get_real_depth_map,get_camera_intrinsic_matrix,get_camera_extrinsic_matrix
DEST=ROOT/"data/v2/stage1_rgbd";SIZE=256;CAMS=["agentview","robot0_eye_in_hand"]
def main():
 for obj,path,n in TASKS:
  mf=OUT/obj/"manifest.json"
  if not mf.exists():continue
  manifest=json.loads(mf.read_text())
  selected=list(dict.fromkeys(e["episode"] for e in manifest["events"][:n]))
  target=DEST/obj;target.mkdir(parents=True,exist_ok=True)
  with h5py.File(manifest["source"]) as f:
   d=f["data"];env=None
   try:
    for name in selected:
     dest=target/(name+".h5")
     if dest.exists():
      with h5py.File(dest) as v:assert len(v[CAMS[0]+"/depth_m"])==len(d[name]["states"])
      continue
     old=ROOT/"data/rendered"/("alphabet_soup_"+name+"_full_rgbd.h5")
     if obj=="alphabet_soup" and old.exists():
      with h5py.File(old) as v:assert len(v[CAMS[0]+"/depth_m"])==len(d[name]["states"])
      dest.symlink_to(old);print("REUSED",obj,name,flush=True);continue
     if env is None:env=OffScreenRenderEnv(bddl_file_name=str(resolve(d)),use_camera_obs=False)
     ep=d[name];restore(env,ep);count=len(ep["states"]);part=dest.with_suffix(".h5.part")
     # A stale partial is retained separately, never mistaken for a complete episode.
     if part.exists():part.rename(part.with_name(part.name+"."+str(int(time.time()))))
     with h5py.File(part,"w") as out:
      out.attrs.update(source=str(manifest["source"]),episode=name,frames=count,size=SIZE,image_convention="opencv",status="writing")
      sets={}
      for cam in CAMS:
       g=out.create_group(cam)
       sets[cam]={}
       for key,shape,typ in [("rgb",(SIZE,SIZE,3),"u1"),("depth_m",(SIZE,SIZE),"f4"),("K",(3,3),"f8"),("camera_to_world",(4,4),"f8")]:
        sets[cam][key]=g.create_dataset(key,(count,*shape),dtype=typ,chunks=(1,*shape),compression="lzf")
      for t in range(count):
       state(env,ep,t)
       for cam in CAMS:
        rgb,depth=env.sim.render(camera_name=cam,width=SIZE,height=SIZE,depth=True)
        sets[cam]["rgb"][t]=np.flipud(rgb);sets[cam]["depth_m"][t]=np.flipud(get_real_depth_map(env.sim,depth))
        sets[cam]["K"][t]=get_camera_intrinsic_matrix(env.sim,cam,SIZE,SIZE);sets[cam]["camera_to_world"][t]=get_camera_extrinsic_matrix(env.sim,cam)
       if t%50==0:print("FRAME",obj,name,t,count,flush=True)
      out.attrs["status"]="complete"
     part.rename(dest);print("COMPLETE",obj,name,count,flush=True)
   finally:
    if env is not None:env.close()
if __name__=="__main__":main()
