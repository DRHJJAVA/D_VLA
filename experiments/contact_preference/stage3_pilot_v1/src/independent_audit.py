import os
os.environ["MUJOCO_GL"]="egl"
from pathlib import Path
import json,hashlib,importlib.util,traceback
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path("/mnt/nvme1/drh/D_VLA");O=R/"experiments/contact_preference/stage3_pilot_v1"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 S=json.loads((O/"summary.json").read_text())
 sp=importlib.util.spec_from_file_location("stage3",O/"src/build_supports_v2.py");m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 spec=importlib.util.spec_from_file_location("robot",R/"src/v3/stage2_robot_only_depth.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 import mujoco
 robot=mod.RobotDepth(32);rng=np.random.default_rng(2026);boxmax=0
 try:
  for ca in S["cases"]:
   z0=np.load(R/"experiments/region_pose/iter_024/train"/(ca["key"]+".npz"))
   for t in ca["frames"]:
    robot.data.qpos[:]=z0["finger_q_"+str(t)];mujoco.mj_forward(robot.model,robot.data)
    for gi in [4,7]:
     # Analytic AABB signed distance compared with triangle distance, outside & inside.
     rot=robot.data.geom_xmat[gi].reshape(3,3);pos=robot.data.geom_xpos[gi];sz=robot.model.geom_size[gi]
     local=rng.uniform(-.025,.025,(50,3));world=local@rot.T+pos
     mesh,_=m.surface(robot,[gi]);d,_=m.distance(mesh,world)
     q=np.abs(local)-sz;analytic=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(1),0)
     boxmax=max(boxmax,float(np.max(np.abs(d-np.abs(analytic)))))
 finally:robot.close()
 maxpost=0;rows=[];mappingmax=0;webmax=0
 for ca in S["cases"]:
  z0=np.load(R/"experiments/region_pose/iter_024/train"/(ca["key"]+".npz"))
  for row in ca["rows"]:
   p=O/row["data"];z=np.load(p)
   assert np.array_equal(z["query_xyz"],z0["query_xyz"]) and np.array_equal(z["query_pixel"],z0["query_pixel"])
   hp=m.apply(z["query_xyz"],z["query_to_current_hand"]);mappingmax=max(mappingmax,float(np.max(np.abs(hp-z["mapped_hand_xyz"]))))
   mass=np.column_stack([z["positive"],z["separation"],z["unknown"]])
   assert np.allclose(mass.sum(1),1) and np.all(mass>=0)
   bad=z["penetrated"];assert np.all(z["unknown"][bad]==1)
   if not row["motion_positive_basis"]:assert np.all(z["positive"]==0)
   if (O/"webdata"/row.get("webstem","missing")).exists():pass
   n=len(z["distance_m"]);select=np.linspace(0,n-1,6,dtype=int)
   for i in select:
    d=z["distance_m"][i];s=z["sigma_base_m"][i];w=s
    if d/s>10:continue
    # Rescale integration variable to unit interval to avoid numerical cancellation.
    fs=quad(lambda u:norm.pdf((d-w*u)/s)/s,0,1,epsabs=1e-10)[0]
    fc=norm.pdf(d/s)/s
    if fc+fs>0:maxpost=max(maxpost,abs(float(m.post(np.array([d]),np.array([s]),np.array([w]),.5)[0])-fc/(fc+fs)))
   rows.append(dict(key=ca["key"],frame=row["frame"],n=n,positive_dominant=int((z["display_class"]==0).sum()),
     negative_dominant=int((z["display_class"]==1).sum()),unknown_dominant=int((z["display_class"]==2).sum())))
 old=json.loads((O/"input_locked.json").read_text());assert all(sha(Path(p))==v for p,v in old.items())
 assets=Path(importlib.util.find_spec("robosuite").origin).parent/"models/assets/grippers"
 import xml.etree.ElementTree as ET
 xml=assets/"panda_gripper.xml";root=ET.parse(xml).getroot()
 meshes=[]
 for node in root.findall(".//mesh"):
  name=node.attrib.get("file")
  if name:
   found=list(assets.rglob(Path(name).name))
   if len(found)==1:meshes.append(found[0])
 ah={str(p):sha(p) for p in [xml]+meshes}
 (O/"robot_assets_postread_locked.json").write_text(json.dumps(ah,indent=2))
 assert boxmax<1e-10 and maxpost<1e-8 and mappingmax==0
 report=dict(time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),analytic_box_distance_max_m=boxmax,
   numerical_integral_posterior_max_difference=maxpost,source_mapping_max_m=mappingmax,input_unchanged=True,
   rows=rows,robot_asset_files=len(ah),robot_asset_lock_timing="postread; no claim of precompute lock",
   physical_contact_calibration=False,score_network_trained=False,human_interval_used=False)
 (O/"independent_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k!="rows"},ensure_ascii=True))
if __name__=="__main__":
 try:main()
 except Exception as e:
  (O/"independent_failure.log").write_text(traceback.format_exc())
  print(json.dumps(dict(error=type(e).__name__,message=str(e))));raise SystemExit(1)
