#!/usr/bin/env python3
"""Probe saved MuJoCo states for true finger/object collision contacts."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from restore_audit import OffScreenRenderEnv, libero_utils
from asset_paths import localize_assets


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--libero-root",type=Path,required=True)
    p.add_argument("--episode",default="demo_0")
    args=p.parse_args()
    with h5py.File(args.input,"r") as f:
        data=f["data"]
        bddl=args.libero_root/"libero/libero/bddl_files/libero_object"/Path(data.attrs["bddl_file_name"]).name
        ep=data[args.episode]
        xml=ep.attrs["model_file"]
        if isinstance(xml,bytes):xml=xml.decode()
        xml=libero_utils.postprocess_model_xml(xml,{})
        xml,_=localize_assets(xml,args.libero_root)
        states=ep["states"]
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),use_camera_obs=False)
        rows=[]
        try:
            env.reset();env.reset_from_xml_string(xml);env.sim.reset()
            target_name=args.input.stem.removeprefix("pick_up_the_").split("_and_place_it_in_")[0]
            body_names=[env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]
            target_bodies=[(i,name) for i,name in enumerate(body_names) if name and name.startswith(target_name+"_")]
            site_names=[env.sim.model.site_id2name(i) for i in range(env.sim.model.nsite)]
            print("target_bodies",target_bodies,flush=True)
            print("gripper_sites",[x for x in site_names if x and ("grip" in x or "eef" in x)],flush=True)
            grip_site_id=env.sim.model.site_name2id("gripper0_grip_site")
            for frame in range(len(states)):
                env.sim.set_state_from_flattened(np.asarray(states[frame],dtype=np.float64))
                env.sim.forward()
                touches=[]
                for i in range(env.sim.data.ncon):
                    contact=env.sim.data.contact[i]
                    a=env.sim.model.geom_id2name(contact.geom1) or f"geom_{contact.geom1}"
                    b=env.sim.model.geom_id2name(contact.geom2) or f"geom_{contact.geom2}"
                    if "finger" in a.lower() or "finger" in b.lower():
                        touches.append({"geom1":a,"geom2":b,
                                        "position_world":np.asarray(contact.pos).astype(float).tolist(),
                                        "distance":float(contact.dist)})
                rows.append({"frame":frame,"all_contact_count":int(env.sim.data.ncon),
                             "finger_contacts":touches,
                             "target_body_positions":{name:np.asarray(env.sim.data.body_xpos[i]).astype(float).tolist() for i,name in target_bodies},
                             "target_body_poses":{name:{"position_world":np.asarray(env.sim.data.body_xpos[i]).astype(float).tolist(),
                                                        "rotation_world":np.asarray(env.sim.data.body_xmat[i]).reshape(3,3).astype(float).tolist()} for i,name in target_bodies},
                             "gripper_pose":{"position_world":np.asarray(env.sim.data.site_xpos[grip_site_id]).astype(float).tolist(),
                                             "rotation_world":np.asarray(env.sim.data.site_xmat[grip_site_id]).reshape(3,3).astype(float).tolist()}})
                if frame%25==0:print(f"frame {frame}/{len(states)}",flush=True)
        finally:env.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={"episode":args.episode,"frames":len(rows),"frames_with_finger_contacts":sum(bool(x["finger_contacts"]) for x in rows),"records":rows}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"frames":len(rows),"finger_contact_frames":report["frames_with_finger_contacts"],"output":str(args.output)}),flush=True)


if __name__=="__main__":main()
