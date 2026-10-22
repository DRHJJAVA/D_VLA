#!/usr/bin/env python3
"""Extract candidate rigid-object grasp events from MuJoCo contact probes."""
import argparse
import json
from pathlib import Path

import numpy as np

SIDES = {"left": "finger1_pad", "right": "finger2_pad"}


def qualifying_contacts(row, object_prefix):
    result = {"left": [], "right": []}
    for contact in row["finger_contacts"]:
        names = (contact["geom1"], contact["geom2"])
        if not any(name.startswith(object_prefix) for name in names):
            continue
        for side, token in SIDES.items():
            if any(token in name for name in names):
                result[side].append(contact)
    return result


def runs(flags):
    start = None
    for index, value in enumerate(flags + [False]):
        if value and start is None: start = index
        elif not value and start is not None:
            yield start, index - 1
            start = None


def pose_point(point, pose):
    position=np.asarray(pose["position_world"],dtype=float)
    rotation=np.asarray(pose["rotation_world"],dtype=float)
    return (rotation.T @ (np.asarray(point,dtype=float)-position)).tolist()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--probe",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--object-prefix",required=True)
    p.add_argument("--object-body",required=True)
    p.add_argument("--object-asset-id",required=True)
    p.add_argument("--min-contact-frames",type=int,default=3)
    p.add_argument("--min-lift-m",type=float,default=0.05)
    p.add_argument("--max-relative-drift-m",type=float,default=0.03)
    args=p.parse_args()
    source=json.loads(args.probe.read_text())
    rows=source["records"]
    if not rows or "target_body_poses" not in rows[0]:
        raise RuntimeError("Probe must include body and gripper poses")
    per_frame=[qualifying_contacts(row,args.object_prefix) for row in rows]
    flags=[bool(x["left"] and x["right"]) for x in per_frame]
    events=[]
    for event_index,(start,end) in enumerate(runs(flags)):
        duration=end-start+1
        positions=np.asarray([rows[i]["target_body_poses"][args.object_body]["position_world"] for i in range(start,end+1)])
        lift=float(np.max(positions[:,2])-positions[0,2])
        relative=[]
        for i in range(start,end+1):
            object_pos=np.asarray(rows[i]["target_body_poses"][args.object_body]["position_world"])
            gripper=rows[i]["gripper_pose"]
            grip_pos=np.asarray(gripper["position_world"])
            grip_rotation=np.asarray(gripper["rotation_world"])
            relative.append(grip_rotation.T @ (object_pos-grip_pos))
        relative=np.asarray(relative)
        drift=float(np.max(np.linalg.norm(relative-relative[0],axis=1)))
        accepted=duration>=args.min_contact_frames and lift>=args.min_lift_m and drift<=args.max_relative_drift_m
        contact_points={}
        for side in SIDES:
            points=[]
            for i in range(start,min(end+1,start+args.min_contact_frames)):
                pose=rows[i]["target_body_poses"][args.object_body]
                for contact in per_frame[i][side]:
                    world=contact["position_world"]
                    points.append({"frame":i,"position_world":world,"position_object":pose_point(world,pose),
                                   "geom1":contact["geom1"],"geom2":contact["geom2"]})
            contact_points[side]=points
        events.append({"episode_id":source["episode"],"object_asset_id":args.object_asset_id,
                       "grasp_event_id":f"{source['episode']}_grasp_{event_index}",
                       "contact_frame":start,"end_frame":end,"duration_frames":duration,
                       "left_contacts":contact_points["left"],"right_contacts":contact_points["right"],
                       "object_pose":rows[start]["target_body_poses"][args.object_body],
                       "gripper_pose":rows[start]["gripper_pose"],
                       "lift_m":lift,"max_relative_drift_m":drift,"passes_numeric_filter":accepted,
                       "quality_flag":"candidate_manual_review" if accepted else "numeric_filter_failed"})
    report={"probe":str(args.probe),"criteria":{"min_contact_frames":args.min_contact_frames,
            "min_lift_m":args.min_lift_m,"max_relative_drift_m":args.max_relative_drift_m},
            "events":events,"candidate_count":sum(e["passes_numeric_filter"] for e in events)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"runs":len(events),"candidates":report["candidate_count"],"output":str(args.output)}))


if __name__=="__main__":main()
