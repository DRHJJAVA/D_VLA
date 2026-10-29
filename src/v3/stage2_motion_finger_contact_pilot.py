"""Measured robot-state assisted contact evidence. Robot-only geometry; no simulator object/contact truth."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,json,hashlib,cv2,h5py,numpy as np
from scipy.spatial import cKDTree
from stage2_probe_motion_fingerpads import tc,cloud,project,corners
from stage2_robot_only_depth import RobotDepth
R=Path('/mnt/nvme1/drh/D_VLA')
BASE=R/'audits/v3/finger_contact_repair_pilot_v1'
OUT=R/'audits/v3/motion_finger_contact_pilot_v1'
def main():
 cv2.setNumThreads(4);OUT.mkdir(exist_ok=False)
 rows=json.loads((BASE/'protocol.json').read_text())['episodes']
 E={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
 config=dict(episodes=[{'id':x['id'],'split':x['split']} for x in rows],robot_state_alignment=-1,
  inner_pad_patch_halfwidth_m=.008,pad_z_interval_m=[.0854,.1014],object_side_distance_m=[.0003,.002],
  robot_depth_exclusion_m=.002,nonempty_gap_m=.006,visual_common_frame_threshold_m=.003,
  min_cloud_points=20,minimum_lift_m=.025,minimum_overlap=.5,minimum_supported_moving_frames=5,
  no_simulator_object_states=True,no_simulator_contacts=True,no_manual_region_labels=True,
  outputs='visible RGB-D points in object-side inner finger-pad band; proximity estimate, not measured force/contact truth',
  successful_attempt_definition='closed command + nonempty measured gap + visual surface stable in hand coordinates during lift; local held grasp, not full task outcome')
 (OUT/'protocol.json').write_text(json.dumps(config,indent=2))
 summaries=[]
 for row in rows:
  e=E[row['id']]
  with h5py.File(e['source_path'],'r') as f:
   g=f['data'][e['source_episode']];q=g['obs/gripper_states'][:];act=g['actions'][:,-1];ee=g['obs/ee_pos'][:]
  q=np.r_[np.full((1,2),np.nan),q[:-1]]
  n=len(q);close=act>0
  starts=np.flatnonzero(close&~np.r_[False,close[:-1]]);ends=np.flatnonzero(close&~np.r_[close[1:],False])
  records=[];all_points=[];all_sides=[];all_cams=[];all_pixels=[];offsets=[0]
  heldcloud=[];overlap=np.full(n,np.nan)
  with h5py.File(row['rgbd'],'r') as f:
   hand=f['robot0_eye_in_hand/camera_to_world'][:]@np.linalg.inv(tc())
   size=f['agentview/rgb'].shape[1]
   robot=RobotDepth(size)
   for t in range(n):
    pts=[];sides=[];cams=[];pixels=[];obj=[];stripcounts=[0,0];rejected=[0,0]
    if t>0:
     for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
      c=f[cam];d=c['depth_m'][t];K=c['K'][t];T=c['camera_to_world'][t]
      rd=robot.render(q[t],np.linalg.inv(hand[t])@T,K)
      # Observed material in front of robot visual surface, rather than the robot itself.
      nonrobot=(d<rd-.002).reshape(-1)
      p=cloud(d,K,T);ph=(p-hand[t,:3,3])@hand[t,:3,:3]
      left=q[t,0]-.0005;right=q[t,1]+.0005
      spatial=(abs(ph[:,0])<=.008)&(ph[:,2]>=.0854)&(ph[:,2]<=.1014)
      roi=nonrobot&(abs(ph[:,0])<.03)&(ph[:,2]>.08)&(ph[:,2]<.18)&(ph[:,1]<left)&(ph[:,1]>right)
      if roi.any():obj.append(ph[roi])
      for side,k in [(1,0),(-1,1)]:
       delta=side*(q[t,k]-side*.0005-ph[:,1])
       near=spatial&(delta>=.0003)&(delta<=.002)
       good=near&nonrobot
       rejected[k]+=int((near&~nonrobot).sum());stripcounts[k]+=int(good.sum())
       ix=np.flatnonzero(good)
       if len(ix):
        vs,us=np.unravel_index(ix,d.shape)
        pts.append(p[ix]);sides.extend([k]*len(ix));cams.extend([ci]*len(ix));pixels.extend(np.stack([us,vs],1))
    surface=np.concatenate(obj) if obj else np.empty((0,3))
    if len(surface):
     _,ind=np.unique(np.rint(surface/.002).astype(int),axis=0,return_index=True);surface=surface[ind]
    heldcloud.append(surface)
    if t and len(surface)>=20 and len(heldcloud[t-1])>=20:
     overlap[t]=float((cKDTree(heldcloud[t-1]).query(surface)[0]<.003).mean())
    point=np.concatenate(pts) if pts else np.empty((0,3))
    all_points.extend(point);all_sides.extend(sides);all_cams.extend(cams);all_pixels.extend(pixels);offsets.append(len(all_points))
    records.append({'frame':t,'pad_gap_mm':None if t==0 else float((q[t,0]-q[t,1]-.001)*1000),'visible_pad_band_points':stripcounts,'robot_or_ambiguous_removed':rejected,'hand_frame_object_points':len(surface),'previous_frame_surface_overlap':None if not np.isfinite(overlap[t]) else float(overlap[t])})
   robot.close()
   attempts=[]
   for ai,(a,b) in enumerate(zip(starts,ends)):
    gap=q[:,0]-q[:,1]-.001
    moving=(hand[:,2,3]-hand[a,2,3])>.025
    valid=(gap>.006)&moving&(overlap>=.5)&(np.array([len(x) for x in heldcloud])>=20)
    support=np.flatnonzero(valid&(np.arange(n)>=a)&(np.arange(n)<=b))
    median=float(np.nanmedian(gap[a+1:b+1][-max(3,(b-a)//2):]))
    passed=len(support)>=5
    # Gate evaluated over each attempt; failures retain raw diagnostics but no accepted contact points.
    status='held_candidate_with_visual_support' if passed else ('empty_closure_no_held_evidence' if median<.006 else 'insufficient_visual_support')
    eligible=np.flatnonzero((np.arange(n)>=a)&(np.arange(n)<=b)&(gap>.006)&(overlap>=.5)) if passed else np.array([],int)
    # Only frames at/after first supported lifted observation are accepted.
    if len(support):eligible=eligible[eligible>=support[0]]
    attempts.append({'attempt':ai+1,'start_frame':int(a),'end_frame':int(b),'gap_late_median_mm':median*1000,'moving_visual_support_frames':support.tolist(),'accepted_contact_frames':eligible.tolist(),'status':status,'whole_task_success_claim':False})
   np.savez_compressed(OUT/(row['id']+'.npz'),points_world=np.asarray(all_points).reshape(-1,3),point_side=np.array(all_sides,np.uint8),point_camera=np.array(all_cams,np.uint8),point_pixel=np.asarray(all_pixels,dtype=np.int32).reshape(-1,2),frame_offsets=np.array(offsets),hand_to_world=hand,measured_finger_q=q,visual_overlap=overlap,command_close=close)
  result={'id':row['id'],'split':row['split'],'frames':n,'attempts':attempts,'frame_diagnostics':records}
  (OUT/(row['id']+'.json')).write_text(json.dumps(result,indent=2))
  summaries.append({k:v for k,v in result.items() if k!='frame_diagnostics'});print(json.dumps(summaries[-1]),flush=True)
 (OUT/'summary.json').write_text(json.dumps({'episodes':summaries},indent=2))
 print('complete',flush=True)
if __name__=='__main__':main()
