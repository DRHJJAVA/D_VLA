"""Optical-flow feasibility check for visual object/gripper co-motion.
RGB-D train/val only; no actions, simulator truth, names, labels or held-out data.
This is a diagnostic, not a learned contact map or training label.
"""
from pathlib import Path
import json
import cv2,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
SRC=R/'data/v3/samples/visual_interaction_uniform448_3dcc'
COMP=R/'data/v3/proposals/foreground3d_components'
OUT=R/'audits/v3/stage2_trainval_comotion_opticalflow_prelim.json'
def stats(flow,mask):
 vec=flow[mask]
 if len(vec)<20:return None
 median=np.median(vec,axis=0)
 magnitude=float(np.linalg.norm(median))
 return dict(pixels=len(vec),median_dx=float(median[0]),median_dy=float(median[1]),
  median_vector_magnitude=magnitude,median_point_speed=float(np.median(np.linalg.norm(vec,axis=1))))
def main():
 assert not OUT.exists()
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   with np.load(SRC/split/(id+'.npz')) as z:pixel=z['pixel'].copy();onsets=z['onsets'].copy()
   with np.load(COMP/split/(id+'.npz')) as z:
    path=Path(str(z['rgbd_path']));box=z['outer_box'].astype(int)
   with h5py.File(path) as f:
    rgb=f['agentview/rgb'];depth=f['agentview/depth_m'];first=depth[0]
    events=[]
    for event in onsets:
     t1=max(0,int(event)-2);t2=min(len(rgb)-1,int(event)+2)
     g1=cv2.cvtColor(rgb[t1],cv2.COLOR_RGB2GRAY)
     g2=cv2.cvtColor(rgb[t2],cv2.COLOR_RGB2GRAY)
     flow=cv2.calcOpticalFlowFarneback(g1,g2,None,.5,3,21,3,5,1.2,0)
     px=pixel[:,0];py=pixel[:,1]
     d1=depth[t1]
     obj=np.zeros(first.shape,bool)
     stable=(d1[py,px]>.05)&(np.abs(d1[py,px]-first[py,px])<.015)
     obj[py[stable],px[stable]]=True
     x1,y1,x2,y2=box
     xlo=max(0,x1-35);ylo=max(0,y1-35);xhi=min(first.shape[1],x2+35);yhi=min(first.shape[0],y2+35)
     roi=np.zeros(first.shape,bool);roi[ylo:yhi,xlo:xhi]=True
     robot=roi&(d1<first-.02)&(d1>.05)
     robot[ylo:yhi:2,xlo:xhi:2]=robot[ylo:yhi:2,xlo:xhi:2]
     o=stats(flow,obj);g=stats(flow,robot)
     cos=None
     if o and g:
      ov=np.array([o['median_dx'],o['median_dy']]);gv=np.array([g['median_dx'],g['median_dy']])
      cos=float(ov@gv/(max(np.linalg.norm(ov)*np.linalg.norm(gv),1e-9)))
     events.append(dict(event=int(event),frames=[t1,t2],visible_object=o,nearby_new_foreground=g,
      flow_cosine=cos,
      object_visible_fraction=float(stable.mean()),
      robot_foreground_pixels=int(robot.sum())))
   row=dict(id=id,split=split,events=events)
   rows.append(row);print(json.dumps(row),flush=True)
 summary={}
 for split in ('train','val'):
  ev=[v for r in rows if r['split']==split for v in r['events']]
  valid=[v for v in ev if v['visible_object'] and v['nearby_new_foreground']]
  summary[split]=dict(episodes=sum(r['split']==split for r in rows),
   events=len(ev),flow_comparable=len(valid),
   median_object_flow_pixels=float(np.median([v['visible_object']['median_vector_magnitude'] for v in valid])) if valid else None,
   median_robot_flow_pixels=float(np.median([v['nearby_new_foreground']['median_vector_magnitude'] for v in valid])) if valid else None,
   median_direction_cosine=float(np.median([v['flow_cosine'] for v in valid])) if valid else None,
   fraction_cosine_above_point7=float(np.mean([v['flow_cosine']>.7 for v in valid])) if valid else None)
 report=dict(status='trainval_rgbd_comotion_flow_feasibility_preliminary',
  no_simulator_truth=True,no_actions=True,no_names=True,
  reference='Farneback 2D flow over removal onset +/-2 frames, first-frame object pixels still depth-visible before lift, and nearby new foreground',
  caveat='nearby foreground is not an identified gripper tip; small/occluded object flow is noisy; cosine is not grasp quality',
  summary=summary,rows=rows)
 OUT.write_text(json.dumps(report,indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
