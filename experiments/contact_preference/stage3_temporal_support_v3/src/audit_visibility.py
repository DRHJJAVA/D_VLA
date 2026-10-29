from pathlib import Path
import os,sys,json
os.environ['MUJOCO_GL']='egl'
import numpy as np,h5py
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/stage3_temporal_support_v3'
sys.path.insert(0,str(R))
from src.v3.stage2_robot_only_depth import RobotDepth
s=json.loads((O/'summary.json').read_text());robot=RobotDepth(448);rows=[]
for ci,c in enumerate(s['cases']):
 mo=np.load(R/'experiments/region_pose/iter_003/evidence/motion'/f"{c['asset']}_{c['demo']}.npz")
 with h5py.File(R/'data/v3/trainval_rgbd_448'/c['asset']/(c['demo']+'.h5'),'r') as h:
  for t in c['frames']:
   z=np.load(O/'points'/f"{c['key']}_frame{t}.npz");j=int(np.where(h['frame_indices'][:]==t)[0][0]);world=z['mapped_world_xyz'];labels=[]
   for cam in ['agentview','robot0_eye_in_hand']:
    C=h[cam]['camera_to_world'][j].astype(float);K=h[cam]['K'][j].astype(float)
    camera=np.linalg.solve(C,np.column_stack([world,np.ones(len(world))]).T).T[:,:3]
    projection=K@camera.T;uv=(projection[:2]/projection[2]).T
    ij=np.rint(np.nan_to_num(uv,nan=-100,posinf=-100,neginf=-100)).astype(int)
    inside=(camera[:,2]>0)&(ij[:,0]>=0)&(ij[:,0]<448)&(ij[:,1]>=0)&(ij[:,1]<448)
    obs=np.full(len(world),np.nan);rd=np.full(len(world),np.nan);dep=h[cam]['depth_m'][j]
    render=robot.render(mo['measured_finger_q'][t],np.linalg.solve(mo['hand_to_world'][t].astype(float),C),K)
    ids=np.flatnonzero(inside);obs[ids]=dep[ij[ids,1],ij[ids,0]];rd[ids]=render[ij[ids,1],ij[ids,0]]
    delta=camera[:,2]-obs;valid=inside&np.isfinite(obs)&(obs>0)&(obs<5)
    label=np.full(len(world),4,np.int8)
    label[valid&(delta<-.003)]=3;label[valid&(delta>.003)]=2
    label[valid&(delta>.003)&(rd>0)&(rd<5)&(abs(rd-obs)<=.003)&(rd<camera[:,2]-.003)]=1
    label[valid&(abs(delta)<=.003)]=0
    labels.append(label)
   labels=np.array(labels);assert np.array_equal(labels,z['camera_visibility'])
   combined=np.full(len(world),4,np.int8)
   for k in range(len(world)):
    pair=labels[:,k].tolist()
    combined[k]=3 if 3 in pair else 0 if 0 in pair else 2 if 2 in pair else 1 if 1 in pair else 4
   assert np.array_equal(combined,z['visibility'])
   rows.append({'case':ci,'frame':t,'n':len(world),'counts':np.bincount(combined,minlength=5).tolist(),'equal':True})
robot.close()
out={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),'all_passed':True,'actual_depth_and_robot_rerender':True,'rows':rows,'limits':'robot-depth agreement is an occlusion proxy, not object identity/contact truth'}
(O/'audits/visibility.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('visibility independent passed',len(rows))
