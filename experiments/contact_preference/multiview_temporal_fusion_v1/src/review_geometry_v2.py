import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import sys,json,numpy as np,h5py,importlib.util,traceback
from scipy.spatial import ConvexHull,cKDTree
import trimesh
B=Path('/mnt/nvme1/drh/D_VLA');R=B/'experiments/contact_preference/multiview_temporal_fusion_v1'
sys.path.insert(0,str(B/'src/v3'))
from stage2_robot_only_depth import RobotDepth
sp=importlib.util.spec_from_file_location('old_surface',B/'experiments/contact_preference/stage3_pilot_v1/src/build_supports_v2.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
def tf(p,T):return np.c_[p,np.ones(len(p))]@T.T[:,:3]
def main():
 s=json.loads((R/'refined_summary.json').read_text());robot=RobotDepth(448)
 for c in s['cases']:
  z=np.load(R/'refined_cases'/(c['key']+'.npz'));ids=z['supported_representatives'];points=z['common_xyz'][ids];ref=z['reference_xyz']
  reg=np.load(B/'experiments/region_pose/iter_024/train'/(c['key']+'.npz'))
  asset=c['key'].split('_demo_')[0];demo='demo_'+c['key'].split('_demo_')[1].split('_')[0]
  rows=[]
  with h5py.File(B/'data/v3/trainval_rgbd_448'/asset/(demo+'.h5'),'r') as h:
   for t in c['frames'][1:]:
    Q=reg['query_to_current_hand_%d'%t];H=reg['hand_to_world_%d'%t];W=H@Q;K=h['agentview/K'][t];T=h['agentview/camera_to_world'][t]
    robot.render(reg['finger_q_%d'%t],np.linalg.solve(H,T),K)
    active,_=old.surface(robot,[3,4,6,7]);mount,_=old.surface(robot,[1])
    hulls=[ConvexHull(old.surface(robot,[i])[0].vertices).equations for i in [3,4,6,7]]
    def region(x):
     ph=tf(x,Q);d,_=old.distance(active,ph);dm,_=old.distance(mount,ph);inside=np.zeros(len(x),bool)
     for eq in hulls:inside|=(ph@eq[:,:3].T+eq[:,3]).max(1)<=0
     return (d<=.003)&(dm>=d)&~inside,d,dm,inside
    near,d,dm,inside=region(points);rn,*_=region(ref)
    depthclass=[];nworld=tf(points,W)
    for cam in ['agentview','robot0_eye_in_hand']:
     K=h[cam+'/K'][t].astype(float);T=h[cam+'/camera_to_world'][t].astype(float);pc=tf(nworld,np.linalg.inv(T));uv=pc@K.T;uv=uv[:,:2]/uv[:,2:]
     ij=np.rint(uv).astype(int);ok=(pc[:,2]>0)&(ij[:,0]>=0)&(ij[:,0]<448)&(ij[:,1]>=0)&(ij[:,1]<448)
     diff=np.full(len(points),np.nan);dep=h[cam+'/depth_m'][t];diff[ok]=pc[ok,2]-dep[ij[ok,1],ij[ok,0]]
     cls=np.where(~ok,3,np.where(diff<-.003,2,np.where(diff>.003,1,0)));depthclass.append(cls)
    depthclass=np.array(depthclass)
    new=z['new_vs_reference'][ids];wextra=z['wrist_new_voxel'][ids]
    # Frontal visibility contradiction invalidates this current-frame geometric candidate.
    conflict=(depthclass==2).any(0)
    near_unknown=near&conflict
    usable=near&~conflict
    blocked_both=(depthclass==1).all(0)
    np.savez_compressed(R/'refined_cases'/(c['key']+'_view%d.npz'%t),source_indices=ids,near_finger=near,near_without_front_conflict=usable,current_front_conflict=conflict,near_unknown=near_unknown,depth_class=depthclass,active_distance=d,mount_distance=dm,inside_collision=inside,reference_near_finger=rn)
    row={'frame':t,'reference_near_sources':int(rn.sum()),'fused_near_cells':int(near.sum()),'near_front_conflict_unknown':int(near_unknown.sum()),'near_no_front_conflict_cells':int(usable.sum()),'new_near_cells':int((usable&new).sum()),'wrist_extra_near_cells':int((usable&wextra).sum()),'new_near_behind_main':int((usable&new&(depthclass[0]==1)).sum()),'new_near_behind_both':int((usable&new&blocked_both).sum()),'wrist_extra_near_behind_both':int((usable&wextra&blocked_both).sum())}
    rows.append(row)
  c['contact_vicinity_review']=rows
  print(json.dumps({'case':c['case'],'rows':rows}),flush=True)
 robot.close()
 (R/'fusion_review_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:(R/'metrics_failure_v2.log').write_text(traceback.format_exc());print(json.dumps({'error':type(e).__name__}));raise SystemExit(1)
