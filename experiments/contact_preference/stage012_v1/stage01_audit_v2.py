from pathlib import Path
import json, hashlib
import numpy as np
import h5py
from datetime import datetime, timezone, timedelta
R=Path('/mnt/nvme1/drh/D_VLA')
O=R/'experiments/contact_preference/stage012_v1'
O.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def stamp(): return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
splitp=R/'data/v3/manifests/stage2_episode_split.json'
sp=json.loads(splitp.read_text())
ss={k:set(v) for k,v in sp['splits'].items()}
overlaps={a+'__'+b:sorted(ss[a]&ss[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
cases={'drawer_demo_0':'KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_demo.hdf5','stove_demo_0':'KITCHEN_SCENE9_turn_on_the_stove_demo.hdf5','two_objects_demo_0':'LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5'}
cf=R/'experiments/contact_frames/cf017'
inventory={}
for name,fn in cases.items():
 raw=cf/'raw'/fn
 rgbd=cf/'rgbd'/(name+'.h5')
 motion=cf/'cases'/name/'motion.npz'
 with h5py.File(raw,'r') as h:
  g=h['data/demo_0']
  raw_n=int(g['actions'].shape[0])
  obs_n=int(g['obs/gripper_states'].shape[0])
  shapes={k:list(g['obs'][k].shape) for k in ['ee_pos','ee_ori','gripper_states']}
 with h5py.File(rgbd,'r') as h:
  replay_n=int(h['frame_indices'].shape[0])
  indexed=np.array_equal(h['frame_indices'][:],np.arange(replay_n))
 inventory[name]={'role':'previously viewed cross-task development case; not blind test','raw_file':str(raw.relative_to(R)),'raw_sha256':sha(raw),'rgbd_file':str(rgbd.relative_to(R)),'rgbd_sha256':sha(rgbd),'motion_file':str(motion.relative_to(R)),'motion_sha256':sha(motion),'frames':{'raw_actions':raw_n,'raw_observation':obs_n,'replay_rgbd':replay_n},'raw_observation_shapes':shapes,'replay_frame_indices_zero_based_contiguous':bool(indexed)}
stage0={'created_beijing':stamp(),'split_source':str(splitp.relative_to(R)),'split_sha256':sha(splitp),'unit':sp['unit'],'split_counts':{k:len(v) for k,v in sp['splits'].items()},'count_fields_agree':all(len(sp['splits'][k])==sp['counts'][k] for k in sp['splits']),'intersections':overlaps,'unique_count':len(set.union(*ss.values())),'selection_bias':sp['selection_bias'],'case_inventory':inventory,'reserved_boundary':'reserved demo40-44 contents, images, states and blind-test results not read; these three cross-task cases are development evidence only','stage3_training_started':False}
(O/'stage0_audit.json').write_text(json.dumps(stage0,ensure_ascii=False,indent=2))
def roundstats(a):
 a=np.asarray(a);return {'n':int(a.size),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(np.max(a))}
stage1={'created_beijing':stamp(),'cases':{},'source_query':{}}
for name,fn in cases.items():
 raw=cf/'raw'/fn; rgbd=cf/'rgbd'/(name+'.h5'); base=cf/'cases'/name
 m=np.load(base/'motion.npz'); ref=np.load(base/'artifacts/reference.npz')
 with h5py.File(raw,'r') as h:
  obs=h['data/demo_0/obs'];q=np.asarray(obs['gripper_states'][:]);e=np.asarray(obs['ee_pos'][:])
 H=np.asarray(m['hand_to_world']);fq=np.asarray(m['measured_finger_q']);T=len(H)
 qerr=np.max(np.abs(fq[1:]-q[:T-1]),axis=1)
 local=np.einsum('tij,tj->ti',np.transpose(H[1:,:3,:3],(0,2,1)),H[1:,:3,3]-e[:T-1])
 local_med=np.median(local,axis=0);local_dev=np.linalg.norm(local-local_med,axis=1)
 with h5py.File(rgbd,'r') as h:
  K=np.asarray(h['agentview/K'][0],dtype=np.float64);C=np.asarray(h['agentview/camera_to_world'][0],dtype=np.float64)
  uv=np.asarray(ref['surface_uv'],dtype=np.int64);xyz=np.asarray(ref['surface_xyz'],dtype=np.float64)
  D=np.asarray(h['agentview/depth_m'][0]); z=D[uv[:,1],uv[:,0]].astype(np.float64)
  pc=np.stack(((uv[:,0]-K[0,2])*z/K[0,0],(uv[:,1]-K[1,2])*z/K[1,1],z),axis=1)
  pred=pc@C[:3,:3].T+C[:3,3]; xyzerr=np.linalg.norm(pred-xyz,axis=1)
  back=(xyz-C[:3,3])@np.linalg.inv(C[:3,:3]).T
  uv_back=np.stack((K[0,0]*back[:,0]/back[:,2]+K[0,2],K[1,1]*back[:,1]/back[:,2]+K[1,2]),axis=1)
  repro=np.linalg.norm(uv_back-uv,axis=1)
  Ks=np.asarray(h['agentview/K'][:]);Cs=np.asarray(h['agentview/camera_to_world'][:])
  wK=np.max(np.abs(Ks[:,2,:]-np.array([0,0,1])),axis=1);wC=np.max(np.abs(Cs[:,3,:]-np.array([0,0,0,1])),axis=1)
 ds=np.load(base/'artifacts/dataset.npz');times=np.asarray(ds['frame_time'])
 stage1['cases'][name]={'frames':T,'q_t_vs_raw_obs_t_minus_1_max_abs':float(qerr.max()),'t0_q_unknown':bool(np.all(~np.isfinite(fq[0]))),'t0_hand_unknown':bool(np.all(~np.isfinite(H[0]))),'ee_to_hand_local_offset_median_m':local_med.tolist(),'ee_to_hand_offset_deviation_m':roundstats(local_dev),'replay_frame_time_step_s':roundstats(np.diff(times)),'replay_time_is_metadata_not_raw_physical_clock':True,'agent_source_points':len(uv),'source_depth_to_world_error_m':roundstats(xyzerr),'world_to_source_pixel_error_px':roundstats(repro),'K_bottom_row_max_abs_error':float(wK.max()),'C_bottom_row_max_abs_error':float(wC.max()),'positive_source_depth_fraction':float(np.mean(z>0)),'RGB_alignment_by_pixel_equality_verified':False}
p=R/'experiments/region_pose/iter_024/train/porcelain_mug_demo_9_attempt1_query36.npz'
a=np.load(p);g=R/'data/v3/trainval_rgbd_448/porcelain_mug/demo_9.h5'
uv=np.asarray(a['query_pixel'],dtype=np.int64);xyz=np.asarray(a['query_xyz']);K=np.asarray(a['query_K'],dtype=np.float64);C=np.asarray(a['query_camera_to_world'],dtype=np.float64)
with h5py.File(g,'r') as h:
 d=np.asarray(h['agentview/depth_m'][36]);rgb=np.asarray(h['agentview/rgb'][36])
z=d[uv[:,1],uv[:,0]].astype(np.float64)
pc=np.stack(((uv[:,0]-K[0,2])*z/K[0,0],(uv[:,1]-K[1,2])*z/K[1,1],z),axis=1)
err=np.linalg.norm(pc@C[:3,:3].T+C[:3,3]-xyz,axis=1)
stage1['source_query']={'case':'porcelain_mug_demo_9 query36','saved_source':str(p.relative_to(R)),'source_sha256':sha(p),'rendered_rgbd_sha256':sha(g),'query_points':len(uv),'query_xyz_from_actual_frame36_depth_error_m':roundstats(err),'query_rgb_matches_actual_frame36_fraction':float(np.mean(np.all(rgb[uv[:,1],uv[:,0]]==a['query_rgb'],axis=1)))}
(O/'stage1_audit.json').write_text(json.dumps(stage1,ensure_ascii=False,indent=2))
print(json.dumps({'stage0':{'split_counts':stage0['split_counts'],'unique':stage0['unique_count'],'overlaps':overlaps,'case_frames':{k:v['frames'] for k,v in inventory.items()}},'stage1':{'q_error_max':max(v['q_t_vs_raw_obs_t_minus_1_max_abs'] for v in stage1['cases'].values()),'source_xyz_error_max':max(v['source_depth_to_world_error_m']['max'] for v in stage1['cases'].values()),'source_query_error_max':stage1['source_query']['query_xyz_from_actual_frame36_depth_error_m']['max']}},ensure_ascii=False))
