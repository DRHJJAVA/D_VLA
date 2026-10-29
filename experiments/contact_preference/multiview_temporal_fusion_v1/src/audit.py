import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np,h5py,json,hashlib,sys,traceback
from scipy.spatial import cKDTree
from datetime import datetime
from zoneinfo import ZoneInfo
B=Path('/mnt/nvme1/drh/D_VLA');R=B/'experiments/contact_preference/multiview_temporal_fusion_v1'
sys.path.insert(0,str(B/'src/v3'))
from stage2_robot_only_depth import RobotDepth
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 locks=[]
 for p in [R/'input_locked.json',R/'source_locked.json',R/'source_addendum.json',B/'experiments/contact_preference/stage3_pilot_v2/completed_locked.json']:
  d=json.loads(p.read_text());bad=[q for q,v in d.items() if sha(q)!=v];assert not bad;locks.append({'lock':str(p),'count':len(d),'mismatches':bad})
 s=json.loads((R/'refined_summary.json').read_text());robot=RobotDepth(448);rows=[]
 for c in s['cases']:
  key=c['key'];z=np.load(R/'refined_cases'/(key+'.npz'));asset=key.split('_demo_')[0];demo='demo_'+key.split('_demo_')[1].split('_')[0]
  mot=np.load(B/'experiments/region_pose/iter_003/evidence/motion'/(asset+'_'+demo+'.npz'));reg=np.load(B/'experiments/region_pose/iter_024/train'/(key+'.npz'));sel=z['supported'];p=z['common_xyz'];rgb=z['rgb'];maxw=maxq=maxpix=0.;robot_bad=0
  with h5py.File(B/'data/v3/trainval_rgbd_448'/asset/(demo+'.h5'),'r') as h:
   for vi in np.unique(z['view_id']):
    ids=np.flatnonzero(z['view_id']==vi);t=int(z['frame'][ids[0]]);cam=['agentview','robot0_eye_in_hand'][int(z['camera'][ids[0]])]
    uv=z['pixel'][ids];dep=h[cam+'/depth_m'][t];K=h[cam+'/K'][t].astype(float);T=h[cam+'/camera_to_world'][t].astype(float)
    pc=np.linalg.solve(K,np.c_[uv,np.ones(len(ids))].T).T*dep[uv[:,1],uv[:,0],None]
    pw=(T@np.c_[pc,np.ones(len(pc))].T).T[:,:3]
    W=np.eye(4) if t==c['query_frame'] else reg['hand_to_world_%d'%t]@reg['query_to_current_hand_%d'%t]
    pq=np.linalg.solve(W,np.c_[pw,np.ones(len(pw))].T).T[:,:3]
    maxw=max(maxw,float(abs(pw-z['source_world_xyz'][ids]).max()));maxq=max(maxq,float(abs(pq-p[ids]).max()))
    assert np.array_equal(h[cam+'/rgb'][t][uv[:,1],uv[:,0]],rgb[ids])
    uv2=(pc@K.T);uv2=uv2[:,:2]/uv2[:,2:];maxpix=max(maxpix,float(abs(uv2-uv).max()))
    rd=robot.render(mot['measured_finger_q'][t],np.linalg.solve(mot['hand_to_world'][t],T),K)
    robot_bad+=int(np.sum(~(dep[uv[:,1],uv[:,0]]<rd[uv[:,1],uv[:,0]]-.002)))
  assert maxw<1e-12 and maxq<1e-12 and robot_bad==0
  assert np.all(z['normal_consistent'][sel]) and np.all(z['normal_dot'][:,sel]>0)
  assert np.all(z['loo_shift'][sel]<=.002) and np.all(z['ref_distance'][sel]<=.003)
  assert np.all(z['color_distance'][sel]<=c['color_l2_limit_q95']+1e-12)
  wit=z['witness'];actualcount=(wit>=0).sum(1);assert np.array_equal(actualcount,z['distinct_frame_support'])
  for j,t in enumerate(c['frames']):
   have=np.flatnonzero(wit[:,j]>=0);target=wit[have,j];assert np.all(z['frame'][target]==t)
   assert np.max(np.linalg.norm(p[have]-p[target],axis=1))<=.003+1e-12
   assert np.max(np.linalg.norm(rgb[have]/255.-rgb[target]/255.,axis=1))<=c['color_l2_limit_q95']+1e-12
   assert np.all(z['connected'][target]&z['stable'][target]&z['normal_consistent'][target])
  assert np.all(actualcount[sel]>=2) and np.array_equal(z['unknown'],~sel)
  si=z['supported_representatives'];assert np.all(sel[si]);assert len(np.unique(z['voxel'][si],axis=0))==len(si)
  main=sel&(z['camera']==0)&(z['main_frame_support']>=2)
  dmain=cKDTree(p[main]).query(p[si])[0] if main.any() else np.full(len(si),np.inf)
  dref=cKDTree(z['reference_xyz']).query(p[si])[0]
  result={'case':c['case'],'key':key,'source_count':len(p),'world_xyz_max_error_m':maxw,'common_xyz_max_error_m':maxq,'reprojection_max_px':maxpix,'robot_filter_violations':robot_bad,'supported_representatives':len(si),'different_frame_witness_verified':True,'supported_distance_reference_gt1mm':int((dref>.001).sum()),'supported_distance_reference_gt2mm':int((dref>.002).sum()),'supported_distance_main_gt1mm':int((dmain>.001).sum()),'supported_distance_main_gt2mm':int((dmain>.002).sum()),'unknown_reasons_overlap':{'outside_reference_association':int((z['ref_distance']>.003).sum()),'unstable':int((~z['stable']).sum()),'no_seed_component':int((~z['connected']).sum()),'normal_inconsistent':int((~z['normal_consistent']).sum()),'fewer_than_2_times':int((actualcount<2).sum()),'appearance_inconsistent':int((z['color_distance']>c['color_l2_limit_q95']+1e-12).sum())}}
  rows.append(result);print(json.dumps(result),flush=True)
 robot.close()
 report={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),'locks':locks,'cases':rows,'no_scoring_training':True,'new_contact_labels':False,'limits':['fusion sources remain weak object identity associations','1mm cells are not independent surface area','signed depth behind is not proof of occlusion or contact']}
 (R/'audits/completion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:(R/'audit_failure.log').write_text(traceback.format_exc());print(json.dumps({'error':type(e).__name__}));raise SystemExit(1)
