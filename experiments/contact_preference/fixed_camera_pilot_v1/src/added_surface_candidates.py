from pathlib import Path
import json,h5py,numpy as np,datetime
from scipy.spatial import cKDTree
root=Path('/mnt/nvme1/drh/D_VLA');out=root/'experiments/contact_preference/fixed_camera_pilot_v1'
cfg=json.loads((out/'config.json').read_text())
stage=json.loads((root/'experiments/contact_preference/stage012_v1/stage2_audit.json').read_text())['cross_task_localization']
old=root/'experiments/contact_frames/cf017'
results={}
for name,c in cfg['cases'].items():
    rec=stage[name];H=np.load(old/'cases'/name/'motion.npz')['hand_to_world'][rec['reference_frame']]
    q=np.load(old/'cases'/name/'motion.npz')['measured_finger_q'][rec['reference_frame']]
    centers=np.stack([H[:3,:3]@np.array([0,q[0],0.093])+H[:3,3],H[:3,:3]@np.array([0,q[1],0.093])+H[:3,3],H[:3,:3]@np.array([0,0,0.097])+H[:3,3]])
    with np.load(old/'cases'/name/'artifacts/reference.npz') as z:X=z['surface_xyz']
    prior=X[np.min(np.linalg.norm(X[:,None,:]-centers[None,:,:],axis=2),axis=1)<rec['hand_radius_m_diagnostic']]
    assert len(prior)==rec['initial_true_depth_points']
    tree=cKDTree(prior)
    outputs={}
    all_new=[]
    with h5py.File(out/(name+'.h5'),'r') as h:
        for cam in cfg['new_fixed_cameras']:
            d=np.asarray(h[cam]['depth_m'][0],float)
            K=np.asarray(h[cam]['K'][0],float);C=np.asarray(h[cam]['camera_to_world'][0],float)
            v,u=np.indices(d.shape);u=u.ravel();v=v.ravel();z=d.ravel()
            valid=np.isfinite(z)&(z>0)&(z<3)
            u=u[valid];v=v[valid];z=z[valid]
            xc=np.stack([(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z],axis=1)
            world=xc@C[:3,:3].T+C[:3,3]
            near=np.min(np.linalg.norm(world[:,None,:]-centers[None,:,:],axis=2),axis=1)<rec['hand_radius_m_diagnostic']
            Xn=world[near]
            dist=tree.query(Xn,k=1)[0] if len(Xn) else np.zeros(0)
            outputs[cam]={'raw_depth_points_in_same_hand_neighborhood':int(len(Xn)),'beyond_3mm_from_prior_source_points':int(np.sum(dist>0.003))}
            all_new.append(Xn)
    all_new=np.concatenate(all_new) if all_new else np.empty((0,3))
    dist=tree.query(all_new,k=1)[0] if len(all_new) else np.zeros(0)
    novel=all_new[dist>0.003]
    cells=np.floor(novel/0.001).astype(np.int64)
    _,index=np.unique(cells,axis=0,return_index=True)
    uniq=novel[np.sort(index)]
    np.savez_compressed(out/(name+'_new_source_candidates.npz'),new_surface_points=uniq,all_added_points=all_new,prior_surface_points=prior)
    results[name]={'prior_source_points':len(prior),'views':outputs,'new_raw_points':len(all_new),'new_gt3mm_unique_1mm_representatives':len(uniq),'definition':'same hand-neighborhood geometry at original source frame0; no object identity or physical-contact claim'}
(out/'new_source_candidate_audit.json').write_text(json.dumps({'created_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),'cases':results},ensure_ascii=False,indent=2))
print(json.dumps({k:{'new_unique':v['new_gt3mm_unique_1mm_representatives'],'per_camera':v['views']} for k,v in results.items()}))
