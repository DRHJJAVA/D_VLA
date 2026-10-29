from pathlib import Path
import json,h5py,numpy as np,hashlib,datetime
root=Path('/mnt/nvme1/drh/D_VLA')
out=root/'experiments/contact_preference/fixed_camera_pilot_v1'
cfg=json.loads((out/'config.json').read_text())
old=root/'experiments/contact_frames/cf017'
stage=json.loads((root/'experiments/contact_preference/stage012_v1/stage2_audit.json').read_text())
contact={'stove_demo_0':86,'drawer_demo_0':70,'two_objects_demo_0':300}
report={'created_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),'cases':{}}
def cloud(d,K,C,step=3):
    h,w=d.shape
    v,u=np.mgrid[0:h:step,0:w:step]
    u=u.ravel();v=v.ravel();z=d[v,u].astype(np.float64)
    valid=np.isfinite(z)&(z>0)&(z<3)
    u=u[valid];v=v[valid];z=z[valid]
    x=(u-K[0,2])*z/K[0,0];y=(v-K[1,2])*z/K[1,1]
    X=np.stack([x,y,z],axis=1)
    return X@C[:3,:3].T+C[:3,3]
for name,c in cfg['cases'].items():
    added=out/(name+'.h5')
    base=old/'rgbd'/(name+'.h5')
    motion=np.load(old/'cases'/name/'motion.npz')
    H=motion['hand_to_world'][contact[name]]
    hand=H[:3,:3]@np.array([0,0,0.097])+H[:3,3]
    patch=stage['cross_task_localization'][name]
    radius=patch['hand_radius_m_diagnostic']
    Hr=motion['hand_to_world'][patch['reference_frame']]
    qr=motion['measured_finger_q'][patch['reference_frame']]
    centers=np.stack([Hr[:3,:3]@np.array([0,qr[0],0.093])+Hr[:3,3],Hr[:3,:3]@np.array([0,qr[1],0.093])+Hr[:3,3],Hr[:3,:3]@np.array([0,0,0.097])+Hr[:3,3]])
    with np.load(old/'cases'/name/'artifacts/reference.npz') as z:
        xyz=z['surface_xyz']
        keep=np.min(np.linalg.norm(xyz[:,None,:]-centers[None,:,:],axis=2),axis=1)<radius
        source=xyz[keep]
    print('source_compare',name,len(source),patch['initial_true_depth_points'])
    assert len(source)==patch['initial_true_depth_points']
    r={'source_patch_points':len(source),'contact_review_frame':contact[name],'fixed_views':{},'near_hand_8cm':{}}
    with h5py.File(added,'r') as n,h5py.File(base,'r') as b:
        frames=n['frame_indices'][:].astype(int).tolist();r['frames']=frames
        assert frames==c['frames']
        assert len(frames)==n.attrs['num_frames']
        slot0=frames.index(0);slot=frames.index(contact[name])
        all_clouds={}
        for cam in ['agentview','robot0_eye_in_hand','frontview','sideview','birdview']:
            h=b if cam in b else n
            ix=contact[name] if h is b else slot
            ix0=0 if h is b else slot0
            d=h[cam]['depth_m'][ix]
            K=h[cam]['K'][ix];C=h[cam]['camera_to_world'][ix]
            assert np.allclose(K[2],[0,0,1],atol=1e-5)
            assert np.allclose(C[3],[0,0,0,1],atol=1e-5)
            const_extrinsic=bool(np.allclose(h[cam]['camera_to_world'][0],h[cam]['camera_to_world'][-1],atol=1e-7)) if h is n else None
            m=np.linalg.inv(h[cam]['camera_to_world'][ix0])
            local=source@m[:3,:3].T+m[:3,3]
            z=local[:,2]
            uf=np.rint(K[0,0]*local[:,0]/z+K[0,2]).astype(int)
            vf=np.rint(K[1,1]*local[:,1]/z+K[1,2]).astype(int)
            inside=(z>0)&(uf>=0)&(uf<d.shape[1])&(vf>=0)&(vf<d.shape[0])
            depth0=h[cam]['depth_m'][ix0]
            visible=np.zeros(len(source),bool)
            j=np.flatnonzero(inside)
            visible[j]=np.abs(depth0[vf[j],uf[j]]-z[j])<=0.005
            P=cloud(d,K,C,step=3)
            near=P[np.linalg.norm(P-hand,axis=1)<0.08]
            cells=np.unique(np.floor(near/0.003).astype(np.int64),axis=0)
            all_clouds[cam]=cells
            r['fixed_views'][cam]={'fixed_extrinsic':const_extrinsic,'valid_depth_fraction':float(np.mean(np.isfinite(d)&(d>0))),'source_patch_in_image':int(inside.sum()),'source_patch_depth_consistent_5mm':int(visible.sum())}
            r['near_hand_8cm'][cam]={'sampled_points':int(len(near)),'voxel3mm':int(len(cells))}
        oldcells=np.concatenate([all_clouds[x] for x in ['agentview','robot0_eye_in_hand']])
        newcells=np.concatenate([all_clouds[x] for x in ['frontview','sideview','birdview']])
        oldset=set(map(tuple,oldcells));newset=set(map(tuple,newcells))
        r['near_hand_8cm']['union']={'old_voxels':len(oldset),'added_voxels':len(newset),'new_unique_voxels':len(newset-oldset),'all_voxels':len(oldset|newset)}
        r['new_h5_sha256']=hashlib.sha256(added.read_bytes()).hexdigest()
    report['cases'][name]=r
(out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:{'patch':v['source_patch_points'],'new_unique':v['near_hand_8cm']['union']['new_unique_voxels'],'fixed':{c:x['fixed_extrinsic'] for c,x in v['fixed_views'].items() if x['fixed_extrinsic'] is not None},'visible':{c:x['source_patch_depth_consistent_5mm'] for c,x in v['fixed_views'].items()}} for k,v in report['cases'].items()}))
