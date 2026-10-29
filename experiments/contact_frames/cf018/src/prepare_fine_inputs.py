from pathlib import Path
import h5py,json,hashlib,numpy as np,time
from PIL import Image
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018')
cases=json.loads((R/'episode_manifest.json').read_text())['cases']
out=R/'prepared_fine'
out.mkdir(exist_ok=True)
summary=[]
for case in cases:
 name=case['case']
 folder=out/name
 folder.mkdir(exist_ok=True)
 n=case['frames']
 ids=list(range(n))
 if ids[-1]!=n-1:ids.append(n-1)
 sensor=[]
 image_hashes={}
 with h5py.File(case['rgbd_path']) as v,h5py.File(case['raw_path']) as d:
  obs=d[f"data/demo_{case['demo_index']}/obs"]
  acts=d[f"data/demo_{case['demo_index']}/actions"]
  assert np.array_equal(v['frame_indices'][:],np.arange(n))
  for t in ids:
   row={'frame':t,'time_s':round(t/20,3),'obs_index':t-1 if t>=1 else None,'state_valid':t>=1,'camera':{}}
   if t>=1:
    ee=obs['ee_pos'][t-1].astype(float)
    row.update({'ee_pos_m':ee.round(5).tolist(),'ee_ori_raw':obs['ee_ori'][t-1].astype(float).round(5).tolist(),'gripper_joint_m':obs['gripper_states'][t-1].astype(float).round(6).tolist(),'action_raw':acts[t-1].astype(float).round(5).tolist()})
   for cam in ['agentview','robot0_eye_in_hand']:
    rgb=v[f'{cam}/rgb'][t]
    dest=folder/f'{t:04d}_{cam}.jpg'
    Image.fromarray(rgb).save(dest,quality=92,subsampling=0)
    image_hashes[dest.name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    datum={'image':dest.name}
    if t>=1:
     T=v[f'{cam}/camera_to_world'][t].astype(float)
     K=v[f'{cam}/K'][t].astype(float)
     xyz=np.linalg.solve(T,np.r_[ee,1.])[:3]
     if xyz[2]>0:
      px=K@xyz
      u,vv=(px[:2]/px[2])
      ui,vi=int(round(u)),int(round(vv))
      dep=v[f'{cam}/depth_m'][t]
      if 5<=ui<dep.shape[1]-5 and 5<=vi<dep.shape[0]-5:
       patch=dep[vi-5:vi+6,ui-5:ui+6]
       valid=np.isfinite(patch)&(patch>0)
       if valid.any():
        datum.update({'ee_projected_uv':[round(float(u),1),round(float(vv),1)],'local_depth_median_m':round(float(np.median(patch[valid])),4),'local_depth_valid_fraction':round(float(valid.mean()),3),'ee_camera_z_m':round(float(xyz[2]),4),'depth_identity':'unassigned'})
    row['camera'][cam]=datum
   sensor.append(row)
 (folder/'sensors.json').write_text(json.dumps({'case':name,'time_alignment':'render frame t reads obs/action t-1 for t>=1, t0 state unknown','fps':20,'rows':sensor},ensure_ascii=False,indent=2))
 (folder/'images_sha256.json').write_text(json.dumps(image_hashes,indent=2))
 summary.append({'case':name,'frames':n,'sampled_frames':len(ids),'images':len(image_hashes),'sensor_rows':len(sensor)})
(R/'prepared_summary_fine.json').write_text(json.dumps({'created_at':time.strftime('%Y-%m-%d %H:%M:%S%z'),'cases':summary,'source_manifest_sha256':hashlib.sha256((R/'episode_manifest.json').read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
print(json.dumps({'stage':'prepared','cases':len(summary),'images':sum(x['images'] for x in summary),'sampled_frames':sum(x['sampled_frames'] for x in summary)}))
