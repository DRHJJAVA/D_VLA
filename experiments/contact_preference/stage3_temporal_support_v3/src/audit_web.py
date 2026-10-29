from pathlib import Path
import json,numpy as np,h5py
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/stage3_temporal_support_v3'
s=json.loads((O/'summary.json').read_text());rows=[];uris=set();maxerr=0
for ci,c in enumerate(s['cases']):
 d=json.loads((O/f'webdata/case{ci}.json').read_text())
 with h5py.File(R/'data/v3/trainval_rgbd_448'/c['asset']/(c['demo']+'.h5'),'r') as h:
  for t in c['frames']:
   z=dict(np.load(O/'points'/f"{c['key']}_frame{t}.npz"));v=d['views'][str(t)]
   assert np.array_equal(d['pixel'],z['query_pixel']);assert np.array_equal(d['source_index'],z['query_source_index'])
   assert np.max(abs(np.array(v['world'])-z['mapped_world_xyz']))<5.01e-9
   for key in ['positive','separation','unknown','seed','carry','motion_support','continuity','geometry_positive','geometry_separation','baseline_positive','baseline_separation','baseline_unknown']:
    assert np.max(abs(np.array(v[key])-z[key]))<5.01e-7
   for key in ['category','baseline_category','geometry_category','visibility','camera_visibility','excluded_scope','penetration','opposing']:assert np.array_equal(v[key],z[key])
   j=int(np.where(h['frame_indices'][:]==t)[0][0])
   for cam,key in [('agentview','main'),('robot0_eye_in_hand','wrist')]:
    C=np.array(v[key]['C']);K=np.array(v[key]['K'])
    assert np.array_equal(C,h[cam]['camera_to_world'][j]) and np.array_equal(K,h[cam]['K'][j])
    # Independently compare exact solve with display adjugate inverse.
    p=z['mapped_world_xyz'];a,b,cc=C[0,:3];dd,e,f=C[1,:3];g,hh,i=C[2,:3]
    inv=np.array([[e*i-f*hh,cc*hh-b*i,b*f-cc*e],[f*g-dd*i,a*i-cc*g,cc*dd-a*f],[dd*hh-e*g,b*g-a*hh,a*e-b*dd]])/np.linalg.det(C[:3,:3])
    q=(p-C[:3,3])@inv.T
    truth=np.linalg.solve(C,np.column_stack([p,np.ones(len(p))]).T).T[:,:3]
    maxerr=max(maxerr,float(np.max(abs(q-truth))))
    uv=q@K.T;uv=uv[:,:2]/uv[:,2:3];valid=(q[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<448)&(uv[:,1]>=0)&(uv[:,1]<448)
    rows.append({'case':ci,'frame':t,'camera':cam,'in_image':int(valid.sum()),'total':len(p)})
 for uri in d['images'].values():assert (R/'docs'/uri.lstrip('/')).exists();uris.add(uri)
assert maxerr<1e-12
result={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),'all_passed':True,'projection_inverse_max_m':maxerr,'current_calibration_exact':True,'numeric_rounding_m':5.01e-9,'support_rounding':5.01e-7,'image_urls':len(uris),'views':rows,'supersedes':'initial webdata uses reference-frame adapter; earlier empty-view diagnostic is not an actual camera visibility conclusion'}
(O/'audits/webdata_and_projection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False))
