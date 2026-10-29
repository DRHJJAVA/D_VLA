from pathlib import Path
import json,hashlib,numpy as np,h5py,traceback
from scipy.spatial import cKDTree
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo
B=Path('/mnt/nvme1/drh/D_VLA');D=B/'experiments/contact_preference/contact_location_5cam_v1';F=B/'experiments/contact_preference/multiview_temporal_fusion_v1';T=B/'experiments/contact_preference/contact_location_reliability_v1'
def save(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2))
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(8*1024*1024),b''):h.update(v)
 return h.hexdigest()
def vox(a):return set(map(tuple,np.floor(a/.001).astype(np.int64)))
def main():
 cfg=json.loads((D/'config.json').read_text());summary=[];maxuv=0.;maxdepth=0.
 fusionSummary=json.loads((D/'fusion/refined_summary.json').read_text())
 for ci,c in enumerate(cfg['cases']):
  k=c['key'];z=np.load(D/'fusion/refined_cases'/f'{k}.npz');old=np.load(F/'refined_cases'/f'{k}.npz')
  a=json.loads((T/'webdata'/f'case{ci}.json').read_text());b=json.loads((D/'reliability/webdata'/f'case{ci}.json').read_text())
  ov=vox(old['common_xyz'][old['supported']]);nv=vox(z['common_xyz'][z['supported']])
  cs=[];row=fusionSummary['cases'][ci];cameraSets=[]
  for cam in range(5):cameraSets.append(vox(z['common_xyz'][z['supported']&(z['camera']==cam)]))
  with h5py.File(D/'rgbd'/c['asset']/(c['demo']+'.h5')) as h:
   for cam,name in enumerate(cfg['cameras']):
    rr=[r for r in row['rows'] if r['camera']==cam];C=h[name+'/camera_to_world'][:]
    cs.append({'name':name,'roi':sum(x['roi_count'] for x in rr),'robot_rejected':sum(x['robot_rejected'] for x in rr),'observations':int((z['camera']==cam).sum()),'supported_observations':int((z['supported']&(z['camera']==cam)).sum()),'supported_voxels':len(cameraSets[cam]),'exclusive_voxels':len(cameraSets[cam]-set().union(*[cameraSets[j] for j in range(5) if j!=cam])),'unknown_observations':int((z['unknown']&(z['camera']==cam)).sum()),'fixed_extrinsic':bool(np.max(abs(C-C[:1]))<1e-7),'frames':[{key:x[key] for key in ['frame','source_count','supported','unknown','robot_rejected']} for x in rr]})
    for j,t in enumerate(h['frame_indices'][:]):
     ids=np.flatnonzero((z['camera']==cam)&(z['frame']==t))
     if not len(ids):continue
     C=h[name+'/camera_to_world'][j];K=h[name+'/K'][j];dep=h[name+'/depth_m'][j];pix=z['pixel'][ids].astype(int)
     cp=(z['source_world_xyz'][ids]-C[:3,3])@np.linalg.inv(C[:3,:3]).T
     uv=cp[:,:2]/cp[:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
     e=float(np.max(abs(uv-pix)));de=float(np.max(abs(cp[:,2]-dep[pix[:,1],pix[:,0]])));maxuv=max(maxuv,e);maxdepth=max(maxdepth,de)
     assert e<.001 and de<1e-5
   # Query views are additional observations only, not contact truth.
   j=int(np.flatnonzero(h['frame_indices'][:]==c['query'])[0]);queryviews=[]
   for vi,name in enumerate(cfg['cameras']):
    Image.fromarray(h[name+'/rgb'][j]).save(D/'images'/f'q{ci}_v{vi}.jpg',quality=94)
    queryviews.append({'name':name,'image':f'/图示/五相机对照/images/q{ci}_v{vi}.jpg','K':h[name+'/K'][j].tolist(),'C':h[name+'/camera_to_world'][j].tolist()})
  result={'title':c['title'],'key':k,'query':c['query'],'camera_names':cfg['cameras'],'camera_contributions':cs,'fusion':{'two_voxels':len(ov),'five_voxels':len(nv),'added_voxels':len(nv-ov),'removed_voxels':len(ov-nv),'new_camera_observations':int((z['camera']>=2).sum()),'new_camera_supported':int((z['supported']&(z['camera']>=2)).sum()),'caveat':'1mm observed cells, not contact area or independent trials; adaptive color/normal evidence can also change shared old-camera membership'},'frames':[],'query_views':queryviews,'traces':b['traces'],'two_traces':a['traces'],'cycles':b['cycles'],'two_cycles':a.get('active_finger_cycles',[])}
  reg=np.load(B/'experiments/region_pose/iter_024/train'/f'{k}.npz')
  for i,t in enumerate(c['frames']):
   fa=a['frames'][i];fb=b['frames'][i];assert fa['frame']==fb['frame']==t
   ix=old['supported_representatives'];ix=ix[np.linspace(0,len(ix)-1,min(5000,len(ix)),dtype=int)]
   Q=reg[f'query_to_current_hand_{t}'];rgb=old['rgb'][ix];rgb=rgb*255 if rgb.max()<=1 else rgb
   fa['object']=(old['common_xyz'][ix]@Q[:3,:3].T+Q[:3,3]).tolist();fa['object_rgb']=rgb.astype(int).tolist()
   # Match new and old cameras/viewpoint when displaying either source inventory.
   fa['views']=fb['views']
   for f in [fa,fb]:
    for vn,pts in f['variants'].items():
     if not pts:f['reports'][vn]['observation_states']=[[0]*5 for _ in cfg['cameras']]
     for p in pts:
      p['patch']=p['patch'][::max(1,len(p['patch'])//60)]
   fv={'frame':t,'views':fb['views'],'two':{k:fa[k] for k in ['variants','reports','object','object_rgb','visual_vertices','visual_faces','gripper_vertices','gripper_faces']},'five':{k:fb[k] for k in ['variants','reports','object','object_rgb','visual_vertices','visual_faces','gripper_vertices','gripper_faces']},'old_points':fb['old_points'],'geometry':fb['geometry']}
   # Raw registered reference anchor check in all 5 views is kept separately from fit results.
   result['frames'].append(fv)
  save(D/'webdata'/f'case{ci}.json',result)
  summary.append({'case':c['title'],'fusion':result['fusion'],'camera_contributions':cs,'frames':[{'frame':f['frame'],'two_collision':f['two']['reports']['collision_leave_frame_out']['accepted'],'five_collision':f['five']['reports']['collision_leave_frame_out']['accepted'],'two_visual':f['two']['reports']['visual_leave_frame_out']['accepted'],'five_visual':f['five']['reports']['visual_leave_frame_out']['accepted'],'five_visual_finger_occluded':f['five']['reports']['visual_leave_frame_out'].get('active_visual_finger_occluded',0)} for f in result['frames']],'five_active_finger_reappearance':result['cycles'],'two_active_finger_reappearance':result['two_cycles']})
 save(D/'comparison_summary.json',{'time':now(),'cameras':cfg['cameras'],'cases':summary,'reprojection_max_px':maxuv,'depth_reconstruction_max_m':maxdepth,'source_camera_ids_checked':True,'no_contact_truth':True,'scoring_training':False})
 # Lock all prior data and verify they remain unchanged, without changing old files.
 oldlocks=[F/'completed_locked.json',B/'experiments/contact_preference/occluded_surface_fit_v1/completed_locked.json',B/'experiments/contact_preference/occluded_surface_fit_v1/review_fix_01/completed_locked.json']
 counts=[]
 for p in oldlocks:
  lock=json.loads(p.read_text());assert all(sha(Path(k))==v for k,v in lock.items());counts.append(len(lock))
 for p in [D/'raw_input_locked.json',D/'source_locked.json',D/'fusion_input_locked.json',D/'reliability/input_locked.json']:
  lock=json.loads(p.read_text());assert all(sha(Path(k))==v for k,v in lock.items())
 for ci,c in enumerate(cfg['cases']):
  z=np.load(D/'fusion/refined_cases'/(c['key']+'.npz'))
  for t in c['frames']:
   s=np.load(D/'reliability/data'/f'c{ci}_f{t}_sources.npz');ix=s['source_ids'];assert (z['frame'][ix]!=t).all();assert np.array_equal(s['cloud'],z['common_xyz'][ix])
 save(D/'audits/completion_numeric.json',{'time':now(),'prior_lock_counts':counts,'input_locks_pass':True,'heldout_source_frame_exclusion':True,'all_source_pixels_reprojected':True,'reprojection_max_px':maxuv,'depth_max_m':maxdepth,'report_zero_array_alignment':'empty estimate arrays expanded to five camera display slots only; predictions unchanged'})
 print(json.dumps({'packed':len(summary),'reprojection_max_px':maxuv,'depth_max_m':maxdepth,'old_locks':counts}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(D/'pack_failure.json',{'type':type(e).__name__,'message':str(e),'stack':[(x.filename,x.lineno) for x in traceback.extract_tb(e.__traceback__)]});traceback.print_exc(file=(D/'pack_failure.log').open('w'))
