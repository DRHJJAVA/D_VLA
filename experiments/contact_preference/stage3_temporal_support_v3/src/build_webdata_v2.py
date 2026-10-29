from pathlib import Path
import json,numpy as np,h5py
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/stage3_temporal_support_v3';F=R/'experiments/contact_preference/multiview_temporal_fusion_v1'
s=json.loads((O/'summary.json').read_text())
for ci,c in enumerate(s['cases']):
 f=json.loads((F/f'webdata_v2/case{ci}.json').read_text());result={'key':c['key'],'title':c['title']+' '+c['demo'],'query':f['query'],'frames':c['frames'],'images':f['images'],'views':{},'rows':c['rows']}
 for t in c['frames']:
  z=dict(np.load(O/'points'/f"{c['key']}_frame{t}.npz"))
  result['pixel']=z['query_pixel'].tolist();result['source_index']=z['query_source_index'].tolist()
  result['views'][str(t)]={'world':z['mapped_world_xyz'].round(8).tolist(),'main':f['views'][str(t)]['main'],'wrist':f['views'][str(t)]['wrist']}
  v=result['views'][str(t)]
  with h5py.File(R/'data/v3/trainval_rgbd_448'/c['asset']/(c['demo']+'.h5'),'r') as h:
   j=int(np.where(np.asarray(h['frame_indices'])==t)[0][0])
   for cam,key in [('agentview','main'),('robot0_eye_in_hand','wrist')]:
    v[key]={'K':h[cam]['K'][j].astype(float).tolist(),'C':h[cam]['camera_to_world'][j].astype(float).tolist()}

  for k in ['category','baseline_category','geometry_category','visibility','camera_visibility','excluded_scope','penetration','opposing']:
   v[k]=z[k].tolist()
  for k in ['positive','separation','unknown','baseline_positive','baseline_separation','baseline_unknown','geometry_positive','geometry_separation','seed','carry','motion_support','continuity']:
   v[k]=z[k].round(6).tolist()
  v['distance_mm']=(z['distance_m']*1000).round(5).tolist();v['sigma_mm']=(z['sigma_proxy_m']*1000).round(5).tolist()
 (O/f'webdata/case{ci}.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')))
summary={'cases':[{'id':i,'title':c['title']+' '+c['demo'],'key':c['key'],'frames':c['frames']} for i,c in enumerate(s['cases'])],'no_training':True}
(O/'webdata/summary.json').write_text(json.dumps(summary,ensure_ascii=False))
link=R/'docs/图示/阶段3遮挡支持'
if not link.exists():link.symlink_to(O,target_is_directory=True)
print('webdata_ready')
