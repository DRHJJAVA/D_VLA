from pathlib import Path
import json,numpy as np,h5py
from PIL import Image
B=Path('/mnt/nvme1/drh/D_VLA');R=B/'experiments/contact_preference/multiview_temporal_fusion_v1'
(R/'webdata').mkdir(exist_ok=True);(R/'images').mkdir(exist_ok=True)
S=json.loads((R/'fusion_review_summary.json').read_text());A=json.loads((R/'audits/completion.json').read_text())
def arr(p):return np.round(p,7).tolist()
for c in S['cases']:
 ci=c['case'];key=c['key'];z=np.load(R/'refined_cases'/(key+'.npz'));si=z['supported_representatives'];ui=z['unknown_representatives']
 ui=ui[np.linspace(0,len(ui)-1,min(len(ui),3000),dtype=int)]
 reg=np.load(B/'experiments/region_pose/iter_024/train'/(key+'.npz'))
 _,ri=np.unique(np.floor(z['reference_xyz']/.001).astype(int),axis=0,return_index=True)
 data={'key':key,'query':c['query_frame'],'ref':{'points':arr(z['reference_xyz'][ri]),'rgb':z['reference_rgb'][ri].tolist(),'pixel':z['reference_pixel'][ri].tolist(),'original_index':ri.tolist()},'fused':{'points':arr(z['common_xyz'][si]),'rgb':z['rgb'][si].tolist(),'pixel':z['pixel'][si].tolist(),'frame':z['frame'][si].tolist(),'camera':z['camera'][si].tolist(),'source_id':si.tolist(),'new':z['new_vs_reference'][si].astype(int).tolist(),'wrist_extra':z['wrist_new_voxel'][si].astype(int).tolist(),'support_frames':z['distinct_frame_support'][si].tolist(),'shift_mm':arr(z['loo_shift'][si]*1000)},'unknown':{'points':arr(z['common_xyz'][ui]),'source_id':ui.tolist(),'pixel':z['pixel'][ui].tolist(),'frame':z['frame'][ui].tolist(),'camera':z['camera'][ui].tolist(),'reason':[]},'views':{},'images':{},'audit':A['cases'][ci]}
 for i in ui:
  reason=[]
  if z['ref_distance'][i]>.003:reason.append('超出已知物体表面关联范围')
  if not z['normal_consistent'][i]:reason.append('表面方向不一致/退化')
  if not z['stable'][i]:reason.append('留一配准不稳定')
  if not z['connected'][i]:reason.append('不与可信目标来源连通')
  if z['distinct_frame_support'][i]<2:reason.append('不同帧支持不足')
  if z['color_distance'][i]>c['color_l2_limit_q95']+1e-12:reason.append('外观对应不足')
  data['unknown']['reason'].append('；'.join(reason) or '未满足所有来源检查')
 asset=key.split('_demo_')[0];demo='demo_'+key.split('_demo_')[1].split('_')[0]
 with h5py.File(B/'data/v3/trainval_rgbd_448'/asset/(demo+'.h5'),'r') as h:
  for t in c['frames']:
   for camid,cam in enumerate(['agentview','robot0_eye_in_hand']):
    oldname=('c%d_query.jpg'%ci) if t==c['query_frame'] and camid==0 else 'c%d_f%d_%s.jpg'%(ci,t,cam)
    old=B/'experiments/contact_preference/stage3_pilot_v1/images'/oldname
    if old.exists():url='/图示/阶段3支持/images/'+oldname
    else:
     name='c%d_f%d_%s.jpg'%(ci,t,cam);Image.fromarray(h[cam+'/rgb'][t]).save(R/'images'/name,quality=95);url='/图示/跨时双相机融合/images/'+name
    data['images'][str(t)+'_'+str(camid)]=url
   if t==c['query_frame']:continue
   m=np.load(R/'refined_cases'/(key+'_view%d.npz'%t));Q=reg['query_to_current_hand_%d'%t];W=reg['hand_to_world_%d'%t]@Q
   st=next(x for x in c['contact_vicinity_review'] if x['frame']==t)
   st['reference_near_representative_cells']=int(m['reference_near_finger'][ri].sum())
   v={'Q':arr(Q),'near':m['near_without_front_conflict'].astype(int).tolist(),'near_unknown':m['near_unknown'].astype(int).tolist(),'reference_near':m['reference_near_finger'][ri].astype(int).tolist(),'depth_class':m['depth_class'].tolist(),'stats':st}
   for short,cam in [('main','agentview'),('wrist','robot0_eye_in_hand')]:
    v[short]={'K':arr(h[cam+'/K'][t]),'C':arr(np.linalg.solve(h[cam+'/camera_to_world'][t].astype(float),W))}
   data['views'][str(t)]=v
 (R/'webdata'/('case%d.json'%ci)).write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
 c['audit']=A['cases'][ci]
(R/'webdata/summary.json').write_text(json.dumps(S,ensure_ascii=False,indent=2))
link=B/'docs/图示/跨时双相机融合'
if not link.exists():link.symlink_to(R,target_is_directory=True)
print(json.dumps({'cases':len(S['cases']),'new_images':len(list((R/'images').glob('*.jpg'))),'known_images_reused':27}))
