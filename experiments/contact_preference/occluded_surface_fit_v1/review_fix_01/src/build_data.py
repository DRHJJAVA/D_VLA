import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import json,numpy as np,importlib.util,hashlib,traceback
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/occluded_surface_fit_v1';D=O/'review_fix_01'
sp=importlib.util.spec_from_file_location('fit',O/'src/fit_surfaces.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 lock=json.loads((O/'completed_locked.json').read_text());assert all(m.sha(p)==h for p,h in lock.items())
 cfg=json.loads((O/'config.json').read_text());robot=m.RobotDepth(448);audit={'time':m.now(),'old_lock_files':len(lock),'rows':[]};maxproj=0.
 inputs={str(O/'completed_locked.json'):m.sha(O/'completed_locked.json')}
 for ci,c in enumerate(cfg['cases']):
  key=c['key'];regpath=R/'experiments/region_pose/iter_024/train'/f'{key}.npz';reg=np.load(regpath);inputs[str(regpath)]=m.sha(regpath)
  web=json.loads((O/'webdata'/f'case{ci}.json').read_text());srcpath=R/'experiments/contact_preference/multiview_temporal_fusion_v1/refined_cases'/f'{key}.npz';src=np.load(srcpath);inputs[str(srcpath)]=m.sha(srcpath)
  ids=src['supported_representatives'];sub=np.linspace(0,len(ids)-1,min(3500,len(ids)),dtype=int);rgb=src['rgb'][ids[sub]]
  if rgb.max()<=1:rgb=rgb*255
  for fr in web['frames']:
   t=fr['frame'];z=np.load(O/'points'/f'{key}_frame{t}.npz');old=np.load(R/'experiments/contact_preference/stage3_pilot_v2/points_evidence'/f'{key}_frame{t}.npz');Q=z['Q'];H=z['H'];C=np.array(fr['views'][0]['C']);K=np.array(fr['views'][0]['K']);robot.render(reg[f'finger_q_{t}'],np.linalg.solve(H,C),K)
   visual,_=m.geo.surface(robot,[2,5]);collision,_=m.geo.surface(robot,[3,4,6,7])
   ids=np.array([p['id'] for p in fr['points']]);hand=m.apply(Q,z['estimated_ref'][ids]);dv,pv=m.geo.distance(visual,hand);dc,pc=m.geo.distance(collision,hand)
   fr['visual_vertices']=visual.vertices.tolist();fr['visual_faces']=visual.faces.tolist();fr['object_rgb']=rgb.astype(int).tolist();fr['hand_to_world']=H.tolist()
   for i,p in enumerate(fr['points']):
    idx=p['id'];a=int(z['anchor'][idx]);base=float(old['sigma_base_m'][a]*1000)
    p.update(visual_distance_mm=float(dv[i]*1000),collision_distance_mm=float(dc[i]*1000),base_sigma_mm=base,geometry_disagreement_ratio=float(abs(dv[i]-dc[i])*1000/base),visual_closest=pv[i].tolist(),collision_closest=pc[i].tolist())
    p['display_geometry_conflict']=p['geometry_disagreement_ratio']>1
    p['interval_hand']=[(np.array(p['gripper'])+v*.001*np.array(p['normal'])).tolist() for v in [p['lower'],p['upper']]]
   for v in fr['views']:
    C=np.array(v['C']);K=np.array(v['K']);T=np.linalg.solve(C,H);v['hand_to_camera']=T.tolist()
    cp=m.apply(T,hand);uv=cp[:,:2]/cp[:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]];err=float(np.max(abs(uv-v['estimate_uv'])));maxproj=max(maxproj,err);assert err<1e-3
    iv=np.array([p['interval_hand'] for p in fr['points']]);cam=m.apply(T,iv.reshape(-1,3));uv=cam[:,:2]/cam[:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]];v['interval_uv']=uv.reshape(-1,2,2).tolist()
    assert np.linalg.det(T[:3,:3])>0.99
   audit['rows'].append({'case':c['title'],'frame':t,'estimated':len(fr['points']),'active_occluded':fr['active_occluded_count'],'geometry_conflicts':sum(p['display_geometry_conflict'] for p in fr['points']),'active_points':[{'id':p['id'],'visual_mm':p['visual_distance_mm'],'collision_mm':p['collision_distance_mm'],'base_sigma_mm':p['base_sigma_mm'],'ratio':p['geometry_disagreement_ratio']} for p in fr['points'] if p['active_occluded']]})
   # All nominal results/labels must stay exactly unchanged.
   original=json.loads((O/'webdata'/f'case{ci}.json').read_text())['frames'][web['frames'].index(fr)] if False else None
   assert np.array_equal(np.array(fr['old_class']),old['display_class'])
   assert all(p['gap']==float(z['gap_m'][p['id']]*1000) and p['status']==int(z['status'][p['id']]) for p in fr['points'])
  save(D/'webdata'/f'case{ci}.json',web)
 robot.close();assert all(m.sha(p)==h for p,h in lock.items())
 audit['hand_to_camera_projection_max_px']=maxproj;audit['nominal_points_labels_unchanged']=True;audit['scoring_training']=False
 save(D/'audits/geometry_and_display.json',audit);save(D/'input_locked.json',inputs);save(D/'source_locked.json',{str(D/'src/build_data.py'):m.sha(D/'src/build_data.py')})
 print(json.dumps({'status':'data_ready','max_projection_px':maxproj,'rows':len(audit['rows'])}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(D/'failure.json',{'type':type(e).__name__,'message':str(e),'stack':[(x.filename,x.lineno) for x in traceback.extract_tb(e.__traceback__)]});traceback.print_exc(file=(D/'failure.log').open('w'))
