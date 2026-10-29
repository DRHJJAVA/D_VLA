import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import json,importlib.util,numpy as np,h5py,traceback
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
D=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_preference/contact_location_reliability_v1')
sp=importlib.util.spec_from_file_location('run_reliability',D/'src/run.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
def main():
 cfg=json.loads((D/'config.json').read_text());robot=r.m.RobotDepth(448);rows=[];maxerr=0.
 for ci,c in enumerate(cfg['cases']):
  web=json.loads((D/'webdata'/f'case{ci}.json').read_text());k=c['key'];reg=np.load(r.R/'experiments/region_pose/iter_024/train'/f'{k}.npz');src=np.load(r.m.F/'refined_cases'/f'{k}.npz')
  ref=reg['query_xyz'];querytree=cKDTree(ref[r.unique_ids(ref)]);density={}
  for a in [t['anchor'] for t in web['traces']]:density[str(a)]=[len(querytree.query_ball_point(ref[a],rad)) for rad in [.003,.005]]
  web['query_patch_counts']=density
  with h5py.File(r.R/'data/v3/trainval_rgbd_448'/c['asset']/(c['demo']+'.h5'),'r') as h:
   for fi,t in enumerate(c['frames']):
    fr=web['frames'][fi];Q=reg[f'query_to_current_hand_{t}'];H=reg[f'hand_to_world_{t}'];q=reg[f'finger_q_{t}'];j=int(np.where(h['frame_indices'][:]==t)[0][0])
    saved=np.load(D/'data'/f'c{ci}_f{t}_sources.npz');sid=saved['source_ids'];assert (src['frame'][sid]!=t).all();assert np.array_equal(saved['cloud'],src['common_xyz'][sid])
    assert len(np.unique(np.floor(saved['cloud']/.001).astype(np.int64),axis=0))==len(sid)
    for vi,cn in enumerate(['agentview','robot0_eye_in_hand']):
     C=h[cn+'/camera_to_world'][j];K=h[cn+'/K'][j];dep=h[cn+'/depth_m'][j];rd=robot.render(q,np.linalg.solve(H,C),K);vd=r.render_group(robot,[2,5]);cd=r.render_group(robot,[3,4,6,7])
     common=binary_erosion((vd>0)&(vd<1)&(cd>0)&(cd<1),iterations=2)
     exposed=common&(abs(vd-rd)<1e-5)
     fr['geometry'][vi]['not_occluded_by_other_robot_pixels']=int(exposed.sum())
     fr['geometry'][vi]['exposed_visual_residual_mm']=r.stats((dep-vd)[exposed]*1000)
     fr['geometry'][vi]['exposed_collision_residual_mm']=r.stats((dep-cd)[exposed]*1000)
     fr['geometry'][vi]['exposed_visual_within3mm']=int((exposed&(abs(dep-vd)<=.003)).sum())
     fr['geometry'][vi]['exposed_collision_within3mm']=int((exposed&(abs(dep-cd)<=.003)).sum())
     groups=[fr['old_points']]+list(fr['variants'].values())
     for pts in groups:
      for p in pts:
       obs=p['observations'][vi];uv=np.rint(obs['uv']).astype(int);valid=(uv>=0).all() and (uv<448).all()
       z=r.m.apply(np.linalg.solve(C,H),np.array([p['hand']]))[0,2]
       obs['active_visual_finger_occluded']=bool(valid and vd[uv[1],uv[0]]<z-.003 and abs(vd[uv[1],uv[0]]-rd[uv[1],uv[0]])<1e-5 and abs(dep[uv[1],uv[0]]-vd[uv[1],uv[0]])<=.003)
     for tr in web['traces']:
      obs=tr['frames'][fi]['observations'][vi];uv=np.rint(obs['uv']).astype(int);valid=(uv>=0).all() and (uv<448).all()
      ph=r.m.apply(Q,ref[tr['anchor']][None]);z=r.m.apply(np.linalg.solve(C,H),ph)[0,2]
      obs['active_visual_finger_occluded']=bool(valid and vd[uv[1],uv[0]]<z-.003 and abs(vd[uv[1],uv[0]]-rd[uv[1],uv[0]])<1e-5 and abs(dep[uv[1],uv[0]]-vd[uv[1],uv[0]])<=.003)
     rgb=h[cn+'/rgb'][j];im=Image.fromarray(rgb.astype(np.uint8)).convert('RGB');arr=np.asarray(im).copy()
     for depth,color in [(cd,[230,60,164]),(vd,[0,230,230])]:
      mask=(depth>0)&(depth<1);edge=mask&~binary_erosion(mask)
      arr[edge]=color
     Image.fromarray(arr).save(D/'images'/f'geometry_c{ci}_f{t}_v{vi}.jpg',quality=94)
     fr['views'][vi]['geometry_image']=f'/图示/接触位置可靠性/images/geometry_c{ci}_f{t}_v{vi}.jpg'
    for name,pts in fr['variants'].items():
     fr['reports'][name]['active_visual_finger_occluded']=sum(any(o['active_visual_finger_occluded'] for o in p['observations']) for p in pts)
     for p in pts:
      # Nominal point remains exactly on original sampled ray, never snapped to zero gap.
      recon=np.array(p['gripper'])+p['gap']*.001*np.array(p['normal'])
      err=float(np.linalg.norm(recon-np.array(p['hand'])));maxerr=max(maxerr,err);assert err<1e-8
    rows.append({'case':c['title'],'frame':t,'geometry':fr['geometry'],'variants':fr['reports']})
  web['active_finger_cycles']=[]
  for tr in web['traces']:
   for vi in [0,1]:
    occ=[f for f in tr['frames'] if f['observations'][vi]['active_visual_finger_occluded']]
    for first in occ:
     later=[f for f in tr['frames'] if f['frame']>first['frame'] and f['observations'][vi]['state']==0]
     if later:web['active_finger_cycles'].append({'anchor':tr['anchor'],'camera':vi,'occluded_frame':first['frame'],'reappeared_frame':later[0]['frame'],'residual_mm':later[0]['observations'][vi]['depth_residual_mm']});break
  r.save(D/'webdata'/f'case{ci}.json',r.clean(web))
 a=json.loads((D/'summary.json').read_text());a['rows']=rows
 for ci,ca in enumerate(a['cases']):
  w=json.loads((D/'webdata'/f'case{ci}.json').read_text());ca['active_finger_cycles']=w['active_finger_cycles'];ca['query_patch_counts']=w['query_patch_counts']
 a['max_ray_reconstruction_m']=maxerr;a['source_exclusion_independently_reconstructed']=True
 r.save(D/'summary.json',r.clean(a));robot.close();r.save(D/'audits/independent_checks.json',{'time':r.m.now(),'source_exclusion':True,'voxel_representative_is_real':True,'ray_reconstruction_max_m':maxerr,'finger_occlusion_distinguished_from_robot':True,'diagnostic_contours':'known robot only, not contact ground truth'})
 print(json.dumps({'status':'postcheck_complete','ray_error_m':maxerr,'active_finger_cycles':[len(x['active_finger_cycles']) for x in a['cases']]}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  r.save(D/'post_failure.json',{'type':type(e).__name__,'message':str(e),'stack':[(x.filename,x.lineno) for x in traceback.extract_tb(e.__traceback__)]});traceback.print_exc(file=(D/'post_failure.log').open('w'))
