import os,sys,json,hashlib,traceback,ast,types,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ['MUJOCO_GL']='egl'
import numpy as np,h5py,importlib.util
from scipy.spatial import cKDTree
from PIL import Image
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/contact_preference/dual_camera_articulated_v1'
sys.path.insert(0,str(R))
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def loadmod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
M=loadmod('fit_tools',R/'experiments/contact_preference/occluded_surface_fit_v1/src/fit_surfaces.py')
# Reuse only original pure RGB-D rigid fitting functions, not the previous script's top-level work.
tree=ast.parse((R/'experiments/contact_preference/stage012_v1/stage2_reuse_localize.py').read_text())
ns={'np':np};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['fit','apply','robust']],type_ignores=[]),'reused_rigid_functions','exec'),ns)
cfg=json.loads((D/'config.json').read_text());old=json.loads((R/'experiments/contact_preference/stage012_v1/stage2_audit.json').read_text())['cross_task_localization']
def tf(P,T):return P@T[:3,:3].T+T[:3,3]
def back(dep,K,C,uv):
 v=uv[:,1];u=uv[:,0];z=dep[v,u];return tf(np.c_[(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z],C)
def project(P,K,C):
 p=tf(P,np.linalg.inv(C));uv=p[:,:2]/np.maximum(p[:,2:],1e-9);return uv*np.array([K[0,0],K[1,1]])+K[:2,2],p[:,2]
def vox(P,mm=1):
 return np.unique(np.floor(P/(mm/1000)).astype(np.int64),axis=0,return_index=True)[1]
def arr(x,d=6):return np.round(np.asarray(x),d).tolist()
def fit_surface(P,cams,mesh):
 # Same independent local surface fits and geometric normal intersection as earlier; no forced offset.
 G,N,_,_=M.sample_mesh(mesh,2026,512)
 tr=cKDTree(P); rows=[]
 for k,(g,n) in enumerate(zip(G,N)):
  dist,idx=tr.query(g)
  if dist>0.012:continue
  estimates=[]
  for rad in cfg['patch_radii_m']:
   ix=tr.query_ball_point(P[idx],rad)
   if len(ix)<12:break
   pa=M.fit_patch(P[idx],P,np.asarray(ix),rad,cams)
   if pa is None:break
   for model in pa['models']:
    hit=M.intersect(pa,g,n,model,search=cfg['normal_search_m'])
    if hit is None:break
    estimates.append((float(hit[0]),hit[1],float(model['rms'])))
   else:continue
   break
  if len(estimates)!=4:continue
  gaps=np.array([z[0] for z in estimates]);rm=max(z[2] for z in estimates);lo=float(gaps.min()-2*rm);hi=float(gaps.max()+2*rm)
  q=estimates[0][1];rows.append({'sample':k,'p':arr(q),'gap_mm':float(gaps[0]*1000),'lo_mm':lo*1000,'hi_mm':hi*1000,'zero_compatible':bool(lo<=0<=hi),'status':'unknown_surface_estimate_not_contact'})
 return rows
def main():
 robot=M.RobotDepth(448);summary={'time':now(),'cameras':cfg['cameras'],'cases':[],'training':False,'source':'actual RGB-D plus existing visual tracks, no simulation object pose/contact truth'}
 maxrep=0.;maxdep=0.
 try:
  for case in cfg['cases']:
   key=case['key'];C=R/'experiments/contact_frames/cf017/cases'/key
   motion=np.load(C/'motion.npz');ref=np.load(C/'artifacts/reference.npz');tracks=np.load(C/'artifacts/tracks.npz')
   sourcebox=old[key]['initial_uv_bbox'];uv0=ref['surface_uv'];srcmask=(uv0[:,0]>=sourcebox[0])&(uv0[:,0]<=sourcebox[2])&(uv0[:,1]>=sourcebox[1])&(uv0[:,1]<=sourcebox[3]);source=ref['surface_xyz'][srcmask];sourceuv=uv0[srcmask]
   # Retain the exact old initial source patch subset by hand neighborhood geometry where possible.
   hr=motion['hand_to_world'][old[key]['reference_frame']];qs=motion['measured_finger_q'][old[key]['reference_frame']]
   source_center=source.mean(0);neartrack=np.linalg.norm(ref['track_xyz']-source_center,axis=1)<0.04
   heldout=ref['heldout'].astype(bool)
   result={'key':key,'title':case['title'],'cameras':cfg['cameras'],'frames':[],'clip_frames':int(len(motion['hand_to_world'])),'source_frame':0,'source_patch_points':len(source),'source_patch_role':'prior real observed neighborhood; not contact label','frame_url':'/图示/双相机操作/'+key+'/{camera}_{frame}.jpg'}
   (D/'images'/key).mkdir(exist_ok=True)
   with h5py.File(R/f'experiments/contact_frames/cf017/rgbd/{key}.h5','r') as h:
    for cam in cfg['cameras']:
     for t in range(result['clip_frames']):
      Image.fromarray(h[cam+'/rgb'][t]).save(D/'images'/key/f'{cam}_{t}.jpg',quality=86)
    for t in case['frames']:
     H=motion['hand_to_world'][t].astype(float);q=motion['measured_finger_q'][t];IH=np.linalg.inv(H)
     views=[];pts=[];colors=[];pixels=[];cameraids=[];report=[]
     for ci,cam in enumerate(cfg['cameras']):
      dep=h[cam+'/depth_m'][t];K=h[cam+'/K'][t];Cw=h[cam+'/camera_to_world'][t]
      rd=robot.render(q,IH@Cw,K)
      v,u=np.where(np.isfinite(dep)&(dep>0)&(dep<2));uv=np.c_[u,v];Pw=back(dep,K,Cw,uv);Ph=tf(Pw,IH)
      roi=np.linalg.norm(Ph-np.array([0,0,.097]),axis=1)<cfg['context_radius_m']
      allowed=roi&(dep[v,u]<rd[v,u]-cfg['robot_depth_margin_m'])
      selected=np.flatnonzero(allowed);PP=Ph[selected];Pworld=Pw[selected];U=uv[selected]
      pi,z=project(Pworld,K,Cw)
      if len(U):maxrep=max(maxrep,float(abs(pi-U).max()));maxdep=max(maxdep,float(abs(z-dep[U[:,1],U[:,0]]).max()))
      pts.append(PP);pixels.append(U);colors.append(h[cam+'/rgb'][t][U[:,1],U[:,0]]);cameraids.append(np.full(len(U),ci))
      report.append({'camera':cam,'roi_pixels':int(roi.sum()),'robot_excluded':int((roi&~allowed).sum()),'retained_real_points':len(U)})
      views.append({'K':K,'C':Cw,'dep':dep,'robot':rd})
     PP=np.concatenate(pts);RGB=np.concatenate(colors);UV=np.concatenate(pixels);cams=np.concatenate(cameraids)
     ids=vox(PP);P=PP[ids];rgb=RGB[ids];camids=cams[ids]
     mesh,_=M.geo.surface(robot,[2,5]);base,_=M.geo.surface(robot,[0,1])
     # Distance to visible finger geometry, not hand-center radius or finite grasp box.
     sampled,_,_,_=M.sample_mesh(mesh,2026,12000);approx=cKDTree(sampled).query(P)[0]
     ix=np.flatnonzero(approx<0.012);near=np.zeros(len(P),bool);d=np.full(len(P),np.nan)
     if len(ix):
      dis,closest=M.geo.distance(mesh,P[ix]);d[ix]=dis
      bd,_=M.geo.distance(base,P[ix]);near[ix]=(dis<=cfg['display_distance_band_m'])&(dis<bd)
     fitrows=fit_surface(P,camids,mesh) if len(P)>=12 else []
     # Existing local optical tracks, actual depth and robot occlusion. No world/static object assumption.
     truv=tracks['tracks'][t];vis=tracks['visibility'][t]>.9;U=np.rint(truv).astype(int);valid=vis&(U[:,0]>=0)&(U[:,0]<448)&(U[:,1]>=0)&(U[:,1]<448)&neartrack
     ii=np.flatnonzero(valid);V=views[0];cur=np.zeros_like(ref['track_xyz'],dtype=float)
     if len(ii):
      uu=U[ii];zz=V['dep'][uu[:,1],uu[:,0]];rr=V['robot'][uu[:,1],uu[:,0]]
      ok=(zz>0)&np.isfinite(zz)&(zz<rr-.002);ii=ii[ok];cur[ii]=back(V['dep'],V['K'],V['C'],U[ii])
     good=np.zeros(len(valid),bool);good[ii]=True;trainids=np.flatnonzero(good&~heldout);testids=np.flatnonzero(good&heldout)
     fit=ns['robust'](ref['track_xyz'][trainids],cur[trainids],seed=2026)
     trackinfo={'available_train':len(trainids),'available_heldout':len(testids),'accepted_for_surface_propagation':False,'status':'unknown_insufficient_correspondence'}
     propagated=np.empty((0,3));propdata=[]
     if fit is not None:
      rr,tt,inlier,res=fit;errors=np.linalg.norm(ns['apply'](ref['track_xyz'][testids],rr,tt)-cur[testids],axis=1)*1000
      trainres=np.linalg.norm(ns['apply'](ref['track_xyz'][trainids],rr,tt)-cur[trainids],axis=1)*1000
      med=float(np.median(trainres[inlier]));propok=int(inlier.sum())>=8 and float(inlier.mean())>=.6 and med<=2 and len(errors)>=2 and float(errors.max())<=3
      trackinfo.update({'inliers':int(inlier.sum()),'ratio':float(inlier.mean()),'median_inlier_mm':med,'heldout_errors_mm':arr(errors,3),'accepted_for_surface_propagation':bool(propok),'status':'supported_mapping_not_contact' if propok else 'unknown_mapping_quality'})
      if propok:
       pw=ns['apply'](source,rr,tt);propagated=tf(pw,IH)
       ds,_=M.geo.distance(mesh,propagated)
       for j in np.flatnonzero(ds<=.008):
        obs=[]
        for v in views:
         xy,zz=project(pw[j:j+1],v['K'],v['C']);a=np.rint(xy[0]).astype(int)
         if np.any(a<0) or np.any(a>=448) or zz[0]<=0:obs.append('out_of_view');continue
         dd=float(v['dep'][a[1],a[0]]);rd=float(v['robot'][a[1],a[0]])
         obs.append('gripper_occluded' if rd<zz[0]-.003 and abs(dd-rd)<.003 else ('depth_compatible' if abs(dd-zz[0])<=.003 else 'unresolved_occlusion_or_depth'))
        propdata.append({'p':arr(propagated[j]),'source_pixel':sourceuv[j].tolist(),'distance_mm':float(ds[j]*1000),'views':obs,'status':'unknown_contact_even_if_mapping_supported'})
     # Protect source provenance: current frames saved unchanged with per-point source camera/pixel.
     np.savez_compressed(D/'data'/f'{key}_{t}.npz',points_hand=P,points_world=tf(P,H),rgb=rgb,camera=camids,pixel=UV[ids],near_visual_finger=near,distance_visual_finger_m=d,propagated_hand=propagated)
     # Joint RGB display projections in exact camera conventions.
     def projections(X):
      return [arr(project(tf(X,H),v['K'],v['C'])[0],2) for v in views]
     nearids=np.flatnonzero(near);fitP=np.array([z['p'] for z in fitrows]).reshape(-1,3)
     displayids=np.unique(np.r_[np.linspace(0,len(P)-1,min(5500,len(P))).astype(int),nearids]) if len(P) else np.array([],int)
     dat={'frame':t,'camera_report':report,'retained_1mm_cells':len(P),'near_visual_finger_cells':int(near.sum()),'surface_estimates':fitrows,'near_uv':projections(P[near]),'fit_uv':projections(fitP),'cloud':arr(P[displayids]),'rgb':rgb[displayids].tolist(),'near_cloud':arr(P[near]),'mesh_vertices':arr(mesh.vertices),'mesh_faces':mesh.faces.tolist(),'track':trackinfo,'propagated':propdata,'prop_uv':projections(np.array([z['p'] for z in propdata]).reshape(-1,3)),'source_frame':0,'source_patch_points':len(source),'H':arr(H),'views':[{'K':arr(v['K']),'C':arr(v['C'])} for v in views]}
     result['frames'].append(dat);save(D/'data'/f'{key}.json',result)
     print(json.dumps({'case':key,'frame':t,'cells':len(P),'near':int(near.sum()),'fits':len(fitrows),'mapping':trackinfo['status']},ensure_ascii=False),flush=True)
   summary['cases'].append({'key':key,'frames':[{'frame':x['frame'],'near':x['near_visual_finger_cells'],'fits':len(x['surface_estimates']),'track':x['track'],'cameras':x['camera_report']} for x in result['frames']]})
   save(D/'summary_partial.json',summary)
  summary.update({'time_completed':now(),'pixel_reprojection_max':maxrep,'depth_reconstruction_max_m':maxdep,'notes':['3mm display bands are not positive contact labels','current-frame dual-view fusion does not independently reconstruct occluded surface','source propagation only under frozen correspondence gates; no lift criterion']})
  save(D/'summary.json',summary);save(D/'stage_state.json',{'time':now(),'stage':'computed'})
 finally:robot.close()
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(D/'failure.json',{'time':now(),'type':type(e).__name__,'stack':[{'file':Path(v.filename).name,'line':v.lineno} for v in traceback.extract_tb(e.__traceback__)]})
  with (D/'failure.log').open('w') as z:traceback.print_exc(file=z)
  print('FAILED:'+type(e).__name__,flush=True);sys.exit(1)
