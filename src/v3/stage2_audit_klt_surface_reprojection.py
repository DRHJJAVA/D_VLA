"""Audit sparse KLT-derived rigid poses on dense initial RGB-D object surface.
No contact/robot/object simulator truth. Depth occlusion includes self-occlusion.
"""
from pathlib import Path
import json,hashlib
import cv2,h5py,numpy as np
from stage2_track_surface_klt_rgbd import cloud
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'audits/v3/surface_klt_rgbd_pilot_v1'
OUT=R/'audits/v3/surface_klt_rgbd_dense_reprojection_v1'
def project(points,K,T,dep,colors,initial_colors):
 cam=(points-T[:3,3])@T[:3,:3];z=cam[:,2]
 uv=np.stack((K[0,0]*cam[:,0]/np.maximum(z,.001)+K[0,2],K[1,1]*cam[:,1]/np.maximum(z,.001)+K[1,2]),1)
 ii=(z>.05)&(uv[:,0]>=0)&(uv[:,0]<dep.shape[1]-1)&(uv[:,1]>=0)&(uv[:,1]<dep.shape[0]-1)
 ix=np.flatnonzero(ii)
 if not len(ix):return {'inframe_unique':0},np.empty((0,2),int),np.empty(0)
 pix=np.rint(uv[ix]).astype(int)
 # Front-most transformed initial point per pixel: self z-buffer within reconstructed subset.
 linear=pix[:,1]*dep.shape[1]+pix[:,0];order=np.lexsort((z[ix],linear))
 _,first=np.unique(linear[order],return_index=True);take=order[first];ix=ix[take];pix=pix[take]
 obs=dep[pix[:,1],pix[:,0]];valid=np.isfinite(obs)&(obs>.05)
 ix=ix[valid];pix=pix[valid];delta=obs[valid]-z[ix]
 if not len(ix):return {'inframe_unique':0},pix,delta
 match=np.abs(delta)<.015
 co=np.abs(colors[pix[:,1],pix[:,0]].astype(float)-initial_colors[ix].astype(float)).mean(1)/255
 stats=dict(inframe_unique=len(ix),depth_match_15mm=float(match.mean()),nearer_surface_fraction=float((delta<-.015).mean()),
    behind_or_pose_error_fraction=float((delta>.015).mean()),median_abs_depth_mm=float(np.median(np.abs(delta))*1000),
    matched_color_l1=float(np.median(co[match])) if match.any() else None)
 return stats,pix,delta
def main():
 OUT.mkdir(parents=True,exist_ok=True);p=json.loads((D/'protocol.json').read_text())
 config=dict(source_protocol_sha256=hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest(),depth_match_m=.015,
    displacement_group_m=.01,zbuffer='nearest transformed initial point per target pixel',
    no_contact=True,no_simulator_truth=True,no_threshold_selection_from_results=True,
    caveat='depth self-consistency of partial initial surface; nearer surface may be self-occlusion or robot; moving counts not accuracy')
 cp=OUT/'protocol.json'
 if cp.exists():assert json.loads(cp.read_text())==config
 else:cp.write_text(json.dumps(config,indent=2))
 allrows=[]
 for row in p['episodes']:
  dst=OUT/(row['id']+'.json')
  if dst.exists():allrows.append(json.loads(dst.read_text()));continue
  audit=json.loads((D/(row['id']+'.json')).read_text())
  with np.load(D/(row['id']+'.npz')) as z:poses=z['pose']
  with np.load(row['component']) as z:pix=z['pixel'].astype(int)
  frames=[]
  with h5py.File(row['rgbd'],'r') as f:
   a=f['agentview'];w=f['robot0_eye_in_hand'];rgb0=a['rgb'][0]
   pts,valid=cloud(pix.astype(float),a['depth_m'][0],a['K'][0],a['camera_to_world'][0]);pts=pts[valid];pix=pix[valid]
   col=rgb0[pix[:,1],pix[:,0]];center=pts.mean(0)
   samples=set([0,row['frames']-1]+[round((row['frames']-1)*q) for q in [.25,.4,.55,.75]])
   samples.update({'tomato_sauce_demo_3':[45,75,110,150],'porcelain_mug_demo_4':[36,57,70,90],'white_yellow_mug_demo_2':[45,61,75,100]}.get(row['id'],[]))
   for v in audit['rows']:
    t=v['frame'];entry=dict(frame=t,pose_candidate=v['pose_candidate'])
    if not v['pose_candidate']:frames.append(entry);continue
    T=poses[t];moved=pts@T[:3,:3].T+T[:3,3]
    displacement=float(np.linalg.norm(moved.mean(0)-center))
    entry.update(centroid_displacement_m=displacement,moving_over_1cm=displacement>.01)
    ims=[]
    for cam in ['agentview','robot0_eye_in_hand']:
     c=f[cam];rgb=c['rgb'][t];stats,uv,delta=project(moved,c['K'][t],c['camera_to_world'][t],c['depth_m'][t],rgb,col)
     entry[cam]=stats
     if t in samples:
      im=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
      for (u,y),err in zip(uv,delta):
       color=(60,220,230) if abs(err)<.015 else ((180,100,220) if err<-.015 else (255,180,70))
       im[y,u]=color
      ims.append(im)
    if t in samples:
     im=np.concatenate(ims,1);cv2.rectangle(im,(0,0),(im.shape[1],36),(0,0,0),-1)
     text=row['id']+' t'+str(t)+' displacement '+format(displacement*100,'.1f')+'cm depth diagnostic'
     cv2.putText(im,text,(5,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
     ip=OUT/(row['id']+'_t'+str(t)+'.jpg');assert not ip.exists();cv2.imwrite(str(ip),im,[cv2.IMWRITE_JPEG_QUALITY,89])
    frames.append(entry)
  valid=[x for x in frames if x['pose_candidate']];moving=[x for x in valid if x['moving_over_1cm']]
  def agg(xx):
   return {cam:{k:float(np.mean([x[cam][k] for x in xx if k in x[cam]])) if any(k in x[cam] for x in xx) else None for k in ['depth_match_15mm','nearer_surface_fraction','behind_or_pose_error_fraction']} for cam in ['agentview','robot0_eye_in_hand']}
  v=dict(id=row['id'],split=row['split'],frames=row['frames'],pose_frames=len(valid),moving_pose_frames=len(moving),
    moving_first_last=[moving[0]['frame'],moving[-1]['frame']] if moving else None,
    all_pose_mean=agg(valid),moving_pose_mean=agg(moving),rows=frames)
  dst.write_text(json.dumps(v,indent=2));allrows.append(v);print(json.dumps({k:x for k,x in v.items() if k!='rows'}),flush=True)
 sp=OUT/'summary.json';assert not sp.exists();sp.write_text(json.dumps(dict(episodes=allrows,config=config),indent=2))
 print('complete',len(allrows),flush=True)
if __name__=='__main__':main()
