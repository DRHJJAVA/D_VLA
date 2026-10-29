"""Visible object-side RGB-D samples inside the finite inter-finger grasp volume.
No contact-truth labels, no hidden surface completion, no new training."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,hashlib,numpy as np,h5py,cv2
from stage2_probe_motion_fingerpads import cloud
from stage2_robot_only_depth import RobotDepth
R=Path('/mnt/nvme1/drh/D_VLA')
D=Path(__file__).resolve().parents[1]/'evidence/motion'
OLD=Path(__file__).resolve().parents[1]/'evidence/reference_cloud'
OUT=Path(__file__).resolve().parents[1]/'evidence/region'
PAGE=Path(__file__).resolve().parents[1]/'evidence/region_internal.html'
IMG=Path(__file__).resolve().parents[1]/'evidence/region_images'
def writeply(path,p,rgb):
 with path.open('x') as f:
  f.write('ply\nformat ascii 1.0\ncomment Visible RGBD samples inside inter-finger volume, NOT contact truth\nelement vertex '+str(len(p))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
  for xyz,c in zip(p,rgb):f.write(' '.join([*(f'{x:.8f}' for x in xyz),*(str(int(x)) for x in c)])+'\n')
def main():
 OUT.mkdir(exist_ok=False,parents=True);IMG.mkdir(exist_ok=False,parents=True);assert not PAGE.exists()
 rows={x['id']:x for x in json.loads((Path(__file__).resolve().parents[1]/'evidence/base/protocol.json').read_text())['episodes']}
 items=json.loads((D/'review_frames.json').read_text());cases=[];summary=[]
 for it in items:
  ident,t=it['id'],it['frame'];name=ident+'_attempt'+str(it['attempt']);held=it['accepted_by_motion_visual_gate']
  with np.load(D/(ident+'.npz')) as z: H=z['hand_to_world'][t];q=z['measured_finger_q'][t]
  lo=np.array([-.008,q[1]+.0005,.0854]);hi=np.array([.008,q[0]-.0005,.1014])
  parts=[];rgbparts=[];camparts=[];uvparts=[];panels=[]
  with h5py.File(rows[ident]['rgbd'],'r') as f:
   rob=RobotDepth(f['agentview/rgb'].shape[1])
   for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
    g=f[cam];depth=g['depth_m'][t];K=g['K'][t];T=g['camera_to_world'][t];rgb=g['rgb'][t]
    p=cloud(depth,K,T);local=(p-H[:3,3])@H[:3,:3];rd=rob.render(q,np.linalg.inv(H)@T,K)
    mask=np.isfinite(p).all(1)&(depth.reshape(-1)>0)&(depth<rd-.002).reshape(-1)&np.all(local>=lo,axis=1)&np.all(local<=hi,axis=1)
    if not held:mask[:]=False
    v,u=np.indices(depth.shape);uv=np.stack([u,v],-1).reshape(-1,2)
    parts.append(p[mask]);rgbparts.append(rgb.reshape(-1,3)[mask]);camparts.append(np.full(mask.sum(),ci,np.uint8));uvparts.append(uv[mask])
    overlay=rgb.copy();overlay.reshape(-1,3)[mask]=[30,245,70]
    # Focus both images on known finger volume, with surrounding object context.
    center=(lo+hi)/2;cw=center@H[:3,:3].T+H[:3,3];cc=(cw-T[:3,3])@T[:3,:3]
    if cc[2]>0:
     cx,cy=int(K[0,0]*cc[0]/cc[2]+K[0,2]),int(K[1,1]*cc[1]/cc[2]+K[1,2])
    else:cx,cy=depth.shape[1]//2,depth.shape[0]//2
    half=90;x0=max(0,min(depth.shape[1]-2*half,cx-half));y0=max(0,min(depth.shape[0]-2*half,cy-half))
    images=[cv2.resize(a[y0:y0+2*half,x0:x0+2*half],(420,420),interpolation=cv2.INTER_NEAREST) for a in [rgb,overlay]]
    panel=np.concatenate(images,axis=1);cv2.rectangle(panel,(0,0),(840,28),(15,25,35),-1)
    cv2.putText(panel,f'{cam} frame {t} | original / inter-finger visible points {int(mask.sum())}',(8,20),0,.48,(255,255,255),1)
    panels.append(panel)
   rob.close()
  p=np.concatenate(parts);rgb=np.concatenate(rgbparts);cams=np.concatenate(camparts);uv=np.concatenate(uvparts)
  local=(p-H[:3,3])@H[:3,:3];assert np.all(local>=lo-1e-8) and np.all(local<=hi+1e-8)
  np.savez_compressed(OUT/(name+'.npz'),xyz_world_m=p,xyz_hand_m=local,rgb=rgb,source_camera=cams,source_pixel=uv,frame=t,hand_to_world=H,volume_min_hand_m=lo,volume_max_hand_m=hi,held_candidate=held,region_not_contact_truth=True)
  writeply(OUT/(name+'.ply'),p,rgb)
  cv2.imwrite(str(IMG/(name+'.jpg')),cv2.cvtColor(np.concatenate(panels),cv2.COLOR_RGB2BGR))
  with np.load(OLD/(name+'.npz')) as z: context=z['xyz_world_m'][z['contact_estimate']==0];old=z['xyz_world_m'][z['contact_estimate']>0]
  loccontext=(context-H[:3,3])@H[:3,:3];center=loccontext.mean(0)
  corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
  unique=len(np.unique(np.rint(p/.001).astype(np.int64),axis=0)) if len(p) else 0
  s={'id':ident,'attempt':it['attempt'],'frame':t,'held_candidate':held,'old_contact_estimate_points':len(old),'visible_region_samples':len(p),'camera_counts':[int((cams==j).sum()) for j in range(2)],'occupied_1mm_voxels':unique,'volume_min_hand_m':lo.tolist(),'volume_max_hand_m':hi.tolist()}
  summary.append(s)
  cases.append(dict(name=name,frame=t,held=held,context=((loccontext-center)*1000).round(3).tolist(),region=((local-center)*1000).round(3).tolist(),old=(((old-H[:3,3])@H[:3,:3]-center)*1000).round(3).tolist(),box=((corners-center)*1000).round(3).tolist(),world=p.round(8).tolist(),cams=cams.tolist(),uv=uv.tolist(),counts=s['camera_counts'],voxels=unique,image='图示/夹爪之间物体点云/'+name+'.jpg'))
  print(name,'old',len(old),'region',len(p),'1mm_voxels',unique,flush=True)
 PAGE.write_text('Internal finite inter-finger region; not contact truth.',encoding='utf-8')
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 (OUT/'protocol.json').write_text(json.dumps({'input':'same six reviewed frames; complete dual-camera depth at native resolution; aligned measured q; robot-only render','selection':'held gate AND nonrobot depth AND finite prism between inner finger pad planes with x +/-8mm and z85.4..101.4mm in hand coordinates','no_synthetic_points':True,'occluded_surface':'unknown','not_contact_truth':True,'no_training':True},indent=2))
 files=list(OUT.iterdir())+list(IMG.iterdir())+[PAGE,Path(__file__)]
 (OUT/'locked.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
 print('complete',flush=True)
if __name__=='__main__':main()
