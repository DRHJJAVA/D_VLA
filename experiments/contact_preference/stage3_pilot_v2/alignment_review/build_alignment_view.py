from pathlib import Path
import json,numpy as np,h5py,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
B=Path('/mnt/nvme1/drh/D_VLA');R=B/'experiments/contact_preference/stage3_pilot_v2';O=R/'alignment_review'
(O/'webdata').mkdir(exist_ok=True)
key='white_yellow_mug_demo_4_attempt1_query40'
stats=json.loads((O/'depth_alignment.json').read_text())['rows']
def tf(x,T):return (np.c_[x,np.ones(len(x))]@T.T)[:,:3]
def arr(x):return np.round(x,7).tolist()
with h5py.File(B/'data/v3/trainval_rgbd_448/white_yellow_mug/demo_4.h5','r') as h:
 for t in [66,68,71]:
  z=np.load(R/'points_evidence'/(key+'_frame%d.npz'%t))
  rob=tf(z['gripper_vertices'],z['hand_to_world'])
  world=z['mapped_world_xyz'];center=np.median(world,axis=0)
  data={'frame':t,'query':40,'rgb':z['query_rgb'].tolist(),'category':z['display_class'].tolist(),'source_pixel':z['query_pixel'].tolist(),'faces':z['gripper_faces'].tolist(),'cameras':{},'stats':next(r['cameras'] for r in stats if r['case']==2 and r['frame']==t)}
  for short,cam in [('main','agentview'),('wrist','robot0_eye_in_hand')]:
   K=h[cam+'/K'][t].astype(float);T=h[cam+'/camera_to_world'][t].astype(float)
   pc=tf(world,np.linalg.inv(T));rc=tf(rob,np.linalg.inv(T))
   dep=h[cam+'/depth_m'][t];rgb=h[cam+'/rgb'][t]
   yy,xx=np.mgrid[0:448:2,0:448:2];zz=dep[yy,xx];uv=np.stack([xx,yy,np.ones_like(xx)],-1)
   scene=(uv@np.linalg.inv(K).T)*zz[...,None];sw=tf(scene.reshape(-1,3),T)
   keep=(np.linalg.norm(sw-center,axis=1)<.17)&(zz.ravel()>0)
   scene=scene.reshape(-1,3)[keep];col=rgb[yy,xx].reshape(-1,3)[keep]
   a=np.load(O/(cam+'_f%d_projection.npz'%t))
   mask=a['in_image'];res=a['residual']
   depthclass=np.where(~mask,3,np.where(res<-.003,2,np.where(res>.003,1,0)))
   data['cameras'][short]={'K':arr(K),'points':arr(pc),'robot':arr(rc),'depth_class':depthclass.tolist(),'residual_mm':[float(v*1000) if np.isfinite(v) else None for v in res],'scene':arr(scene),'scene_rgb':col.tolist(),'image':'/图示/阶段3支持/images/c2_f%d_%s.jpg'%(t,cam)}
  (O/'webdata'/('f%d.json'%t)).write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
print('generated3frames')
