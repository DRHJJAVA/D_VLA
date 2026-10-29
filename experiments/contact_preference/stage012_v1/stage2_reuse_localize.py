from pathlib import Path
import json, hashlib
import numpy as np
import h5py
from PIL import Image,ImageDraw,ImageFont
from datetime import datetime,timezone,timedelta
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/stage012_v1';G=R/'docs/图示/接触偏好_阶段012_正式';G.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def tile(img,title,uv=None,color=(30,240,70),radius=3):
 im=Image.fromarray(np.asarray(img,dtype=np.uint8)).convert('RGB').resize((448,448))
 d=ImageDraw.Draw(im)
 if uv is not None:
  for x,y in uv:
   x=float(x);y=float(y)
   if 0<=x<448 and 0<=y<448:d.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=2)
 d.rectangle((0,0,448,25),fill=(10,14,25));d.text((7,5),title,fill=(255,255,255))
 return im
def save_pair(a,b,name):
 im=Image.new('RGB',(896,448),(15,18,25));im.paste(a,(0,0));im.paste(b,(448,0));p=G/name;im.save(p,quality=92);return str(p.relative_to(R)),sha(p)
def backproject(D,K,C,uv):
 uv=np.asarray(uv);ij=np.rint(uv).astype(int);ok=(ij[:,0]>=0)&(ij[:,0]<D.shape[1])&(ij[:,1]>=0)&(ij[:,1]<D.shape[0])
 z=np.full(len(uv),np.nan);z[ok]=D[ij[ok,1],ij[ok,0]]
 good=ok & np.isfinite(z)&(z>.03)&(z<2)
 p=np.full((len(uv),3),np.nan)
 u,v=uv[good,0],uv[good,1];zz=z[good]
 cam=np.stack(((u-K[0,2])*zz/K[0,0],(v-K[1,2])*zz/K[1,1],zz),axis=1)
 p[good]=cam@C[:3,:3].T+C[:3,3]
 return p,good
def project(x,K,C):
 cam=(x-C[:3,3])@np.linalg.inv(C[:3,:3]).T
 return np.stack((K[0,0]*cam[:,0]/cam[:,2]+K[0,2],K[1,1]*cam[:,1]/cam[:,2]+K[1,2]),axis=1)
def fit(A,B):
 aa=A.mean(axis=0);bb=B.mean(axis=0);U,S,Vt=np.linalg.svd((A-aa).T@(B-bb));D=np.eye(3);D[-1,-1]=np.linalg.det(U@Vt);R=U@D@Vt;t=bb-aa@R
 return R,t
def apply(A,R,t):return A@R+t
def robust(A,B,seed=2026):
 n=len(A)
 if n<6:return None
 gen=np.random.default_rng(seed);best=None
 for _ in range(500):
  ids=gen.choice(n,3,replace=False)
  if np.linalg.norm(np.cross(A[ids[1]]-A[ids[0]],A[ids[2]]-A[ids[0]]))<1e-6:continue
  rr,tt=fit(A[ids],B[ids]);e=np.linalg.norm(apply(A,rr,tt)-B,axis=1);m=e<.003
  if m.sum()<4:continue
  score=(int(m.sum()),-float(np.median(e[m])))
  if best is None or score>best[0]:best=(score,m)
 if best is None:return None
 m=best[1];rr,tt=fit(A[m],B[m]);err=np.linalg.norm(apply(A,rr,tt)-B,axis=1);m=err<.003
 if m.sum()<4:return None
 rr,tt=fit(A[m],B[m]);return rr,tt,m,np.linalg.norm(apply(A,rr,tt)-B,axis=1)
# Reuse validated grasp sources; no new tracking or label inference.
grasp=[('porcelain_mug',9,36,55,'Porcelain mug, grasp example A'),('tomato_sauce',7,48,75,'Tomato can, grasp example B'),('white_yellow_mug',4,40,66,'White/yellow mug, grasp example C')]
records=[]
old24=json.loads((R/'experiments/region_pose/iter_024/completed_locked.json').read_text())
old25=json.loads((R/'experiments/region_pose/iter_025/completed_locked.json').read_text())
for asset,demo,q,desired,title in grasp:
 key=f'{asset}_demo_{demo}_attempt1_query{q}'
 p24=R/f'experiments/region_pose/iter_024/train/{key}.npz'
 p25=R/f'experiments/region_pose/iter_025/targets/train/{key}.npz'
 if not p24.exists() or not p25.exists():raise RuntimeError('missing_grasp_source_'+key)
 rel24=str(p24.relative_to(R));rel25=str(p25.relative_to(R))
 if old24.get(rel24)!=sha(p24) or old25.get(rel25)!=sha(p25):raise RuntimeError('old_lock_mismatch_'+key)
 a=np.load(p24);b=np.load(p25)
 if not np.array_equal(a['query_pixel'],b['query_pixel']) or not np.array_equal(a['query_xyz'],b['query_xyz']):raise RuntimeError('query_mismatch_'+key)
 pos=(b['known']>0)&(b['target']>0);frames=a['selected_frames'].astype(int)
 idx=int(np.argmin(np.abs(frames-desired)));hf=int(frames[idx])
 h=R/f'data/v3/trainval_rgbd_448/{asset}/demo_{demo}.h5'
 with h5py.File(h,'r') as src:
  pre=np.asarray(src['agentview/rgb'][q])
  held=np.asarray(src['agentview/rgb'][hf])
  hlen=int(src['frame_indices'].shape[0])
 source_mask=a['frame']==hf
 source_uv=a['pixel'][source_mask & (a['camera']==0)]
 pict,phash=save_pair(tile(pre,f'{key}: pre-contact frame {q}',b['query_pixel'][pos],(40,240,80),3),tile(held,f'held frame {hf}: observed finite-finger sources',source_uv,(255,190,35),4),f'grasp_{asset}_{demo}.jpg')
 records.append({'key':key,'title':title,'source_024':rel24,'source_025':rel25,'source_024_sha256':sha(p24),'source_025_sha256':sha(p25),'query_frame':q,'held_frame':hf,'positive_query_surface_points':int(pos.sum()),'real_source_observations':int(source_mask.sum()),'held_main_camera_source_observations':int(len(source_uv)),'selected_frames':frames.tolist(),'rendered_frames':hlen,'image':pict,'image_sha256':phash,'semantics':'green is automatic stable surface candidate on pre-contact RGB-D; amber are actually observed finite-finger volume source points at held frame; neither is physical contact truth'})
cf=R/'experiments/contact_frames/cf017'
local={}
for name,ref_frame,radius in [('drawer_demo_0',44,.03),('stove_demo_0',86,.02),('two_objects_demo_0',300,.02)]:
 base=cf/'cases'/name
 ref=np.load(base/'artifacts/reference.npz');mot=np.load(base/'motion.npz');tr=np.load(base/'artifacts/tracks.npz')
 X=np.asarray(ref['surface_xyz']);uv=np.asarray(ref['surface_uv']);H=mot['hand_to_world'][ref_frame];q=mot['measured_finger_q'][ref_frame]
 centers=np.stack([H[:3,:3]@np.array([0,q[0],.093])+H[:3,3],H[:3,:3]@np.array([0,q[1],.093])+H[:3,3],H[:3,:3]@np.array([0,0,.097])+H[:3,3]])
 dist=np.min(np.linalg.norm(X[:,None,:]-centers[None,:,:],axis=2),axis=1);sel=dist<radius;px=uv[sel]
 box=[int(px[:,0].min()),int(px[:,1].min()),int(px[:,0].max()),int(px[:,1].max())] if len(px) else None
 # Patches are diagnostic hand-neighborhoods, not semantic masks or contact labels.
 with h5py.File(cf/'rgbd'/(name+'.h5'),'r') as h:
  img0=np.asarray(h['agentview/rgb'][0]);imgr=np.asarray(h['agentview/rgb'][ref_frame]);K0=np.asarray(h['agentview/K'][0]);C0=np.asarray(h['agentview/camera_to_world'][0])
  source_xyz=X[sel]
  rec={'reference_frame':ref_frame,'source_frame':0,'hand_radius_m_diagnostic':radius,'initial_true_depth_points':int(sel.sum()),'initial_uv_bbox':box,'source_1mm_cells':int(np.unique(np.floor(source_xyz/.001).astype(np.int64),axis=0).shape[0]),'source_points_within_hand_neighborhood_not_proven_contact':True,'initial_source_provenance_sha256':sha(base/'artifacts/reference.npz')}
  if name=='drawer_demo_0':
   tuv=np.asarray(ref['track_uv']);held=np.asarray(ref['heldout'],dtype=bool)
   roi=(tuv[:,0]>=box[0]-10)&(tuv[:,0]<=box[2]+10)&(tuv[:,1]>=box[1]-10)&(tuv[:,1]<=box[3]+10)
   rec['initial_tracks_in_local_roi']=int(roi.sum());rec['heldout_tracks_in_local_roi']=int(np.sum(roi&held))
   rec['local_fit']={}
   for t in [60,70,88,100,120]:
    D=np.asarray(h['agentview/depth_m'][t]);K=np.asarray(h['agentview/K'][t]);C=np.asarray(h['agentview/camera_to_world'][t]);rgb=np.asarray(h['agentview/rgb'][t])
    tracked=tr['tracks'][t];vis=tr['visibility'][t]>.9;Y,valid=backproject(D,K,C,tracked)
    X0=np.asarray(ref['track_xyz'])
    dis=np.linalg.norm(Y-X0,axis=1)
    train=np.where(roi & ~held & vis & valid & (dis>.005))[0]
    rr=robust(X0[train],Y[train],seed=2026+t) if len(train)>=6 else None
    item={'visible_dynamic_training_tracks':int(len(train))}
    if rr is not None:
     rot,trans,inliers,e=rr
     L=np.where(roi & held & vis & valid)[0]
     if len(L):
      near=np.min(np.linalg.norm(X0[L,None,:]-X0[train[inliers]][None,:,:],axis=2),axis=1)<.02
      L=L[near]
     held_err=np.linalg.norm(apply(X0[L],rot,trans)-Y[L],axis=1) if len(L) else np.array([])
     angles=np.arccos(np.clip((np.trace(rot)-1)/2,-1,1))*180/np.pi
     cdelta=float(np.linalg.norm(apply(source_xyz.mean(axis=0,keepdims=True),rot,trans)[0]-source_xyz.mean(axis=0)))
     item.update({'inliers':int(inliers.sum()),'train_residual_median_mm':float(np.median(e[inliers])*1000),'heldout_near_tracks':int(len(L)),'heldout_near_errors_mm':(held_err*1000).tolist(),'heldout_near_median_mm':float(np.median(held_err)*1000) if len(L) else None,'rotation_deg':float(angles),'source_center_displacement_mm':cdelta*1000})
     if t in [70,88]:
      mapped=apply(source_xyz,rot,trans)
      pix=project(mapped,K,C)
      picture,pichash=save_pair(tile(img0,'Drawer: real source depth at frame 0',px,(255,190,35),2),tile(rgb,f'Drawer frame {t}: locally tracked source, diagnostic',pix,(40,240,80),2),f'drawer_source_to_{t}.jpg')
      item.update({'image':picture,'image_sha256':pichash,'mapped_source_pixel_visible_fraction':float(np.mean((pix[:,0]>=0)&(pix[:,0]<448)&(pix[:,1]>=0)&(pix[:,1]<448)))})
    rec['local_fit'][str(t)]=item
  else:
   picture,pichash=save_pair(tile(img0,f'{name} frame 0: real depth hand-neighborhood',px,(255,190,35),3),tile(imgr,f'frame {ref_frame}: gripper context; source identity unknown'),f'{name}_source_context.jpg')
   rec.update({'image':picture,'image_sha256':pichash})
  if name=='stove_demo_0':
   tuv=np.asarray(ref['track_uv']);held=np.asarray(ref['heldout'],dtype=bool)
   roi=(tuv[:,0]>=box[0])&(tuv[:,0]<=box[2])&(tuv[:,1]>=box[1])&(tuv[:,1]<=box[3])
   rec['initial_tracks_exact_knob_patch']=int(roi.sum());rec['heldout_tracks_exact_knob_patch']=int(np.sum(roi&held))
   D=np.asarray(h['agentview/depth_m'][ref_frame]);K=np.asarray(h['agentview/K'][ref_frame]);C=np.asarray(h['agentview/camera_to_world'][ref_frame])
   Y,valid=backproject(D,K,C,tr['tracks'][ref_frame]);vis=tr['visibility'][ref_frame]>.9
   rec['visible_valid_tracked_knob_points_at_reference']=int(np.sum(roi&valid&vis))
  if name=='two_objects_demo_0':
   rec['warning']='hand-neighborhood at frame 300 aligns with tomato can, but broad initial CF017 component mixes multiple objects; no automatic two-object surface identity acceptance'
  local[name]=rec
stage2={'created_beijing':now(),'grasp_reuse':records,'cross_task_localization':local,'no_model_training_or_contact_label_update':True,'old_cf017_global_transform_reused_as_object_motion':False,'geometry_quality_is_not_contact_success':True}
(O/'stage2_audit.json').write_text(json.dumps(stage2,ensure_ascii=False,indent=2))
print(json.dumps({'grasp_cases':[(v['key'],v['positive_query_surface_points'],v['held_main_camera_source_observations']) for v in records],'drawer_fit':local['drawer_demo_0']['local_fit'],'stove':{k:local['stove_demo_0'][k] for k in ['initial_true_depth_points','initial_tracks_exact_knob_patch','heldout_tracks_exact_knob_patch','visible_valid_tracked_knob_points_at_reference']},'images':len(list(G.iterdir()))},ensure_ascii=False))
