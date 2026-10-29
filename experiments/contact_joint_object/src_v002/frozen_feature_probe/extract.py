import os,sys,pathlib,json,hashlib,time,traceback,ast
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import numpy as np,cv2,h5py,torch
from datetime import datetime
from zoneinfo import ZoneInfo
B=pathlib.Path('/mnt/nvme1/drh/D_VLA');E=B/'experiments/contact_joint_object'
OLD=E/'runs/v002/preflight/P1_interaction_extended_v1';R=E/'runs/v002/preflight/P1_frozen_feature_probe';A=B/'docs/lcjp002_feature_probe_assets'
R.mkdir(exist_ok=True);A.mkdir(exist_ok=True)
sys.path[:0]=[str(B/'src/v3'),str(E/'src')]
from stage2_extract_dino import model,preproc
from geometry import finger_geometry
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False));q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(x):
 x=np.asarray(x,dtype=np.float32);return x/max(np.linalg.norm(x),1e-12)
def delta(a,b):return None if a is None or b is None else float(np.clip(1-np.dot(norm(a),norm(b)),0,2))
def pool(f,m):
 w=cv2.resize(m.astype(np.float32),(32,32),interpolation=cv2.INTER_AREA)
 if w.sum()<1e-8:return None
 return norm((f*w[:,:,None]).sum((0,1))/w.sum())
def crop(rgb,m):
 v,u=np.where(m)
 if not len(u):return None
 x0,x1,y0,y1=u.min(),u.max()+1,v.min(),v.max()+1
 side=max(x1-x0,y1-y0);pad=max(2,int(side*.1));x0=max(0,x0-pad);y0=max(0,y0-pad);x1=min(256,x1+pad);y1=min(256,y1+pad)
 z=np.full_like(rgb,[124,116,104]);z[m]=rgb[m]
 z=z[y0:y1,x0:x1];h,w=z.shape[:2];out=np.full((max(h,w),max(h,w),3),[124,116,104],np.uint8);out[:h,:w]=z
 return out
def contours(m):
 cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 return [cv2.approxPolyDP(c,1.2,True)[:,0].tolist() for c in cs if len(c)>=3]
def back(h,cam,t,m):
 v,u=np.where(m);idx=np.linspace(0,len(u)-1,min(512,len(u))).astype(int);u=u[idx];v=v[idx]
 z=h[cam+'/depth_m'][t][v,u];K=h[cam+'/K'][t];C=h[cam+'/camera_to_world'][t]
 x=np.stack([(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z],1)
 return x@C[:3,:3].T+C[:3,3]
# Only the current-only geometry adapters are inherited; no association rules run.
tree=ast.parse((E/'src_v002/interaction_extended_v1/associate_render.py').read_text())
for node in tree.body:
 if isinstance(node,ast.ClassDef) and node.name in ['PrefixDataset','CurrentGeometryView']:
  exec(compile(ast.Module(body=[node],type_ignores=[]),'<current_geometry_adapter>','exec'),globals())
def fingers(h,t,cam,nodes):
 K=h[cam+'/K'][t];C=h[cam+'/camera_to_world'][t];dep=h[cam+'/depth_m'][t];rob=h[cam+'/robot_mask'][t].astype(bool)
 p=(nodes-C[:3,3])@C[:3,:3];z=p[:,2];uv=p[:,:2]/np.maximum(z[:,None],1e-9);uv=uv*np.array([K[0,0],K[1,1]])+np.array([K[0,2],K[1,2]])
 weights=np.zeros((256,256),np.float32)
 for (u,v),zz in zip(uv,z):
  if zz<=0 or not np.isfinite(u+v) or not (0<=u<256 and 0<=v<256):continue
  ui,vi=int(round(u)),int(round(v));ui=min(255,ui);vi=min(255,vi)
  # Visibility weighting only; never used to classify contact.
  q=np.exp(-.5*((float(dep[vi,ui])-zz)/.01)**2)
  if q<1e-8:continue
  y0,y1=max(0,vi-2),min(256,vi+3);x0,x1=max(0,ui-2),min(256,ui+3)
  weights[y0:y1,x0:x1]=np.maximum(weights[y0:y1,x0:x1],q)
 return weights*rob
try:
 t0=time.time();rgbpath=OLD/'rgbd/two_objects.h5';segpath=OLD/'segmentation/two_objects.h5'
 oldjson=B/'docs/lcjp002_interaction_extended_v1_assets/results.json'
 # Whitelist identity bookkeeping only; never copy old decisions, scores or evidence.
 source=json.loads(oldjson.read_text());ep=next(e for e in source['episodes'] if e['id']=='two_objects')
 mapping=[{int(o['id']):dict(o['local_ids']) for o in f['objects'] if 'local_ids' in o} for f in ep['frames']]
 del source,ep
 protocol={'time':now(),'episode':'two_objects/demo_0','frames':388,'encoder':'official frozen DINOv2 ViT-S/14','weights_sha256':sha(pathlib.Path('/home/drh/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')),'input_sha256':sha(rgbpath),'segmentation_sha256':sha(segpath),'identity_fields_only':['id','local_ids'],'training_updates':0,'contact_labels':False,'contact_classifier':False,'feature_dimensions':384,'preprocess':'existing preproc; ImageNet normalization; 448x448; patch grid32x32','all_instances_encoded':True,'crop_controls':[12,9,3],'metrics':['object feature change','finger feature change','elementwise product relation feature change','object-finger visual cosine','observed median point position in gripper frame'],'missing_policy':'null; no interpolation or contact zero','future_input':False,'finger_region':'current robot finger surface projection; depth visibility weights; no palm or arm positive region','display':'selected object outline only; fixed metric axes; no locking threshold'}
 save(R/'protocol.json',protocol)
 md=next((B/'docs').glob('LCJP002_*网络训练与前置验证方案.md'))
 with md.open('a') as f:f.write('\n\n### '+now()+' 冻结特征单视频检查\n用户要求先验证现成特征是否反映关系变化。只用双物体视频388帧，DINOv2-S/14冻结，不建优化器、不训练、不生成接触标签、不运行旧关联门槛。所有当前分割实例均提取特征；#12、#9及候选#3额外做去背景裁剪对照。逐视角展示视觉特征变化、物体×夹指特征乘积的变化、视觉相似度及真实RGB-D相对位置；缺观测留空。旧结果仅继承id/local_ids身份映射，不继承关联分数或状态。输出目录P1_frozen_feature_probe，页面LCJP002_单视频冻结特征检查.html。\n')
 net=model().eval()
 for p in net.parameters():p.requires_grad_(False)
 def modelsha():
  hh=hashlib.sha256()
  for k,v in net.state_dict().items():hh.update(k.encode());hh.update(v.detach().cpu().contiguous().numpy().tobytes())
  return hh.hexdigest()
 before=modelsha();torch.set_num_threads(4)
 def encode(images):
  xs=[preproc(np.expand_dims(im,0)) for im in images]
  xs=[x.unsqueeze(0) if x.ndim==3 else x for x in xs]
  with torch.inference_mode():
   z=net.forward_features(torch.cat(xs,0).cuda())
   if isinstance(z,dict):z=z['x_norm_patchtokens']
   if z.shape[1]==1025:z=z[:,1:]
   assert z.shape[1:]==(1024,384)
   return z.float().reshape(-1,32,32,384).cpu().numpy()
 records=[];vectors=[];cropvectors=[];fingervec=np.full((388,2,384),np.nan,np.float32);frames=[];prev={};allids=set();selfcheck=None
 with h5py.File(rgbpath,'r') as h,h5py.File(segpath,'r') as sg:
  names=json.loads(h.attrs['robot_metadata'])['geom_names'];gi=names.index('gripper0_hand_collision')
  for t in range(388):
   rgb=[h[c+'/rgb'][t] for c in ['agentview','robot0_eye_in_hand']];maps=encode(rgb)
   if t==0:selfcheck=float(np.abs(maps-encode(rgb)).max())
   geo=finger_geometry(CurrentGeometryView(h,t),t,np.zeros((1,3),np.float32));nodes=geo['finger_nodes'];P=h['robot/geom_position'][t,gi];G=h['robot/geom_rotation'][t,gi]
   frame={'t':t,'views':{},'objects':{}}
   for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
    fw=fingers(h,t,cam,nodes);fv=pool(maps[ci],fw);fingervec[t,ci]=np.nan if fv is None else fv
    fgmask=fw>0
    view={'rgb':'lcjp002_interaction_extended_v1_assets/two_objects/'+f'{t:04d}_{cam}_rgb.jpg','finger_contours':contours(fgmask),'finger_pixels':int(fgmask.sum()),'finger_change':delta(fv,prev.get((cam,'finger')))}
    frame['views'][cam]=view
    jobs=[];items=[]
    if fgmask.any():jobs.append(crop(rgb[ci],fgmask));items.append(('finger',None))
    for gid,local in mapping[t].items():
     if cam not in local:continue
     lid=local[cam];path=f'frames/{t}/{ci}/objects/{lid}'
     if path not in sg:continue
     m=np.unpackbits(sg[path+'/mask'][:])[:65536].reshape(256,256).astype(bool)
     if not m.any():continue
     of=pool(maps[ci],m);rel=None if of is None or fv is None else norm(of*fv)
     xyz=back(h,cam,t,m);pos=(np.median(xyz,0)-P)@G
     row={'area':int(m.sum()),'local_id':lid,'contours':contours(m),'object_change':delta(of,prev.get((cam,gid,'obj'))),'relation_change':delta(rel,prev.get((cam,gid,'rel'))),'visual_cosine':None if of is None or fv is None else float(np.dot(of,fv)),'relative_xyz_mm':(pos*1000).tolist(),'crop_change':None,'crop_cosine':None}
     frame['objects'].setdefault(str(gid),{})[cam]=row;allids.add(gid)
     records.append([t,ci,gid]);vectors.append(of);cropvectors.append(np.full(384,np.nan,np.float32))
     if gid in [12,9,3]:jobs.append(crop(rgb[ci],m));items.append((gid,len(records)-1))
     prev[(cam,gid,'obj')]=of;prev[(cam,gid,'rel')]=rel
    if jobs:
     zz=encode(jobs);cf=None
     for z,(gid,idx) in zip(zz,items):
      v=norm(z.mean((0,1)))
      if gid=='finger':cf=v;view['finger_crop_change']=delta(v,prev.get((cam,'finger_crop')));prev[(cam,'finger_crop')]=v
      else:
       cropvectors[idx]=v;row=frame['objects'][str(gid)][cam];row['crop_change']=delta(v,prev.get((cam,gid,'crop')));row['crop_cosine']=None if cf is None else float(np.dot(v,cf));prev[(cam,gid,'crop')]=v
    prev[(cam,'finger')]=fv
   # Adjacent-only changes: disappeared identities do not bridge gaps.
   visible={(cam,int(gid)) for gid,vs in frame['objects'].items() for cam in vs}
   for k in list(prev):
    if len(k)==3 and (k[0],k[1]) not in visible:del prev[k]
   frames.append(frame)
   if t%20==0:save(R/'progress.json',{'time':now(),'frame':t,'total':388,'seconds':time.time()-t0})
 after=modelsha();assert before==after and all(p.grad is None and not p.requires_grad for p in net.parameters())
 with h5py.File(R/'features.h5','w') as out:
  out.create_dataset('record_t_camera_id',data=np.array(records,np.int32));out.create_dataset('object_features',data=np.array(vectors,np.float32),compression='lzf');out.create_dataset('masked_crop_features',data=np.array(cropvectors,np.float32),compression='lzf');out.create_dataset('finger_features',data=fingervec,compression='lzf')
 result={'title':'一个视频：冻结特征是否随交互变化','created':now(),'frames_total':388,'ids':sorted(allids),'labels':{'12':'第一只罐 #12','9':'第二只罐 #9','3':'对照候选 #3'},'protocol':protocol,'frames':frames,'findings':[],'training_updates':0}
 save(A/'results.json',result)
 complete={'time':now(),'frames':388,'views':776,'records':len(records),'ids':len(allids),'seconds':time.time()-t0,'training_updates':0,'weights_before':before,'weights_after':after,'same_input_max_feature_difference':selfcheck,'all_parameter_gradients_absent':True,'features_sha256':sha(R/'features.h5'),'source_sha256':sha(pathlib.Path(__file__))}
 save(R/'complete.json',complete)
except Exception as e:
 save(R/'failure.json',{'time':now(),'type':type(e).__name__,'frames':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
