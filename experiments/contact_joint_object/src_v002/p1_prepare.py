import os,json,pathlib,time,hashlib,datetime,traceback
import numpy as np,h5py,cv2
R=pathlib.Path('/mnt/nvme1/drh/D_VLA')
B=R/'experiments/contact_joint_object'
RUN=B/'runs/v002/preflight/P1_semantics_v1'
ASSET=R/'docs/lcjp002_p1_assets'
CAMERAS=['agentview','robot0_eye_in_hand']
ZH={'alphabet soup':'字母汤罐','bbq sauce':'烧烤酱','butter':'黄油','chocolate pudding':'巧克力布丁','cream cheese':'奶油奶酪','ketchup':'番茄酱','milk':'牛奶','orange juice':'橙汁','salad dressing':'沙拉酱','tomato sauce':'番茄调味酱','basket':'篮子'}
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def rel(p):return str(p.relative_to(R/'docs'))
def color(i):return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[int((i*47)%180),200,240]]]),cv2.COLOR_HSV2RGB)[0,0])
def overlay(img,mask,oid):
 a=img.copy();c=color(oid);a[mask]=(a[mask].astype(float)*.6+np.array(c)*.4).astype(np.uint8)
 contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(a,contours,-1,c,1)
 ys,xs=np.where(mask)
 if len(xs):
  cv2.putText(a,'#'+str(oid),(int(xs.min()),max(12,int(ys.min()))),cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,255),2,cv2.LINE_AA)
  cv2.putText(a,'#'+str(oid),(int(xs.min()),max(12,int(ys.min()))),cv2.FONT_HERSHEY_SIMPLEX,.37,c,1,cv2.LINE_AA)
 return a
def png(p,a):cv2.imwrite(str(p),cv2.cvtColor(a,cv2.COLOR_RGB2BGR))
def main():
 RUN.mkdir(parents=True,exist_ok=True);ASSET.mkdir(parents=True,exist_ok=True)
 rows=[x for x in json.loads((B/'manifests/episodes.json').read_text())['episodes'] if x['split']=='train' and x['episode']=='demo_0']
 assert len(rows)==10 and len({x['task'] for x in rows})==10
 protocol={'created':now(),'stage':'P1','P0':'skipped_by_user','model':'google/siglip2-base-patch16-224','revision':'75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2','scope':'10 tasks / train demo_0 / two cameras','sampling':'5 uniform frames + first minimum gripper opening and peak end-effector speed, each with adjacent -1,0,+1 frames','mask_source':'existing SAM2.1 tiny current-frame automatic masks, 8x8 seed grid; original causal association IDs, not semantic truth','crop':'bbox + 10 percent padding, outside mask neutral RGB 127; semantic model sees crop only','texts':'a photo of {entity}.; instruction manipulated noun and basket plus fixed train metadata noun distractors and table/robot/background','thresholds':{'candidate_margin_logit':1.0,'minimum_mask_pixels':16,'minimum_depth_fraction':.5},'acceptance':'candidate requires top candidate margin >=1, entity top semantic label and >=50% valid depth; all remain visual-review pending','notes':['This step contains no scoring-network training or contact labels.','Rank is semantic match, not recognition accuracy.','Existing proposal IDs are hypotheses; temporal/cross-camera consistency remains under review.','All candidates and unknowns retained.','Task noun names come from instruction metadata; no coordinates or simulator object labels used.']}
 save(RUN/'protocol.json',protocol)
 doc=next((R/'docs').glob('LCJP002_*.md'))
 if 'P1_semantics_v1' not in doc.read_text():
  with doc.open('a',encoding='utf-8') as f:f.write('\n\n## P1 执行登记｜'+now()+'\n\n用户要求跳过第一步，直接执行第二步并用页面展示。执行编号 P1_semantics_v1。只检查10个任务的训练集 demo_0；使用已完成的双相机 RGB-D 和 SAM2.1 观测缓存，重新以冻结 SigLIP2 图文编码器绑定指令实体，不继承旧语义或评分。每段取5个均匀帧，并加入夹爪最小开度与末端最大速度时刻的相邻帧。全部候选、失败和未知保留；自动匹配与人工画面核对分别记录。匹配头尚未训练，本次检验预训练特征能否正确找物体。参数和输入哈希保存在 runs/v002/preflight/P1_semantics_v1/protocol.json 与 inputs.json。此步骤不启动打分网络训练。\n')
 result={'title':'第二步：物体有没有找对','created':now(),'protocol':protocol,'summary':{'tasks':10,'views':0,'candidates':0,'accepted':0,'unknown':0},'tasks':[]}
 inputs=[]
 for row in rows:
  name=row['task'].removeprefix('pick_up_the_').split('_and_place_it_')[0].replace('_',' ')
  instruction=row['task'].removesuffix('_demo').replace('_',' ')
  task={'id':row['id'],'name':name,'name_zh':ZH[name],'instruction':instruction,'entity':name,'entities':[{'name':name,'role':'manipulated','name_zh':ZH[name]},{'name':'basket','role':'destination','name_zh':'篮子（放置位置）'}],'frames':[]}
  rp=B/'cache/rgbd'/(row['id']+'.h5');op=B/'cache/observations'/(row['id']+'.h5')
  inputs.append({'id':row['id'],'split':row['split'],'rgbd_sha256':sha(rp),'observations_sha256':sha(op)})
  with h5py.File(rp,'r') as h,h5py.File(op,'r') as obs:
   T=int(h.attrs['num_frames']);task['frames_total']=T;chosen={}
   def add(t,why):chosen.setdefault(int(np.clip(t,0,T-1)),[]).append(why)
   for t in np.linspace(0,T-1,5).round().astype(int):add(t,'均匀取帧')
   q=h['robot/gripper_qpos'][:];opening=np.linalg.norm(q,axis=-1);p=h['robot/ee_position'][:]
   tc=int(np.argmin(opening));tv=int(np.argmax(np.linalg.norm(np.diff(p,axis=0),axis=-1)))+1
   for center,why in [(tc,'夹爪最小开度附近'),(tv,'夹爪移动最快附近')]:
    for dt in [-1,0,1]:add(center+dt,why)
   for t,reasons in sorted(chosen.items()):
    frame={'t':t,'reasons':list(dict.fromkeys(reasons)),'views':[]};g=obs['frames/'+str(t)]
    for ci,cam in enumerate(CAMERAS):
     folder=ASSET/row['id']/str(t)/str(ci);folder.mkdir(parents=True,exist_ok=True)
     rgb=h[cam+'/rgb'][t];depth=h[cam+'/depth_m'][t];K=h[cam+'/K'][t];C=h[cam+'/camera_to_world'][t]
     png(folder/'rgb.png',rgb);all_overlay=rgb.copy()
     view={'camera':cam,'camera_zh':['外部相机','腕部相机'][ci],'rgb':rel(folder/'rgb.png'),'overlay':rel(folder/'all.png'),'candidates':[],'match':{'status':'unknown','reason':'等待语义模型推理','top_id':None,'runner_up_id':None,'top_logit':None,'margin':None},'audit':{'status':'unreviewed','reason':'等待画面核对'}}
     for key in g:
      if not key.startswith('object_'):continue
      obj=g[key];mk='mask_'+str(ci)
      if mk not in obj:continue
      mask=np.unpackbits(obj[mk][:])[:256*256].reshape(256,256).astype(bool);ys,xs=np.where(mask)
      if not len(xs):continue
      oid=int(key.split('_')[1]);x0,x1=int(xs.min()),int(xs.max())+1;y0,y1=int(ys.min()),int(ys.max())+1
      pad=max(2,int(.1*max(x1-x0,y1-y0)));xa=max(0,x0-pad);xb=min(256,x1+pad);ya=max(0,y0-pad);yb=min(256,y1+pad)
      masked=rgb.copy();masked[~mask]=127;png(folder/(key+'_crop.png'),masked[ya:yb,xa:xb])
      ov=overlay(rgb,mask,oid);png(folder/(key+'_overlay.png'),ov);all_overlay=overlay(all_overlay,mask,oid)
      valid=mask&np.isfinite(depth)&(depth>0);yy,xx=np.where(valid);d=depth[yy,xx]
      camera_xyz=np.c_[(xx-K[0,2])*d/K[0,0],(yy-K[1,2])*d/K[1,1],d]
      world=np.c_[camera_xyz,np.ones(len(d))]@C.T
      cloud={'xyz':np.round(world[:,:3],6).tolist(),'rgb':rgb[yy,xx].tolist(),'uv':np.c_[xx,yy].tolist()}
      save(folder/(key+'_cloud.json'),cloud)
      item={'id':oid,'mask_area':int(mask.sum()),'valid_depth_points':int(valid.sum()),'bbox':[x0,y0,x1,y1],'crop':rel(folder/(key+'_crop.png')),'overlay':rel(folder/(key+'_overlay.png')),'pointcloud':rel(folder/(key+'_cloud.json')),'semantic_top':[],'target_logit':None,'rank':None,'entity_logits':{},'sam_quality':float(obj.attrs.get('sam_quality',0)),'identity_new':bool(obj.attrs.get('new_identity',True))}
      view['candidates'].append(item)
     png(folder/'all.png',all_overlay);frame['views'].append(view);result['summary']['views']+=1;result['summary']['candidates']+=len(view['candidates'])
    task['frames'].append(frame)
  result['tasks'].append(task);save(RUN/'prepare_progress.json',{'time':now(),'tasks':len(result['tasks']),'views':result['summary']['views'],'candidates':result['summary']['candidates']})
 save(RUN/'inputs.json',inputs);save(RUN/'prepared.json',result)
 save(RUN/'prepare_complete.json',{'time':now(),'tasks':len(result['tasks']),'views':result['summary']['views'],'candidates':result['summary']['candidates'],'source_sha256':sha(pathlib.Path(__file__))})
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(RUN/'prepare_failure.json',{'type':type(e).__name__,'locations':[{'file':pathlib.Path(v.filename).name,'line':v.lineno} for v in traceback.extract_tb(e.__traceback__)]});raise SystemExit(1)
