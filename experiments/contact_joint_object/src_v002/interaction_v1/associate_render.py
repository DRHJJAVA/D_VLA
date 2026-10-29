
import argparse, pathlib, json, hashlib, datetime, sys, math, traceback, time
import numpy as np
import h5py, cv2
from scipy.spatial import cKDTree
BASE=pathlib.Path('/mnt/nvme1/drh/D_VLA')
EXP=BASE/'experiments/contact_joint_object'
RUN=EXP/'runs/v002/preflight/P1_interaction_v1'
ASSET=BASE/'docs/lcjp002_interaction_v1_assets'
sys.path.insert(0,str(EXP/'src'))
from geometry import finger_geometry
CAMS=['agentview','robot0_eye_in_hand']
CFG=dict(version='geometry_motion_association_v1',max_geometry_points=320,min_flow_points=6,
         flow_fb_error_px=1.2,flow_max_step_px=35,depth_relative_change_limit=0.25,
         crossview_depth_tolerance_m=0.012,crossview_match_min=0.5,crossview_match_margin=0.1,
         proximity_scale_m=0.02,association_threshold=0.55,motion_scale_m=0.003,
         score_formula='exp(-(nearest_finger_distance/.02)^2) * (.3+.7*motion_support)',
         motion_formula='clip((median_world_displacement-median_gripper_rigid_residual)/(median_world_displacement+.002),0,1) * min(median_gripper_expected_displacement/.003,1)',
         static_identity_hold='previously associated, near finger, valid flow shows gripper and object <.003 m/frame',
         background_plane_width_m=0.25,background_plane_thickness_m=0.005)

class PrefixDataset:
 def __init__(self,ds,t):self.ds=ds;self.t=t;self.max_read=-1
 def __len__(self):return self.t+1
 @property
 def shape(self):return (self.t+1,)+self.ds.shape[1:]
 def __getitem__(self,index):
  first=index[0] if isinstance(index,tuple) else index
  if not isinstance(first,(int,np.integer)) or first<0 or first>self.t:raise IndexError('outside_current_prefix')
  self.max_read=max(self.max_read,int(first));return self.ds[index]
class CurrentGeometryView:
 def __init__(self,h,t):
  self.h=h;self.t=t
  self.temporal={k:PrefixDataset(h[k],t) for k in ['robot/geom_position','robot/geom_rotation']}
 def __getitem__(self,k):
  if k in self.temporal:return self.temporal[k]
  if k=='finger_surface':return self.h[k]
  raise KeyError(k)

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def writej(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sample(a,n=600):return a if len(a)<=n else a[np.linspace(0,len(a)-1,n).astype(int)]
def back(dep,K,C,uv):
 z=dep[uv[:,1],uv[:,0]]
 x=np.column_stack(((uv[:,0]-K[0,2])*z/K[0,0],(uv[:,1]-K[1,2])*z/K[1,1],z))
 return x@C[:3,:3].T+C[:3,3]
def maskof(g):
 a=np.asarray(g['mask'])
 return np.unpackbits(a.reshape(-1))[:65536].reshape(256,256).astype(bool) if a.size!=65536 else a.reshape(256,256).astype(bool)
def projection_score(obj,view,other):
 pts=sample(obj['xyz'],700)
 if len(pts)<8:return 0.,0
 C=view['C'];a=(pts-C[:3,3])@C[:3,:3];z=a[:,2]
 good=z>0
 uv=np.rint(a[:,:2]/np.maximum(z[:,None],1e-8)*np.array([view['K'][0,0],view['K'][1,1]])+view['K'][:2,2]).astype(int)
 good&=(uv[:,0]>=0)&(uv[:,0]<256)&(uv[:,1]>=0)&(uv[:,1]<256)
 uv=uv[good];z=z[good]
 if not len(z):return 0.,0
 dz=view['dep'][uv[:,1],uv[:,0]]
 vis=np.isfinite(dz)&(dz>0)&(np.abs(dz-z)<CFG['crossview_depth_tolerance_m'])&(~view['robot'][uv[:,1],uv[:,0]])
 uv=uv[vis]
 return (float(other['mask'][uv[:,1],uv[:,0]].mean()),len(uv)) if len(uv)>=8 else (0.,len(uv))
class Identity:
 def __init__(self):self.keys={};self.parent={};self.next=1
 def root(self,a):
  while self.parent[a]!=a:self.parent[a]=self.parent[self.parent[a]];a=self.parent[a]
  return a
 def get(self,k):
  if k not in self.keys:self.keys[k]=self.next;self.parent[self.next]=self.next;self.next+=1
  return self.root(self.keys[k])
 def union(self,a,b):
  a,b=self.root(a),self.root(b)
  if a!=b:self.parent[max(a,b)]=min(a,b)
  return min(a,b)
def flows(prev,cur):
 p=cv2.goodFeaturesToTrack(prev['gray'],maxCorners=2500,qualityLevel=.008,minDistance=3,mask=(~prev['robot']).astype(np.uint8)*255,blockSize=3)
 if p is None:return None
 q,s,_=cv2.calcOpticalFlowPyrLK(prev['gray'],cur['gray'],p,None,winSize=(17,17),maxLevel=3)
 if q is None:return None
 r,s2,_=cv2.calcOpticalFlowPyrLK(cur['gray'],prev['gray'],q,None,winSize=(17,17),maxLevel=3)
 if r is None:return None
 p=p[:,0];q=q[:,0];r=r[:,0]
 v=(s[:,0]>0)&(s2[:,0]>0)&(np.linalg.norm(r-p,axis=1)<CFG['flow_fb_error_px'])&(np.linalg.norm(q-p,axis=1)<CFG['flow_max_step_px'])
 pi=np.rint(p).astype(int);qi=np.rint(q).astype(int)
 v&=(pi[:,0]>=0)&(pi[:,0]<256)&(pi[:,1]>=0)&(pi[:,1]<256)&(qi[:,0]>=0)&(qi[:,0]<256)&(qi[:,1]>=0)&(qi[:,1]<256)
 pi=pi[v];qi=qi[v]
 z0=prev['dep'][pi[:,1],pi[:,0]];z1=cur['dep'][qi[:,1],qi[:,0]]
 good=np.isfinite(z0)&np.isfinite(z1)&(z0>0)&(z1>0)&(~prev['robot'][pi[:,1],pi[:,0]])&(~cur['robot'][qi[:,1],qi[:,0]])
 pi=pi[good];qi=qi[good]
 x0=back(prev['dep'],prev['K'],prev['C'],pi);x1=back(cur['dep'],cur['K'],cur['C'],qi)
 return pi,qi,x0,x1
def color(gid):
 h=(gid*0.61803398875)%1
 a=np.uint8([[[int(h*179),180,235]]]);return tuple(int(x) for x in cv2.cvtColor(a,cv2.COLOR_HSV2RGB)[0,0])
def overlay(rgb,objs,active,mode):
 out=rgb.copy()
 selected=objs if mode=='all' else [o for o in objs if o['gid'] in active]
 for o in sorted(selected,key=lambda z:-z['area']):
  m=o['mask'];c=np.array(color(o['gid']))
  alpha=.08 if o['background'] else (.35 if o['gid'] in active else .19)
  out[m]=np.clip(out[m]*(1-alpha)+c*alpha,0,255).astype(np.uint8)
 for o in selected:
  c=color(o['gid']);cnt,_=cv2.findContours(o['mask'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(out,cnt,-1,c,2 if o['gid'] in active else 1)
  y,x=np.where(o['mask'])
  if len(x):
   xy=(int(np.median(x)),max(10,int(y.min())-2));text=('#'+str(o['gid'])+('*' if o['gid'] in active else ''))
   cv2.putText(out,text,xy,cv2.FONT_HERSHEY_SIMPLEX,.34,(0,0,0),2,cv2.LINE_AA)
   cv2.putText(out,text,xy,cv2.FONT_HERSHEY_SIMPLEX,.34,c,1,cv2.LINE_AA)
 return out
def episode(ep,limit=None):
 start=time.time();sp=RUN/'segmentation'/f'{ep}.h5';rp=EXP/'cache/rgbd'/f'{ep}.h5'
 target=ASSET/ep;target.mkdir(parents=True,exist_ok=True)
 frames=[];ident=Identity();prev_views=None;prev_by_key={};active_prev=set();last_seen={};ever=set()
 with h5py.File(rp,'r') as h,h5py.File(sp,'r') as seg:
  n=int(h.attrs['num_frames']);n=min(n,limit) if limit else n
  meta=json.loads(h.attrs['robot_metadata']);hand_idx=meta['geom_names'].index('gripper0_hand_collision')
  for t in range(n):
   views=[];allobjs=[]
   for ci,cam in enumerate(CAMS):
    v=dict(rgb=h[cam]['rgb'][t],dep=h[cam]['depth_m'][t],K=h[cam]['K'][t],C=h[cam]['camera_to_world'][t],robot=h[cam]['robot_mask'][t].astype(bool))
    v['gray']=cv2.cvtColor(v['rgb'],cv2.COLOR_RGB2GRAY);obs=[]
    group=seg[f'frames/{t}/{ci}/objects']
    for lid,g in group.items():
     m=maskof(g);y,x=np.where(m);uv=np.column_stack((x,y))
     valid=np.isfinite(v['dep'][y,x])&(v['dep'][y,x]>0)&(~v['robot'][y,x]);uv=uv[valid]
     if len(uv)<16:continue
     m=np.zeros((256,256),bool);m[uv[:,1],uv[:,0]]=True
     xyz=back(v['dep'],v['K'],v['C'],uv);pts=sample(xyz,600);ext=np.ptp(pts,axis=0)
     vals=np.linalg.eigvalsh(np.cov(pts.T));plane=float(np.sqrt(max(vals[0],0)))
     bg=bool((len(uv)>65536*.15 and np.max(ext)>.25 and plane<.005) or len(uv)>65536*.65)
     key=(ci,int(lid));gid=ident.get(key)
     o=dict(key=key,lid=int(lid),gid=gid,cam=ci,mask=m,area=len(uv),xyz=pts,background=bg,quality=float(g.attrs.get('quality_at_last_detection',g.attrs.get('sam_quality_at_last_detection',g.attrs.get('quality',0)))),source=str(g.attrs.get('source','video')))
     obs.append(o);allobjs.append(o)
    v['objects']=obs;views.append(v)
   a,b=views[0]['objects'],views[1]['objects'];M=np.zeros((len(a),len(b)))
   for i,x in enumerate(a):
    if x['background']:continue
    for j,y in enumerate(b):
     if y['background'] or np.linalg.norm(np.median(x['xyz'],axis=0)-np.median(y['xyz'],axis=0))>.22:continue
     p,np1=projection_score(x,views[1],y);q,np2=projection_score(y,views[0],x)
     if np1>=8 and np2>=8:M[i,j]=min(p,q)
     elif max(np1,np2)>=12:M[i,j]=max(p,q)*.85
   matched=[]
   if len(a) and len(b):
    for i in range(len(a)):
     j=int(M[i].argmax());score=M[i,j]
     row=np.sort(M[i]);col=np.sort(M[:,j])
     margin=min(score-(row[-2] if len(row)>1 else 0),score-(col[-2] if len(col)>1 else 0))
     if score>=CFG['crossview_match_min'] and int(M[:,j].argmax())==i and margin>=CFG['crossview_match_margin']:
      gid=ident.union(a[i]['gid'],b[j]['gid']);matched.append(gid)
   groups={}
   for o in allobjs:
    o['gid']=ident.root(o['gid']);groups.setdefault(o['gid'],[]).append(o)
   active_prev={ident.root(g) for g in active_prev};ever={ident.root(g) for g in ever}
   flowsets=[flows(prev_views[c],views[c]) if prev_views else None for c in range(2)]
   objects=[];active=set()
   R=h['robot/geom_rotation'][t,hand_idx];P=h['robot/geom_position'][t,hand_idx]
   R0=h['robot/geom_rotation'][t-1,hand_idx] if t else R;P0=h['robot/geom_position'][t-1,hand_idx] if t else P
   for gid,oo in groups.items():
    last_seen[gid]=t;bg=all(o['background'] for o in oo);xyz=sample(np.concatenate([o['xyz'] for o in oo]),CFG['max_geometry_points'])
    ds=[None,None];count=0;res=None;disp=None;expected=None;motion=0.
    if not bg:
     geo=finger_geometry(CurrentGeometryView(h,t),t,xyz);sd=np.abs(geo['sdf']);ds=[float(np.quantile(sd[:,j],.03)) for j in range(2)]
     xs=[];ys=[]
     for o in oo:
      fs=flowsets[o['cam']];old=prev_by_key.get(o['key'])
      if fs is None or old is None:continue
      pi,qi,x0,x1=fs;v=old['mask'][pi[:,1],pi[:,0]]&o['mask'][qi[:,1],qi[:,0]]
      if v.any():xs.append(x0[v]);ys.append(x1[v])
     if xs:
      xp=np.concatenate(xs);xc=np.concatenate(ys)
      shift=np.linalg.norm(xc-xp,axis=1);good=shift<.12;xp=xp[good];xc=xc[good];count=len(xp)
      if count>=CFG['min_flow_points']:
       pred=(xp-P0)@R0@R.T+P
       res=float(np.median(np.linalg.norm(xc-pred,axis=1)));disp=float(np.median(np.linalg.norm(xc-xp,axis=1)));expected=float(np.median(np.linalg.norm(pred-xp,axis=1)))
       motion=float(np.clip((disp-res)/(disp+.002),0,1)*min(expected/CFG['motion_scale_m'],1))
    nearest=min(ds) if ds[0] is not None else None
    proximity=math.exp(-(nearest/CFG['proximity_scale_m'])**2) if nearest is not None else 0.
    score=float(proximity*(.3+.7*motion)) if not bg else 0.
    hold=gid in active_prev and proximity>=.5 and count>=CFG['min_flow_points'] and expected is not None and expected<.003 and disp<.003
    state='candidate';reason='background_plane' if bg else ('no_motion_correspondence' if count<CFG['min_flow_points'] else 'insufficient_interaction_evidence')
    if not bg and score>=CFG['association_threshold']:
     state='associated';reason='near_finger_and_coupled_world_motion';active.add(gid);ever.add(gid)
    elif not bg and hold:state='maintained';reason='stationary_near_finger_with_previous_association';active.add(gid)
    objects.append(dict(id=gid,color='#'+''.join(f'{z:02x}' for z in color(gid)),state=state,selected=gid in active,visible_cameras=[CAMS[o['cam']] for o in oo],local_ids={CAMS[o['cam']]:o['lid'] for o in oo},
                        areas={CAMS[o['cam']]:o['area'] for o in oo},sources={CAMS[o['cam']]:o['source'] for o in oo},finger_distances_m=ds,min_finger_distance_m=nearest,
                        motion_valid_points=count,motion_residual_m=res,world_displacement_m=disp,gripper_expected_displacement_m=expected,motion_similarity=motion,
                        relative_motion_m=res,association_score=score,background_like=bg,cross_view_status='matched' if len(oo)>1 else 'single_view',evidence=reason,reason=reason))
   missing=[g for g in ever if g not in groups and t-last_seen.get(g,t)<=16]
   for gid in missing:objects.append(dict(id=gid,state='invisible',selected=False,visible_cameras=[],occlusion_age=t-last_seen[gid],reason='previous_identity_not_visible',association_score=None))
   outviews={}
   for ci,cam in enumerate(CAMS):
    v=views[ci];paths={}
    for mode,img in [('rgb',v['rgb']),('all',overlay(v['rgb'],v['objects'],active,'all')),('associated',overlay(v['rgb'],v['objects'],active,'associated'))]:
     name=f'{t:04d}_{cam}_{mode}.jpg';cv2.imwrite(str(target/name),cv2.cvtColor(img,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,94])
     paths[mode]=f'lcjp002_interaction_v1_assets/{ep}/{name}'
    outviews[cam]={**paths,'visible':bool(v['objects']),'track_ids':sorted({o['gid'] for o in v['objects']})}
   frames.append(dict(t=t,views=outviews,objects=objects,active_ids=sorted(active),identified_ids=sorted(groups),stage='associated' if active else 'searching',note=''))
   active_prev=active;prev_views=views;prev_by_key={o['key']:o for o in allobjs}
   if t%10==0:writej(RUN/'association_progress.json',dict(time=now(),episode=ep,frame=t,total=n,seconds=time.time()-start,active_ids=sorted(active)))
  title={'ketchup':'番茄酱','orange_juice':'橙汁','bbq_sauce':'烧烤酱'}
  key=ep.split('pick_up_the_')[1].split('_and_place')[0]
  result=dict(id=ep,label=title.get(key,key),split='train',frames_total=n,fps=20,frames=frames,summary=dict(frames=n,active_frames=sum(bool(f['active_ids']) for f in frames),unknown_frames=sum(not bool(f['active_ids']) for f in frames),multi_active_frames=sum(len(f['active_ids'])>1 for f in frames)),input_sha256=sha(rp),segmentation_sha256=sha(sp),association_source_sha256=sha(pathlib.Path(__file__)),seconds=time.time()-start)
  writej(RUN/'association'/f'{ep}.json',result)
  manifest=ASSET/'results.json'
  if manifest.exists():data=json.loads(manifest.read_text())
  else:data=dict(title='双视角交互物体锁定',created=now(),status='running',method='冻结SAM2.1分割与视频跟踪；双视角几何融合；两指几何和世界运动关联证据。未训练关系网络，本页不显示部位评分。',summary={},episodes=[])
  data['episodes']=[x for x in data['episodes'] if x['id']!=ep]+[result]
  order=json.loads((RUN/'protocol.json').read_text())['episodes'];data['episodes'].sort(key=lambda x:order.index(x['id']))
  data['status']='complete' if len(data['episodes'])==3 else 'running';data['updated']=now();data['summary']={'episodes':len(data['episodes']),'frames':sum(x['frames_total'] for x in data['episodes']),'active_frames':sum(x['summary']['active_frames'] for x in data['episodes'])}
  writej(manifest,data);writej(RUN/f'association_{key}_complete.json',dict(time=now(),episode=ep,summary=result['summary'],seconds=result['seconds']))
  return result['summary']
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--episode',required=True);parser.add_argument('--limit',type=int);args=parser.parse_args()
 try:
  config=RUN/'association_config.json'
  if config.exists():assert json.loads(config.read_text())==CFG
  else:writej(config,CFG)
  print(json.dumps(episode(args.episode,args.limit)))
 except Exception as exc:
  frames=[{'file':pathlib.Path(x.filename).name,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(exc.__traceback__)]
  writej(RUN/'association_failure.json',dict(time=now(),exception=type(exc).__name__,frames=frames))
  print(json.dumps({'error':type(exc).__name__,'frames':frames}));sys.exit(1)
