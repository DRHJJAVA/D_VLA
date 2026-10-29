"""R4 offline-only supervision compiler; inference data and labels are separate."""
import os,json,pathlib,hashlib,traceback,time,sys,functools
import numpy as np
import torch,h5py
from scipy.spatial import cKDTree
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA')
RUN=ROOT/'experiments/lcjp003/r4/joint/v001'
R3=ROOT/'experiments/lcjp003/r3/joint/v001'
R2=ROOT/'experiments/lcjp003/r2/v2/v001'
CAMS=['agentview','robot0_eye_in_hand']
CFG=json.loads((RUN/'config.json').read_text())
def atomic_json(name,x):
 p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def fail(stage,ex):
 atomic_json('failure.json',{'stage':stage,'type':type(ex).__name__,'summary':str(ex)[:250],'frames':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(ex.__traceback__)],'time':time.time()})
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def legacy():
 sys.path.insert(0,str(R3/'src'))
 import geometry,supervision
 return geometry,supervision
def obs_file(e,t):
 for root in [R3/'recovered_observations/cache'/e['id'],R2/'cache'/e['id']]:
  p=root/('observations_%d.h5'%t)
  if p.exists():return p
  p=root/'observations.h5'
  if p.exists():
   with h5py.File(p,'r') as f:
    if 'frames/'+str(t) in f:return p
 raise FileNotFoundError('current_observation_missing:'+e['id']+':'+str(t))
def array(x,dtype=None):return np.asarray(x,dtype=dtype)
def get_frame(e,t):
 assert e['split'] in ['train','dev','development','val']
 assert int(e['episode'].split('_')[-1]) in list(range(8))+[45,46]
 geometry,supervision=legacy()
 objects,meta=geometry.load_objects(e,t)
 prev=torch.load(R2/'cache'/e['id']/(str(t)+'.pt'),weights_only=False,map_location='cpu')
 fi,world,uv,good,view,evidence=supervision.episode_tracks(e['id'])
 ti=int(np.flatnonzero(fi==t)[0])
 # Future is restricted to teacher construction, never put in inputs.
 wt=np.exp(-np.maximum(fi-t,0)/48.0)*(fi>=t)
 support=(evidence*wt[:,None]).max(0)
 active=good[ti]&np.isfinite(world[ti]).all(-1)
 pri=prev['priors'].numpy();ids=np.asarray(prev['candidate_ids'])
 masses=[];sems=[];dest=[];members=[]
 for o in objects:
  member=active.copy()
  inside=np.zeros(len(member),bool)
  for c in range(2):
   ix=np.flatnonzero(active&(view==c))
   p=uv[ti,ix];valid=(p[:,0]>=0)&(p[:,0]<256)&(p[:,1]>=0)&(p[:,1]<256)
   ix=ix[valid];p=p[valid]
   inside[ix]=o['masks'][c][p[:,1],p[:,0]]
  member &= inside
  masses.append(float(support[member].sum()/max(member.sum(),1)))
  matches=np.isin(ids,o['ids'])
  sems.append(float(pri[matches,0].max()) if matches.any() else .0)
  dest.append(float(pri[matches,1].max()) if matches.any() else .0)
  members.append(member)
 masses=np.asarray(masses);sems=np.asarray(sems);dest=np.asarray(dest)
 probs=masses*np.exp((sems-sems.max())/.15)
 confidence=float(masses.max()) if len(masses) else 0.
 probs=probs/max(probs.sum(),1e-12)
 target=int(probs.argmax());destination=int(dest.argmax())
 # Ambiguous motion/semantic binding reduces supervision instead of forcing a label.
 confidence*=float(probs[target])
 masks=np.zeros((2,2,256,256),np.float32)
 mask_weight=np.zeros_like(masks)
 existence=np.zeros((2,2),np.float32);exist_weight=np.zeros((2,2),np.float32)
 rgb=[];depth=[];robot=[];K=[];T=[];features=[]
 rp=ROOT/'experiments/contact_joint_object/cache/rgbd'/(e['id']+'.h5')
 with h5py.File(rp,'r') as h,h5py.File(obs_file(e,t),'r') as f:
  ix=int(np.flatnonzero(h['frame_indices'][:]==t)[0]);fg=f['frames/'+str(t)]
  for c,name in enumerate(CAMS):
   rgb.append(h[name+'/rgb'][ix]);depth.append(h[name+'/depth_m'][ix]);robot.append(h[name+'/robot_mask'][ix]>0)
   K.append(h[name+'/K'][ix]);T.append(h[name+'/camera_to_world'][ix]);features.append(fg['features_'+str(c)][:])
   for role,oi,quality in [(0,target,confidence),(1,destination,float(max(dest[destination]-.5,0)))]:
    ma=objects[oi]['masks'][c]&~robot[-1]
    masks[c,role]=ma
    if ma.any():
     # Interior foreground and confidently different objects provide balanced BCE.
     known=np.zeros((256,256),np.float32)
     for oj,o in enumerate(objects):
      if oj!=oi:known[o['masks'][c]]=1
     known[ma]=1
     known[robot[-1]]=0
     mask_weight[c,role]=known*quality
     existence[c,role]=1;exist_weight[c,role]=quality
  ee=h['robot/ee_position'][ix]
 # Local track support: scale-adaptive support on the selected visible surface only.
 part=np.zeros((2,256,256),np.float32);part_weight=np.zeros_like(part)
 m=members[target];wp=world[ti,m];sp=support[m]
 if len(wp) and sp.max()>1e-8:
  o=objects[target];scale=float(np.linalg.norm(np.ptp(o['xyz'],axis=0)))
  bandwidth=float(np.clip(scale/16,.002,.012))
  tree=cKDTree(wp);dist,idx=tree.query(o['xyz'],k=1)
  proximity=np.exp(-.5*(dist/bandwidth)**2)
  values=sp[idx]/max(float(sp.max()),1e-8)*proximity
  # Observation support is graded; unknown/occluded pixels receive zero loss weight.
  weights=proximity*np.sqrt(np.clip(sp[idx],0,1))*confidence
  for c in range(2):
   take=o['camera']==c;p=o['uv'][take];vv=values[take];ww=weights[take]
   if len(p):
    part[c,p[:,1],p[:,0]]=vv
    part_weight[c,p[:,1],p[:,0]]=ww
 inputs={'features':torch.from_numpy(np.stack(features)).half(),'rgb':torch.from_numpy(np.stack(rgb)),
 'depth':torch.from_numpy(np.stack(depth)).float(),'robot':torch.from_numpy(np.stack(robot)),
 'K':torch.from_numpy(np.stack(K)).float(),'T':torch.from_numpy(np.stack(T)).float(),
 'ee':torch.from_numpy(ee).float(),'rotation':torch.from_numpy(meta['rotation']).float(),
 'fingers':torch.from_numpy(meta['fingers']).float(),'h':prev['h'].float(),'words':prev['words'].float()}
 labels={'mask':torch.from_numpy(masks),'mask_weight':torch.from_numpy(mask_weight),
 'existence':torch.from_numpy(existence),'exist_weight':torch.from_numpy(exist_weight),
 'part':torch.from_numpy(part),'part_weight':torch.from_numpy(part_weight),
 'target':prev['target'].float(),'grip':prev['grip'].long()}
 # Previous executed action is causal; switch targets use expert labels only for training.
 raw=ROOT/e['raw']
 with h5py.File(raw,'r') as h:
  acts=h['data/'+e['episode']+'/actions']
  prior=float(acts[max(t-1,0),6])
 labels['switch']=torch.cat([torch.tensor([int((prior>0)!=bool(labels['grip'][0]))]),(labels['grip'][1:]!=labels['grip'][:-1]).long()]).float()
 return {'inputs':inputs,'labels':labels,'metadata':{'episode':e['id'],'frame':t,'split':e['split'],
 'teacher_target':target,'teacher_confidence':confidence,'candidate_ids':objects[target]['ids'],
 'target_visible_pixels':[int(x.sum()) for x in masks[:,0]],'future_in_inputs':False}}
def path_for(e,t):return RUN/'cache'/e['id']/(str(t)+'.pt')
def prepare_one(job):
 import fcntl
 e,t=job;p=path_for(e,t)
 p.parent.mkdir(parents=True,exist_ok=True)
 lock=open(p.with_suffix('.lock'),'w');fcntl.flock(lock,fcntl.LOCK_EX)
 if p.exists():return json.loads(p.with_suffix('.json').read_text())
 q=get_frame(e,t);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix('.tmp');torch.save(q,tmp);tmp.replace(p)
 p.with_suffix('.json').write_text(json.dumps(q['metadata']))
 return q['metadata']
def load_sample(record):
 return torch.load(RUN/'cache'/record['episode']/(str(record['frame'])+'.pt'),weights_only=False,map_location='cpu')
if __name__=='__main__':
 import concurrent.futures,fcntl
 lock=open(RUN/'prepare.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  episodes=json.loads((RUN/'episodes.json').read_text());lookup={e['id']:e for e in episodes}
  records=json.loads((R3/'input_manifest.json').read_text())
  evals=json.loads((R3/'eval0000.json').read_text())['records']
  evalkeys={(e['episode'],e['frame']) for e in evals}
  # Reuse exact R3 sampled training windows and all fixed evaluation windows.
  selected={(x['episode'],int(x['frame'])) for x in records if lookup[x['episode']]['split']=='train'}|evalkeys
  jobs=[(lookup[e],t) for e,t in sorted(selected)]
  manifest=[{'episode':e['id'],'frame':t,'split':e['split']} for e,t in jobs]
  atomic_json('input_manifest.json',manifest);atomic_json('evaluation_windows.json',[{'episode':x['episode'],'frame':x['frame'],'split':x['split']} for x in evals])
  schedule=json.loads((RUN/'sampling_schedule.json').read_text()) if (RUN/'sampling_schedule.json').exists() else []
  priority=[(x['episode'],int(x['frame'])) for x in evals]+[(x['episode'],int(x['frame'])) for batch in schedule for x in batch]
  ordered=list(dict.fromkeys(priority+sorted(selected)))
  jobs=[(lookup[e],t) for e,t in ordered]
  results=[];start=time.time()
  limit=int(os.environ.get('R4_PREP_LIMIT','0'))
  if limit:jobs=jobs[:limit]
  with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
   for md in pool.map(prepare_one,jobs,chunksize=1):
    results.append(md)
    if len(results)%10==0 or len(results)==len(jobs):
     atomic_json('preparation_state.json',{'status':'preparing','completed':len(results),'total':len(jobs),'seconds':time.time()-start,'pid':os.getpid()})
  atomic_json('teacher_summary.json',{'count':len(results),'mean_confidence':float(np.mean([x['teacher_confidence'] for x in results])),
   'positive_confidence_fraction':float(np.mean([x['teacher_confidence']>.01 for x in results])),'teacher_is_truth':False})
  atomic_json('preparation_state.json',{'status':'partial_check' if limit else 'complete','completed':len(results),'total':len(jobs),'seconds':time.time()-start,'pid':os.getpid()})
 except Exception as ex:fail('prepare',ex);raise SystemExit(1)
