from pathlib import Path
import json,numpy as np,importlib.util,h5py,datetime,hashlib,traceback
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/contact_preference/contact_persistence_three_v1'
O=D/'diagnosis_breaks';O.mkdir(exist_ok=True)
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
spec=importlib.util.spec_from_file_location('diag_three',D/'src/analyze_v3.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
cases=json.loads((D/'cases.json').read_text());res={}
for cid in ['mug','switch']:
 c=next(c for c in cases if c['id']==cid);base=D/'data'/cid;out=base/'estimates_v2'
 if cid=='mug':
  sf=67;s=[dict(np.load(base/f'{cam}_{sf}_source.npz')) for cam in c['cameras']];tr=[dict(np.load(base/f'{cam}_{sf}_tracks.npz')) for cam in c['cameras']]
  source=np.load(out/f'source_{sf}.npz');rd=[np.load(base/f'{cam}_robot_depth.npz')['depth'] for cam in c['cameras']]
  m=np.load(base/'motion.npz');rows=[]
  with h5py.File(c['rgbd'],'r') as h:
   for t in [48,49,50,51,52,53,76,77,78,80,90,94,95,96,100]:
    X,Y,v,_=M.observations(c,h,m,s,tr,t,rd);ii=np.flatnonzero(v&source['train']);hh=np.flatnonzero(v&source['held'])
    z=M.A.robust(X[ii],Y[ii],2026) if len(ii)>=8 else None
    row={'frame':t,'training_observations':len(ii),'heldout':len(hh)}
    if z is not None:
     rr,tt,im,err=z;he=np.linalg.norm(M.A.apply(X[hh],rr,tt)-Y[hh],axis=1)
     row.update(inliers=int(im.sum()),inlier_median_mm=float(np.median(err[im])*1000),held_median_mm=float(np.median(he)*1000),held_p95_mm=float(np.percentile(he,95)*1000))
     off=len(s[0]['seeds_world']);row['by_camera']=[]
     for cam,mask in [('main',hh<off),('wrist',hh>=off)]:
      a=he[mask];row['by_camera'].append({'camera':cam,'n':len(a),'median_mm':float(np.median(a)*1000) if len(a) else None,'p95_mm':float(np.percentile(a,95)*1000) if len(a) else None})
    rows.append(row)
  res[cid]=rows
 else:
  rows=[];motion=np.load(base/'motion.npz')
  for sf in [1,38,76]:
   src=np.load(out/f'source_{sf}.npz');ever=np.zeros(len(src['world']),bool);prev=None
   for t in range(1,c['frames']):
    p=out/f'{sf}_{t:03d}.npz'
    if not p.exists():continue
    a=np.load(p);active=(a['state']==1)|(a['state']==2)
    if t in [70,71,72,73,74,75,76,77,80,82,90]:
     masks={'stable':a['spread']<=.002,'normal_ok':src['normal_ok'],'normal_cos':a['normal_cos']>=.7,'near':a['distance']+a['spread']<=.003,'no_conflict':~a['conflict'],'seen_or_hidden':a['visible']|a['hidden']}
     cumulative=np.ones(len(active),bool);counts={}
     for k,mask in masks.items():cumulative &=mask;counts[k]=int(cumulative.sum())
     row={'source':sf,'frame':t,'positive':int(active.sum()),'sequential_geometric_pass':counts,'previous_positive':int(prev.sum()) if prev is not None else 0,'q_mm':(motion['measured_finger_q'][t]*1000).tolist()}
     if prev is not None and prev.any():
      row['previous_positive_checks']={'n':int(prev.sum()),'distance_mm':(a['distance'][prev]*1000).tolist(),'relative_delta_mm':(a['relative_delta'][prev]*1000).tolist(),'opening_delta_mm':(a['opening_delta'][prev]*1000).tolist(),'conflict':int(a['conflict'][prev].sum()),'normal_cos':a['normal_cos'][prev].tolist()}
     near=masks['stable']&masks['normal_ok']&masks['near']
     row['near_surface_rejections']={'n':int(near.sum()),'bad_normal':int((near&~masks['normal_cos']).sum()),'conflict':int((near&a['conflict']).sum()),'neither_seen_nor_hidden':int((near&~masks['seen_or_hidden']).sum())}
     rows.append(row)
    prev=active;ever|=active
  res[cid]=rows
save(O/'diagnostic.json',{'time':now,'no_labels_or_parameters_changed':True,'human_ranges_not_training':True,'results':res})
for r in res['mug']:print('mug',json.dumps(r))
for r in res['switch']:
 if r['frame'] in [74,75,76,80,82,90]:print('switch',json.dumps(r))
