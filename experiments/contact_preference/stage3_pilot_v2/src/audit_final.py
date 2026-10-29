import json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import norm,exponnorm,skewnorm
from scipy.integrate import quad
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_preference/stage3_pilot_v2')
O=R.parent/'stage3_pilot_v1'
S=json.loads((R/'evidence_summary.json').read_text())
models=json.loads((R/'fitted_models.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
locks={}
for path in [R/'input_locked.json',R/'source_locked.json',R/'evidence_source_locked.json',O/'completed_locked.json']:
 d=json.loads(path.read_text())
 if 'files' in d:d=d['files']
 bad=[]
 for p,h in d.items():
  if isinstance(h,dict):h=h.get('sha256')
  q=Path(p);q=q if q.is_absolute() else path.parent/q
  if not q.is_file() or sha(q)!=h:bad.append(str(q))
 locks[str(path)]={'count':len(d),'mismatch':bad}
 assert not bad,(path,len(bad))
def pvalue(d,s,m):
 if m['family']=='exp':ls=exponnorm.logpdf(d,m['mu']/s,loc=0,scale=s)
 else:ls=skewnorm.logpdf(d,m['mu']/s,loc=0,scale=np.sqrt(s*s+m['mu']**2))
 lc=norm.logpdf(d,scale=s)
 return np.exp(lc-np.logaddexp(lc,ls))
tot=np.zeros(5,dtype=int);err=0.;weberr=0.;quaderr=0.;rows=[]
for ci,c in enumerate(S['cases']):
 ms=[m for m in models if m['target_episode']==ci]
 assert len(ms)==12
 assert all(ci not in m['training_episode_ids'] for m in ms)
 ev=np.load(R/'points'/('event_features_c%d.npz'%ci))
 pp=np.array([pvalue(ev['mean_signed_gap'],ev[m['error_model']],m) for m in ms])
 ep=np.maximum(2*pp-1,0).min(0);en=np.maximum(1-2*pp,0).min(0)
 for m in ms:
  sig=float(np.median(ev[m['error_model']]));mu=m['mu']
  for d in [0.,sig,3*sig]:
   fun=lambda u: norm.pdf(d-u,scale=sig)*(np.exp(-u/mu)/mu if m['family']=='exp' else 2*norm.pdf(u,scale=mu))
   integral=quad(fun,0,max(mu*12,sig*20),epsabs=1e-9)[0]
   closed=np.exp(exponnorm.logpdf(d,mu/sig,scale=sig) if m['family']=='exp' else skewnorm.logpdf(d,mu/sig,scale=np.sqrt(sig*sig+mu*mu)))
   quaderr=max(quaderr,abs(integral-closed)/max(closed,1e-30))
 for row in c['rows']:
  t=row['frame'];z=np.load(R/row['data']);old=np.load(O/'points'/(c['key']+'_frame%d.npz'%t))
  for k in ['query_xyz','query_pixel','query_rgb','query_source_index','mapped_hand_xyz','mapped_world_xyz']:
   assert np.array_equal(z[k],old[k]),k
  ip=np.array([pvalue(z['signed_distance_m'],z['sigma_'+m['error_model']+'_m'],m) for m in ms])
  pos=np.minimum(ep,np.maximum(2*ip-1,0).min(0))
  neg=np.minimum(en,np.maximum(1-2*ip,0).min(0))
  pos[~z['opposing_normals']]=0
  excluded=z['excluded_mounting_nearest']|z['deep_penetration']
  pos[excluded]=0;neg[excluded]=0
  unknown=1-pos-neg
  for k,a in [('positive',pos),('separation',neg),('unknown',unknown)]:
   assert np.isfinite(a).all() and a.min()>=-1e-12
   err=max(err,float(np.max(np.abs(z[k]-a))))
  assert np.allclose(pos+neg+unknown,1)
  green=z['display_class']==0
  assert not np.any(green&excluded) and not np.any(green&~z['opposing_normals'])
  assert set(z['nearest_gripper_geom'][green].tolist()).issubset({3,4,6,7})
  count=np.bincount(z['display_class'].astype(int),minlength=5);tot+=count
  assert count.tolist()==[row['count'][str(i)] for i in range(5)]
  web=json.loads((R/'webdata'/('c%d_f%d.json'%(ci,t))).read_text())
  wa=np.asarray(web['supports'])
  target=np.stack([pos,neg,unknown],axis=1)
  if wa.shape!=target.shape:wa=wa.T
  weberr=max(weberr,float(np.max(abs(wa-target))))
  assert np.array_equal(web['category'],z['display_class'])
  row['current_visible_green']=int(np.sum(green&z['currently_depth_consistent']))
  row['occluded_green']=int(np.sum(green&~z['currently_depth_consistent']))
  assert row['current_visible_green']==row['green_visible'] and row['occluded_green']==row['green_occluded']
  rows.append({'episode':ci,'frame':t,'counts':count.tolist(),'green_scope_violations':0})
assert err<1e-10 and weberr<1e-5 and quaderr<1e-6
now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
S['summary_count_correction']={'time':now,'reason':'Two inherited visible-green fields were stale zeros; corrected to independently counted green_visible/green_occluded. Arrays and support unchanged.'}
(R/'evidence_summary.json').write_text(json.dumps(S,ensure_ascii=False,indent=2))
ws=json.loads((R/'webdata/summary.json').read_text())
for ci,c in enumerate(S['cases']):
 for ri,row in enumerate(c['rows']):
  ws['cases'][ci]['rows'][ri].update(row)
ws['summary_count_correction']=S['summary_count_correction']
(R/'webdata/summary.json').write_text(json.dumps(ws,ensure_ascii=False,indent=2))
report={'time':now,'locks':locks,'episodes':3,'endpoints':12,'point_time_rows':int(tot.sum()),'counts':tot.tolist(),'independent_support_max_error':err,'web_rounding_max_error':weberr,'conditional_density_integral_relative_error':quaderr,'all_models_heldout_episode_excluded':True,'optimizer_success_count':sum(m['optimizer_success'] for m in models),'model_count':len(models),'source_arrays_equal_old':True,'green_mount_or_nonopposing':0,'rows':rows,'limitations':['likelihood support not calibrated posterior','sub-noise separation may obtain positive support','source identity and real physical contact not ground-truth verified'],'scoring_network_trained':False}
(R/'audits').mkdir(exist_ok=True)
(R/'audits/completion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['locks','rows']},ensure_ascii=False))
