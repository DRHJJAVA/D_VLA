from pathlib import Path
import numpy as np,json,hashlib
from scipy.special import ndtr,expit
from scipy.stats import exponnorm
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/stage3_temporal_support_v3';B=R/'experiments/contact_preference/stage3_pilot_v2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for q in iter(lambda:f.read(8*1024*1024),b''):h.update(q)
 return h.hexdigest()
def contrast(d,s,m):
 fc=np.exp(-.5*(d/s)**2)/(np.sqrt(2*np.pi)*s)
 if m['family']=='exp':fs=exponnorm.pdf(d,m['mu']/s,scale=s)
 else:
  total=np.sqrt(s*s+m['mu']**2)
  fs=2*np.exp(-.5*(d/total)**2)/(np.sqrt(2*np.pi)*total)*ndtr(d*m['mu']/(s*total))
 return (fc-fs)/(fc+fs+1e-300)
def cat(p,n,u):
 a=np.array([p,n,u]);v=a.argmax(0);v[(a==a.max(0)).sum(0)>1]=2;return v
def main():
 models=json.loads((B/'fitted_models.json').read_text());summary=json.loads((O/'summary.json').read_text());rows=[];maxerr=0
 for ci,c in enumerate(summary['cases']):
  prev=None;Pprev=None;occluded_ids=[];shown=[];controltests=[]
  mm=[m for m in models if m['target_episode']==ci]
  for row in c['rows']:
   t=row['frame'];z=dict(np.load(O/'points'/f"{c['key']}_frame{t}.npz"));old=dict(np.load(B/'points_evidence'/f"{c['key']}_frame{t}.npz"))
   for k in ['query_xyz','query_rgb','query_pixel','query_source_index','mapped_world_xyz','mapped_hand_xyz']:
    assert np.array_equal(z[k],old[k])
   ex=np.array([contrast(old['signed_distance_m'],old['sigma_base_m'] if m['error_model']=='base' else old['sigma_proxy_m'],m) for m in mm])
   gp=np.maximum(ex,0).min(0);gn=np.maximum(-ex,0).min(0)
   bad=old['excluded_mounting_nearest']|old['deep_penetration']|np.isin(z['visibility'],[2,3,4])
   gp[bad|~old['opposing_normals']]=0;gn[bad]=0
   maxerr=max(maxerr,float(np.max(abs(gp-z['geometry_positive']))),float(np.max(abs(gn-z['geometry_separation']))))
   if prev is None:ms=np.zeros(len(gp));cc=ms.copy();carry=ms.copy()
   else:
    T=z['H']@np.linalg.inv(prev['H'])
    transported=(T@np.column_stack([prev['mapped_world_xyz'],np.ones(len(gp))]).T).T[:,:3]
    eh=np.sum((z['mapped_world_xyz']-transported)**2,1);e0=np.sum((z['mapped_world_xyz']-prev['mapped_world_xyz'])**2,1)
    var=np.array([old[k]**2+prev[k]**2 for k in ['sigma_base_m','sigma_proxy_m']])
    ratio=(e0-eh)[None,:]/(2*var)
    ms=np.maximum(2*expit(ratio)-1,0).min(0)
    drift=np.sum((z['mapped_hand_xyz']-prev['mapped_hand_xyz'])**2,1)
    cc=np.exp(-drift/(2*var)).min(0)
    carry=np.minimum.reduce([Pprev,gp,cc])
   seed=np.minimum(gp,ms);P=np.maximum(seed,carry);N=gn;U=1-P-N
   for got,expected in [(z['positive'],P),(z['separation'],N),(z['unknown'],U),(z['carry'],carry),(z['motion_support'],ms)]:
    maxerr=max(maxerr,float(np.max(abs(got-expected))))
   assert maxerr<1e-8
   expectedcat=cat(P,N,U);expectedcat[old['excluded_mounting_nearest']]=3;assert np.array_equal(expectedcat,z['category'])
   # Changing visibility from visible to robot-occluded leaves the gate unchanged; a front conflict closes it.
   safe=np.isin(z['visibility'],[0,1])&~old['excluded_mounting_nearest']&~old['deep_penetration']
   assert not (P[bad]>0).any()
   green=expectedcat==0;occ=z['visibility']==1
   occluded_ids.append(set(np.flatnonzero(green&occ).tolist()));shown.append(set(np.flatnonzero(green).tolist()))
   rows.append({'case':ci,'frame':t,'robot_occluded_green':int((green&occ).sum()),'front_conflict_old_green':int(((z['baseline_category']==0)&(z['visibility']==3)).sum()),'other_occlusion_old_green':int(((z['baseline_category']==0)&(z['visibility']==2)).sum()),'scope_green':int((green&z['excluded_scope']).sum()),'support_preserved_by_visibility_semantics':int(safe.sum())})
   prev=z;Pprev=P
  retained=[len(shown[j-1]&occluded_ids[j]) for j in range(1,len(shown))]
  rows.append({'case':ci,'green_same_source_from_previous_to_robot_occlusion':retained,'independent_events':1})
  sigma=float(np.median(prev['sigma_proxy_m']))
  for gap in [0,3*sigma,10*sigma]:
   e=np.array([contrast(np.array([gap]),np.array([sigma]),m)[0] for m in mm])
   controltests.append({'gap_sigma':gap/sigma,'positive':float(np.maximum(e,0).min()),'negative':float(np.maximum(-e,0).min())})
  assert controltests[-1]['positive']==0 and controltests[-1]['negative']>.99
  rows.append({'case':ci,'geometry_controls':controltests})
 locks={}
 for fname in ['input_locked.json','source_locked.json']:
  lock=json.loads((O/fname).read_text());bad=[p for p,h in lock.items() if sha(p)!=h];assert not bad;locks[fname]={'files':len(lock),'mismatches':bad}
 oldlock=json.loads((B/'completed_locked.json').read_text());bad=[p for p,h in oldlock.items() if sha(p)!=h];assert not bad
 locks['old_stage3_v2']={'files':len(oldlock),'mismatches':bad}
 result={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),'all_passed':True,'independent_math_max_error':maxerr,'rows':rows,'locks':locks,'no_scoring_training':True,'no_new_mixture_fit':True,'limits':['same-transform mapped points are not independent contact truth','sparse endpoints cannot validate dense duration','hand-vs-static control is only transport evidence']}
 (O/'audits/independent.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
