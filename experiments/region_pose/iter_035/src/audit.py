import os,sys,json,hashlib,ast,traceback
from pathlib import Path
import numpy as np,h5py
B=Path('/mnt/nvme1/drh/D_VLA');E=B/'experiments/region_pose';R=E/'iter_035'
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for a in iter(lambda:f.read(8388608),b''):h.update(a)
 return h.hexdigest()
def apply(p,T):return p@T[:3,:3].T+T[:3,3]
def rigid(a,b):
 ac=a.mean(0);bc=b.mean(0);u,s,vt=np.linalg.svd((a-ac).T@(b-bc)); rot=vt.T@u.T
 if np.linalg.det(rot)<0:vt[-1]*=-1;rot=vt.T@u.T
 T=np.eye(4);T[:3,:3]=rot;T[:3,3]=bc-rot@ac;return T
def cloud_at(g,t,uv):
 dep=g['depth_m'][t];K=g['K'][t];T=g['camera_to_world'][t]; u,v=uv[:,0],uv[:,1];d=dep[v,u]
 return np.column_stack(((u-K[0,2])*d/K[0,0],(v-K[1,2])*d/K[1,1],d))@T[:3,:3].T+T[:3,3]
def main():
 ident='white_yellow_mug_demo_6';conf=json.loads((R/'config.json').read_text());endpoints=conf['endpoints']
 rec=json.loads(next((R/'train/audits').glob('*.json')).read_text());z=np.load(next((R/'train/correspondences').glob('*.npz')));tr=np.load(next((R/'train/trajectories').glob('*.npz')))
 seed=np.load(R/'seeds'/(ident+'.npz'));assert np.array_equal(seed['initial_uv'],tr['initial_uv'])
 assert np.array_equal(tr['frames'],np.arange(36,83))
 tree=ast.parse((R/'src/track.py').read_text());env={'np':np}
 fs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['rigid','apply','estimate','patchcorr']]
 exec(compile(ast.fix_missing_locations(ast.Module(body=fs,type_ignores=[])),'frozen_quality','exec'),env)
 H=np.load(E/'iter_003/evidence/motion'/(ident+'.npz'))['hand_to_world']
 ref=json.loads((R/'refs.json').read_text())[ident]
 rows=[];maxxyz=0.;maxtransform=0.;maxmetric=0.;slices={}
 with h5py.File(ref['rgbd'],'r') as f:
  g=f['agentview'];query_uv=np.rint(tr['initial_uv']).astype(int);source=cloud_at(g,36,query_uv)
  for t in [36]+endpoints:
   for n in ['rgb','depth_m','K','camera_to_world']:
    ar=g[n][t];slices[f'{t}/{n}']={'shape':list(ar.shape),'sha256':hashlib.sha256(ar.tobytes()).hexdigest()}
  clip=g['rgb'][36:83];slices['clip36_82']={'shape':list(clip.shape),'sha256':hashlib.sha256(clip.tobytes()).hexdigest()}
  for t,row in zip(endpoints,rec['bridge_results']):
   suffix=str(t);assert row['frame']==t
   ids=z['track_id_'+suffix];inlier=z['inlier_'+suffix]
   fm=np.load(R/'train/filter_masks'/f'{ident}_frame{t}.npz')
   final=fm['endpoint__endpoint'].copy() & (fm['corr__corr']>=.65)
   assert np.array_equal(fm['endpoint__alive'][final],ids),'filtered IDs mismatch'
   assert np.array_equal(tr['tracks'][t-36,ids],z['bridge_uv_'+suffix])
   xy=np.rint(z['bridge_uv_'+suffix]).astype(int);b=cloud_at(g,t,xy);a=source[ids]
   maxxyz=max(maxxyz,float(np.max(np.abs(b-z['bridge_xyz_'+suffix]))),float(np.max(np.abs(a-z['query_xyz_'+suffix]))))
   T,ii,d=env['estimate'](a,b)
   assert np.array_equal(ii,inlier)
   assert d['valid']==row['valid']
   maxmetric=max(maxmetric,abs(d['median_inlier_mm']-row['median_inlier_mm']),abs(d['ratio']-row['ratio']))
   independent=rigid(a[ii],b[ii])
   maxtransform=max(maxtransform,float(np.max(np.abs(T-independent))))
   if T is not None:
    composed=np.linalg.inv(H[t])@T
    maxtransform=max(maxtransform,float(np.max(np.abs(composed-np.array(row['query_world_to_anchor_hand'])))))
   hp=apply(b,np.linalg.inv(H[t]))
   assert np.allclose(hp,z['observed_hand_xyz_'+suffix],rtol=0,atol=1e-12)
   rows.append({'frame':t,'matches':len(ids),'inliers':int(ii.sum()),'original_inliers':int((ids[ii]<len(seed['original_uv'])).sum()),'grid_inliers':int((ids[ii]>=len(seed['original_uv'])).sum()),'valid':row['valid'],'ratio':d['ratio'],'median_inlier_mm':d['median_inlier_mm']})
 assert maxxyz<1e-7 and maxtransform<1e-7 and maxmetric<1e-4
 comparisons=[];supported=[55] if rows[0]['valid'] else [];stopped=None
 for c in rec['time_comparisons']:
  f0,f1=c['frames'];a0=next(a for a in rec['bridge_results'] if a['frame']==f0);a1=next(a for a in rec['bridge_results'] if a['frame']==f1)
  if a0['valid'] and a1['valid']:
   delta=np.linalg.norm(apply(z['query_foreground_xyz'],np.array(a0['query_world_to_anchor_hand']))-apply(z['query_foreground_xyz'],np.array(a1['query_world_to_anchor_hand'])),axis=1)*1000
   id0=z[f'track_id_{f0}'][z[f'inlier_{f0}']];id1=z[f'track_id_{f1}'][z[f'inlier_{f1}']]
   common,i0,i1=np.intersect1d(id0,id1,return_indices=True)
   hd=np.linalg.norm(z[f'observed_hand_xyz_{f0}'][z[f'inlier_{f0}']][i0]-z[f'observed_hand_xyz_{f1}'][z[f'inlier_{f1}']][i1],axis=1)*1000
   valid=len(common)>=8 and np.median(delta)<=3 and np.percentile(delta,95)<=6 and np.median(hd)<=3 and np.percentile(hd,95)<=6
   for k,v in [('median_mm',np.median(delta)),('p95_mm',np.percentile(delta,95)),('observed_median_mm',np.median(hd)),('observed_p95_mm',np.percentile(hd,95))]:assert abs(c[k]-v)<1e-9
  else:valid=False
  assert bool(valid)==c['valid']
  if stopped is None and valid:supported.append(f1)
  elif stopped is None:stopped=f1
  comparisons.append(c)
 assert supported==[55,57,60] and stopped==65
 checks={}
 for name in ['input_locked.json','source_locked.json','source_runner_addendum.json','seed_locked.json']:
  dd=json.loads((R/name).read_text())
  for p,h in dd.items():assert sha(Path(p))==h
  checks[name]=len(dd)
 # Main method unchanged except preregistered path/case/weight adapters, tested by reverse transformation.
 original=ast.parse((E/'iter_022/src/extend_cotracker_grid_v3.py').read_text())
 expected=['rigid','apply','estimate','patchcorr']
 for n in expected:
  one=next(x for x in original.body if isinstance(x,ast.FunctionDef) and x.name==n)
  two=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==n)
  assert ast.dump(one,include_attributes=False)==ast.dump(two,include_attributes=False)
 save(R/'input_used_slices_locked.json',slices)
 summary={'episode':ident,'query':36,'initial_points':91,'original_points':21,'grid_points':70,'frames_tracked':47,'endpoints':rows,'comparisons':comparisons,'supported_predefined_endpoints':supported,'first_failure_endpoint':stopped,'later82_diagnostic_only':True,'new_labels':False,'training':False,'validation_read':False,'test_read':False,'stage2_approved':False}
 save(R/'summary.json',summary)
 save(R/'audits/completion.json',{'checks':checks,'source_xyz_max_abs_m':maxxyz,'transform_max_abs':maxtransform,'quality_metric_max_abs':maxmetric,'filtered_ids_equal':True,'trajectory_and_seed_equal':True,'frozen_functions_equal':True,'comparisons_independently_equal':True,'supported_endpoints':supported,'first_failure':stopped,'slice_hashes':len(slices),'slice_hash_timing':'post_read; full source files locked before forward'})
 print(json.dumps({'supported_endpoints':supported,'first_failure':stopped,'xyz_error':maxxyz,'transform_error':maxtransform,'rows':rows}))
try:main()
except Exception as e:
 (R/'audit_traceback.txt').write_text(traceback.format_exc());save(R/'audit_failure.json',{'type':type(e).__name__,'message':str(e)});print(json.dumps({'error_type':type(e).__name__,'message':str(e)}));raise SystemExit(1)
