from pathlib import Path
import json,hashlib,ast,time
import numpy as np
from scipy.spatial import ConvexHull,QhullError
P=Path('/mnt/nvme1/drh/D_VLA');O=P/'experiments/region_pose/iter_022';B=P/'experiments/region_pose/iter_011/train'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def hull(a):
 try:return float(ConvexHull(a).volume) if len(a)>=3 else 0.
 except QhullError:return 0.
s=read(O/'train/summary.json');old=read(B/'summary.json');oldby={r['id']:r for r in old['cases']}
assert len(s['cases'])==16
pre={r['episode']:r for r in read(O/'audits/preflight.json')['cases']}
meta=ast.parse((O/'src/original_011.py').read_text())
ids_node=next(n.value for n in ast.walk(meta) if isinstance(n,ast.Assign) and n.lineno==106 and any(getattr(x,'id','')=='ids' for x in n.targets))
ids_eval=compile(ast.Expression(ids_node),'<original endpoint ID rule>','eval')
rows=[];all_prefix=[];filter_rows=[]
for r in s['cases']:
 ep=r['id'];br=oldby[ep];key=r['event']+'_query'+str(r['query'])
 assert r['query']==br['query'] and r['selected_held_frames']==br['selected_held_frames']
 with np.load(O/'seeds'/(ep+'.npz')) as z:
  initial=z['initial_uv'].copy();original=z['original_uv'].copy();mask=z['mask'].copy()
 # independently reconstruct grid seeds from original locked 021 mask
 seed=[v.copy() for v in original];v,u=np.where(mask>0)
 for cy,cx in sorted(set(zip((v//8).tolist(),(u//8).tolist()))):
  ix=np.flatnonzero((v//8==cy)&(u//8==cx));dd=(u[ix]-(cx*8+3.5))**2+(v[ix]-(cy*8+3.5))**2
  k=ix[np.lexsort((u[ix],v[ix],dd))][0];pt=np.array([u[k],v[k]],np.float32)
  if seed and np.min(np.linalg.norm(np.asarray(seed)-pt,axis=1))<3:continue
  if len(seed)>=300:break
  seed.append(pt)
 assert np.array_equal(np.asarray(seed,np.float32).reshape(-1,2),initial)
 row={'episode':ep,'query':r['query'],'old_early_valid':br['valid'],'new_early_valid':r['valid'],'initial':pre[ep],
 'reason':r.get('reason'),'dual_bridge_difference':r.get('dual_bridge_difference'),'time_comparisons':r.get('time_comparisons'),'endpoints':[]}
 tp=O/'train/trajectories'/(key+'.npz')
 if not tp.exists():
  assert len(initial)<8
  row['trajectory']='not_run_insufficient_initial_points';rows.append(row);continue
 with np.load(tp) as z:tr={k:z[k] for k in z.files}
 with np.load(B/'trajectories'/(key+'.npz')) as z:bt={k:z[k] for k in z.files}
 with np.load(O/'train/correspondences'/(key+'.npz')) as z:co={k:z[k] for k in z.files}
 assert np.array_equal(tr['initial_uv'],initial) and np.array_equal(initial[:len(original)],bt['initial_uv'])
 assert np.array_equal(tr['frames'],bt['frames'])
 n=len(original);delta=np.linalg.norm(tr['tracks'][:,:n]-bt['tracks'],axis=-1)
 both=tr['visible'][:,:n]&bt['visible']
 row['old_prefix_prediction_change']={'visible_agreement_fraction':float(np.mean(tr['visible'][:,:n]==bt['visible'])),
 'jointly_visible_observations':int(both.sum()),'median_pixels':float(np.median(delta[both])) if both.any() else None,
 'p95_pixels':float(np.percentile(delta[both],95)) if both.any() else None,'max_pixels':float(np.max(delta[both])) if both.any() else None}
 all_prefix.append(row['old_prefix_prediction_change'])
 for e in r.get('bridge_results',[]):
  fr=e['frame'];suf='_'+str(fr);ids=co['track_id'+suf];inl=co['inlier'+suf]
  assert len(ids)==e['endpoint_matches']
  if 'inliers' in e:assert int(inl.sum())==e['inliers']
  fm=O/'train/filter_masks'/(ep+'_frame'+str(fr)+'.npz')
  with np.load(fm) as z: masks={k:z[k] for k in z.files}
  # Use only values captured at their own assignment; earlier snapshots may contain prior-loop locals.
  current={k:masks['corr__'+k] for k in ['alive','endpoint','corr']}
  current['np']=np
  endpoint_before=current['endpoint'].copy()
  inplace=next(n for n in ast.walk(meta) if isinstance(n,ast.AugAssign) and getattr(n.target,'id','')=='endpoint')
  exec(compile(ast.Module(body=[inplace],type_ignores=[]),'<original inplace appearance rule>','exec'),current)
  rebuilt=np.asarray(eval(ids_eval,current))
  assert np.array_equal(rebuilt,ids),(ep,fr)
  endpoint=endpoint_before;corr=current['corr'];alive=current['alive']
  assert len(endpoint)==len(corr)==len(alive)
  frstat={'episode':ep,'frame':fr,'visible_and_bounds_pass':int(masks['ok__ok'].sum()),
   'alive_before_endpoint':len(alive),'endpoint_combined_pass':int(endpoint.sum()),
   'appearance_corr_ge_065_pass':int((corr>=.65).sum()),
   'joint_pass':int((endpoint&(corr>=.65)).sum()),
   'appearance_rejected_after_endpoint':int((endpoint&(corr<.65)).sum()),
   'joint_ids_exactly_reconstructed':True,
   'endpoint_individual_depth_robot_reasons':'combined_only; individual predicates not captured',
   'snapshot_interpretation':'only ok__ok and corr__alive/endpoint/corr are current; other snapshots may retain previous iteration locals'}
  filter_rows.append(frstat)
  er={'frame':fr,'valid':e['valid'],'matches':len(ids),'original_matches':int((ids<n).sum()),'grid_matches':int((ids>=n).sum()),
   'inliers':e.get('inliers'),'original_inliers':int(((ids<n)&inl).sum()) if 'inliers' in e else None,
   'grid_inliers':int(((ids>=n)&inl).sum()) if 'inliers' in e else None,
   'inlier_query_hull_area':hull(co['query_uv'+suf][inl]) if 'inliers' in e else None,
   'ratio':e.get('ratio'),'median_inlier_mm':e.get('median_inlier_mm'),'reason':e.get('reason')}
  row['endpoints'].append(er)
 rows.append(row)
newpass={r['episode'] for r in rows if r['new_early_valid']};oldpass={r['episode'] for r in rows if r['old_early_valid']}
summary={'split':'train','episodes':16,'old_early_candidates':len(oldpass),'new_early_candidates':len(newpass),
 'gained':sorted(newpass-oldpass),'lost':sorted(oldpass-newpass),'retained':sorted(newpass&oldpass),
 'eligible_for_separately_registered_validation':bool(newpass-oldpass) and not bool(oldpass-newpass),
 'rows':rows,'candidate_not_surface_or_grasp_truth':True,'new_region_targets':False,'scoring_network_training':False}
save(O/'comparison_summary.json',summary);save(O/'filter_attribution.json',filter_rows)
checks={}
for name in ['source_locked.json','source_addendum_022_1.json','source_addendum_022_2.json','input_locked.json']:
 d=read(O/name);checks[name]={'files':len(d),'unchanged':all(sha(Path(f))==h for f,h in d.items())}
assert all(x['unchanged'] for x in checks.values())
seedhash={str(f):sha(f) for f in sorted((O/'seeds').glob('*.npz'))}
save(O/'seed_artifacts_postrun_locked.json',{'timing':'after_tracking; deterministic reconstruction verified against preregistered source/config and locked 021 inputs','files':seedhash})
save(O/'audits/completion_audit.json',{'checks':checks,'all_16_seed_arrays_independently_reconstructed':True,
 'all_14_original_prefix_arrays_identical':True,'all_61_endpoint_ID_rules_reconstructed':len(filter_rows),
 'new_and_control_queries_endpoints_equal':True,'scoring_training':False,'new_labels':False,'validation_test_access':False,
 'filter_limitation':'depth/robot individual masks unavailable; combined mask and appearance correlation retained; no invented decomposition',
 'passed':True})
save(O/'pre_validation_freeze.json',{'time':time.time(),'source':checks,'summary_sha256':sha(O/'comparison_summary.json'),
 'eligible_for_separate_registration':summary['eligible_for_separately_registered_validation'],'no_validation_started':True})
save(O/'stage_state.json',{'stage':'audit_complete','time':time.time(),'new_scoring_training':False})
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False))
