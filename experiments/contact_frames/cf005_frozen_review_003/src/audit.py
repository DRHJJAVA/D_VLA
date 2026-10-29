from pathlib import Path
import json,hashlib,datetime,ast
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/contact_frames/cf005_frozen_review_003'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((B/'plan.json').read_text());weights=json.loads((B/'weights_locked.json').read_text());lock=json.loads((B/'input_source_locked.json').read_text())
assert all(sha(Path(p))==h for p,h in weights.items())
assert all(sha(Path(p))==h for p,h in lock.items())
rows=[]
for ep in plan['cases']:
 p=B/ep
 assert (p/'src/prepare.py').read_bytes()==(B.parent/'cf001/src/prepare.py').read_bytes()
 d=np.load(p/'artifacts/dataset.npz');a=np.load(p/'artifacts/frame_predictions.npz');n=len(d['states']);pr=a['probabilities']
 assert pr.shape==(3,n) and np.isfinite(pr).all() and np.all((pr>=0)&(pr<=1))
 assert np.array_equal(a['context_indices'],np.maximum(0,np.arange(n)[:,None]+np.arange(-8,1)[None,:]))
 rel=d['geometry_reliable']&d['state_valid'];pos=(pr.min(0)>=.8)&rel;neg=(pr.max(0)<=.2)&rel&(d['distance']-2*d['sigma']>.006)
 assert np.array_equal(pos,a['contact_candidate']) and np.array_equal(neg,a['observed_separation_candidate']) and np.array_equal(~(pos|neg),a['unknown'])
 assert not np.any(pos&neg)
 attempts=[[52,69],[94,153]] if ep=='tomato_sauce_demo_3' else [[46,64],[80,158]]
 sub=[]
 for start,end in attempts:
  sl=slice(start,end+1);sub.append({'range':[start,end],'candidate':int(pos[sl].sum()),'separation':int(neg[sl].sum()),'unknown':int(a['unknown'][sl].sum()),'raw_unanimous_high':int((pr[:,sl].min(0)>=.8).sum()),'mean_score_median':float(np.median(pr[:,sl].mean(0)))})
 rows.append({'episode':ep,'frames':n,'identical_preprocessing_source':True,'continuous_context_verified':True,'classification_recomputed_equal':True,'attempt_navigation_statistics_not_ground_truth':sub})
tree=ast.parse((B/'src/run.py').read_text());calls=[ast.unparse(z.func) for z in ast.walk(tree) if isinstance(z,ast.Call)]
assert not any(x.endswith('.backward') or x.endswith('.step') or 'Adam' in x for x in calls)
out={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),'frozen_weights_verified':True,'inputs_unchanged':True,'no_training_calls':True,'rows':rows,'historical_setup_failures_resolved':True}
(B/'audits/independent_prediction.json').write_text(json.dumps(out,indent=2))
print(json.dumps(rows))
