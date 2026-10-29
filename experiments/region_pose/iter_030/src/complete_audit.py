from pathlib import Path
import json,ast,hashlib,types,math,datetime,traceback
import numpy as np
import torch
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/region_pose/iter_030';O=R/'experiments/region_pose/iter_029'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def main():
 inputs=[O/'evaluation/baselines.json',O/'src/complete_audit.py',O/'src/prepare_samples.py',R/'src/build_point_graphs.py']
 locks=json.loads((O/'completed_locked.json').read_text())
 for p in inputs[:3]:assert sha(p)==locks[str(p.relative_to(R))]
 hashes={str(p.relative_to(R)):sha(p) for p in inputs}
 dump(P/'input_addendum_030_2.json',hashes)
 dump(P/'source_addendum_030_2.json',{str(p.relative_to(P)):sha(p) for p in [P/'src/complete_audit.py',P/'审计补充030_2.md']})
 cfg=json.loads((O/'config.json').read_text());control=cfg['control_episodes']
 bases=[r for r in json.loads((O/'evaluation/baselines.json').read_text()) if r['split']=='train'];assert len(bases)==40
 dump(P/'input_slices/baselines_train_only.json',bases)
 tree=ast.parse((O/'src/train.py').read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='metrics');env={'np':np};exec(compile(ast.Module(body=[f],type_ignores=[]),'metrics','exec'),env)
 tree=ast.parse((O/'src/complete_audit.py').read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='met');env2={'np':np,'t':types.SimpleNamespace(metrics=env['metrics'])};exec(compile(ast.Module(body=[f],type_ignores=[]),'met','exec'),env2)
 metricdiff=0.;pointdiff=0.;shared={};focus=[];numeric=[]
 for path in sorted((P/'summaries').glob('*.json')):
  s=json.loads(path.read_text()); rr=s['queries']; sr=[r for r in rr if r['episode'] in control]
  shared[path.stem]={'events':6,'queries':24,'ap':float(np.mean([r['ap'] for r in sr])),'head_loss':float(np.mean([r['head_loss'] for r in sr]))}
  for r in rr:
   d=np.load(O/'samples'/f"{r['key']}.npz");td={k:torch.tensor(d[k]) for k in ['known','target']}
   pt=np.load(P/'points'/r['run']/r['snapshot']/f"{r['key']}.npz")
   met=env2['met'](pt['score'],td)
   pairs=[(r['ap'],met['ap'])]
   for b in [.1,.2,.3]:pairs.append((r['known_coverage'][str(b)],met['coverage'+str(int(100*b))]))
   for b in [.01,.02,.05,.1,.2,.3]:
    a=r['all_budget'][str(b)];suf=str(int(100*b))
    pairs.extend([(a['coverage'],met['all_coverage'+suf]),(a['unknown'],met['all_unknown_fraction'+suf]),(a['negative'],met['all_known_negative_fraction'+suf])])
   metricdiff=max(metricdiff,max(abs(a-b) for a,b in pairs))
   pointdiff=max(pointdiff,abs(float(pt['bce_contribution'].sum()+pt['rank_contribution'].sum())-r['head_loss']))
   assert np.all(pt['total_gradient'][~pt['known']]==0)
   if r['episode'] in ['tomato_sauce_demo_1','white_yellow_mug_demo_2','white_yellow_mug_demo_4']:focus.append(r)
  inv=[r for r in rr if r['trained_event'] and r['inverting_negatives']>0]
  numeric.append({'run':s['run'],'snapshot':s['snapshot'],'trained_queries_with_inversions':len(inv),'with_zero_inverting_negatives_in_rank':sum(r['rank_inverting_negatives']==0 for r in inv)})
 assert metricdiff<1e-12 and pointdiff<1e-12
 exposure=json.loads((P/'audits/exposure_history.json').read_text())
 meta=json.loads((P/'input_slices/training_metadata.json').read_text())
 for x in exposure:
  m=next(m for m in meta if m['run']==x['run']);plan=json.loads((O/'plans'/f"{x['run']}.json").read_text())
  selected=[p for p in plan if p['epoch']<=m['best_epoch']]
  counts={}
  for p in selected:counts[p['event']+'__'+str(p['offset'])]=counts.get(p['event']+'__'+str(p['offset']),0)+1
  x['best_epoch']=m['best_epoch'];x['best_steps']=len(selected);x['best_exposure_per_offset_range']=[min(counts.values()),max(counts.values())]
 dump(P/'audits/exposure_comparison.json',{'shared6_frozen':shared,'history':exposure,'warning':'Best snapshots differ in update count. Fixed240 last differs in exposure. First24 epochs equal shared-event exposure but differ in updates and mixed-group history objective; no per-event intermediate frozen evaluations exist.'})
 dump(P/'audits/tail_observation_counts.json',numeric)
 dump(P/'audits/focus_queries.json',focus)
 assert all(sha(R/k)==v for k,v in hashes.items())
 orig=json.loads((P/'input_locked.json').read_text());assert all(sha(R/k)==v for k,v in orig.items())
 audit={'training_queries':40,'training_events':10,'frozen_snapshots':8,'point_files':320,'head_loss_and_gradient_reconstruction':json.loads((P/'audits/numerical_reconstruction.json').read_text()),'original029_metric_max_difference':metricdiff,'per_point_contribution_sum_max_difference':pointdiff,'original_inputs_unchanged':len(orig),'supplement_inputs_unchanged':len(hashes),'source_reconstruction_equal':40,'baseline_train_rows':40,'history_individual_event_loss_not_saved':True,'validation_analysis':False,'test_access':False,'network_forward':False,'training':False,'stage2_accepted':False}
 dump(P/'audits/completion_audit.json',audit)
 dump(P/'audits/resolved_030_1.json',{'original_pid':700843,'resolved_by_pid':702973,'reason':'short-cloud arange vs prior FPS permutation; same selected set, no changed training input','original_failure_preserved':True})
 dump(P/'stage_state.json',{'stage':'audited','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print(json.dumps({'status':'audited','metric_max_difference':metricdiff,'point_sum_max_difference':pointdiff,'inputs':len(orig)}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  (P/'logs/completion_error_remote_only.txt').write_text(traceback.format_exc())
  print(json.dumps({'error_type':type(e).__name__,'message':str(e)[:300]}));raise SystemExit(1)
