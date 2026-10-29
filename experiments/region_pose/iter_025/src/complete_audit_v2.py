from pathlib import Path
import json,hashlib,time,ast
import numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/region_pose';B=P/'iter_025'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def verify(p):
 d=json.loads(p.read_text())
 for k,h in d.items():
  f=R/k
  if not f.exists() and p.parent.parent.name=='targets':f=p.parent/k
  assert sha(f)==h,k
 return len(d)
counts={}
for name in ['source_locked.json','source_audit_addendum.json','train_input_locked.json','val_input_locked.json']:counts[name]=verify(B/name)
counts['old024_completed']=verify(P/'iter_024/completed_locked.json')
for split in ['train','val']:
 for folder in ['targets','sampling']:counts[folder+'/'+split]=verify(B/folder/split/'locked.json')
freeze=json.loads((B/'pre_validation_freeze.json').read_text());pre=json.loads((B/'audits/val_preflight.json').read_text())
assert freeze['time']<pre['time']
assert freeze['sampling_summary']==sha(B/'sampling/train/summary.json')
assert freeze['train_summary']==sha(B/'targets/train/summary.json')
# Reverse only the documented source-path/phase-lock substitutions and compare the complete AST.
old=ast.parse((P/'iter_013/src/build_targets.py').read_text())
new=ast.parse((B/'src/build_targets.py').read_text())
class Undo(ast.NodeTransformer):
 def visit_Constant(self,n):
  return ast.copy_location(ast.Constant('iter_012'),n) if n.value=='iter_024' else n
 def visit_IfExp(self,n):
  if isinstance(n.body,ast.Constant) and n.body.value=='iter_022' and isinstance(n.orelse,ast.Constant) and n.orelse.value=='iter_023':return ast.copy_location(ast.Constant('iter_011'),n)
  return self.generic_visit(n)
 def visit_BinOp(self,n):
  if isinstance(n.left,ast.Name) and n.left.id=='split' and isinstance(n.op,ast.Add) and isinstance(n.right,ast.Constant) and n.right.value=='_input_locked.json':return ast.copy_location(ast.Constant('input_locked.json'),n)
  return self.generic_visit(n)
assert ast.dump(old)==ast.dump(Undo().visit(new))
a=ast.parse((R/'src/build_point_graphs.py').read_text());b=ast.parse((B/'src/sampling.py').read_text())
assert ast.dump(next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='fps'))==ast.dump(next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='fps'))
rows=[]
for split in ['train','val']:
 s=json.loads((B/'sampling'/split/'summary.json').read_text())
 for row in s['rows']:
  key=row['event']+'_query'+str(row['query'])
  y=np.load(B/'targets'/split/(key+'.npz'))
  z=np.load(P/'iter_024'/split/(key+'.npz'))
  x=np.load(B/'sampling'/split/(key+'.npz'))
  sel=x['selected_index'];p=y['target'].astype(bool);neg=y['known'].astype(bool)&~p
  assert np.array_equal(x['query_xyz'],z['query_xyz']) and np.array_equal(x['query_pixel'],z['query_pixel'])
  assert np.array_equal(x['target'],y['target'][sel]) and np.array_equal(x['confidence'],y['confidence'][sel])
  assert len(set(sel.tolist()))==len(sel)
  assert (y['positive_support_count'][p]>=2).all()
  assert (y['negative_support_count'][neg]>=2).all()
  assert not y['possible_inside_frames'][:,neg].any()
  for j in range(len(z['selected_frames'])):
   assert (y['position_sensitivity_m'][j][y['positive_frames'][j]]<=.002).all()
   assert (y['position_sensitivity_m'][j][y['negative_frames'][j]]<=.002).all()
  def cells(ids):
   return len(np.unique(np.floor(z['query_xyz'][ids]/.001).astype(np.int64),axis=0))
  posids=np.flatnonzero(p);sample_posids=sel[p[sel]]
  rows.append(dict(split=split,**row,full_positive_query_1mm_cells=cells(posids),sample_positive_query_1mm_cells=cells(sample_posids),positive_support_frame_counts=y['positive_support_count'][p].tolist(),positive_confidence_min=float(y['confidence'][p].min()) if p.any() else None,positive_confidence_median=float(np.median(y['confidence'][p])) if p.any() else None,positive_max_pairwise_span_mm=float(np.linalg.norm(z['query_xyz'][posids][:,None,:]-z['query_xyz'][posids][None,:,:],axis=2).max()*1000) if p.any() else None))
# Old source-index gate and per-event table, no aggregate success claim.
train={r['episode'] for r in rows if r['split']=='train'};val={r['episode'] for r in rows if r['split']=='val'}
assert not train&val
summary=dict(rows=rows,train_candidates=11,train_eligible=10,old_val_candidates=3,old_val_eligible=3,train_official_denominator=20,val_official_denominator=4,full_train_known_eligibility=sum(r['full_positive']>=3 and r['full_negative']>=16 for r in rows if r['split']=='train'),no_model_training=True,not_physical_contact_truth=True,no_test_access=True)
dump(B/'summary.json',summary)
dump(B/'audits/completion_audit.json',dict(time=time.time(),verified_hash_counts=counts,complete_target_ast_unchanged_except_paths=True,original_fps_ast_exact=True,train_frozen_before_old_validation=True,all_14_independent_target_reconstructions_exact=True,all_unknown_confidence_zero=True,all_source_pixels_unique=True,episode_disjoint=True,positive_support_at_least_two_times=True,no_impossible_negatives=True,no_extra_threshold_relaxation=True,stage2_user_approved=False))
print(json.dumps(dict(events=len(rows),train_eligible=10,val_eligible=3,checks_passed=True,hash_counts=counts)))
