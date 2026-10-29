from pathlib import Path
import ast,copy,json,hashlib,numpy as np
P=Path('/mnt/nvme1/drh/D_VLA');O=P/'experiments/region_pose/iter_022'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
for d in ['seeds','train/filter_masks']: (O/d).mkdir(exist_ok=True)
original=P/'experiments/region_pose/iter_011/src/extend_cotracker.py'
source=original.read_text()
(O/'src/original_011.py').write_text(source,encoding='utf-8')
t=ast.parse(source)
main=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='main')
# Bind only the original training cases; no reconstruction from other splits.
main.body=[n for n in main.body if not (isinstance(n,ast.For) and n.lineno==44)]
for i,n in enumerate(main.body):
 if isinstance(n,ast.Assign) and any(getattr(a,'id','')=='cases' for a in n.targets) and n.lineno==43:
  main.body[i]=ast.parse("cases=json.loads((BASE/'cases_locked.json').read_text())").body[0]
loop=next(n for n in main.body if isinstance(n,ast.For) and getattr(n.target,'id','')=='case')
loop.body=[n for n in loop.body if not(isinstance(n,ast.If) and n.lineno==70)]
metadata=[]
class Modify(ast.NodeTransformer):
 def visit_Call(self,n):
  n=self.generic_visit(n)
  if isinstance(n.func,ast.Attribute) and n.func.attr=='goodFeaturesToTrack':
   return ast.copy_location(ast.Call(func=ast.Name(id='augment_features',ctx=ast.Load()),args=[ast.Name(id='ident',ctx=ast.Load())]+n.args,keywords=n.keywords),n)
  return n
 def visit_Assign(self,n):
  n=self.generic_visit(n)
  names=[getattr(a,'id','') for a in n.targets]
  if n.lineno in [97,104,105] and any(k in names for k in ['ok','endpoint','corr']):
   stage=next(k for k in ['ok','endpoint','corr'] if k in names)
   metadata.append({'stage':stage,'original_expression_remote_only':ast.unparse(n.value)})
   extra=ast.parse("capture_stage(ident,frame,"+repr(stage)+",locals())").body[0]
   return [n,ast.copy_location(extra,n)]
  return n
t=Modify().visit(t)
# Keep original numerical quality functions byte-equivalent at AST level.
original_ast=ast.parse(source)
for name in ['rigid','apply','estimate','patchcorr']:
 a=next(n for n in original_ast.body if isinstance(n,ast.FunctionDef) and n.name==name)
 b=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
# Helpers import after existing future/import declarations (original has no future imports).
t.body.insert(0,ast.ImportFrom(module='sampling_helpers',names=[ast.alias(name='augment_features'),ast.alias(name='capture_stage')],level=0))
code=ast.unparse(ast.fix_missing_locations(t))
(O/'src/extend_cotracker_grid.py').write_text(code+'\n',encoding='utf-8')
save(O/'audits/transformation.json',{'original_sha256':sha(original),'quality_function_AST_unchanged':True,
 'changes':['fixed_011_train_cases','separate_BASE_output','disable_iter010_reuse','augment_initial_points','capture_filter_arrays'],
 'filter_expressions_remote_only':metadata,'source_body_not_exported':True})
# Existing global BASE follows this new script's own directory, so output stays in 022.
cases=json.loads((P/'experiments/region_pose/iter_011/train/summary.json').read_text())['cases']
assert all(r['split']=='train' for r in cases)
save(O/'cases_locked.json',cases)
oldlock=json.loads((P/'experiments/region_pose/iter_021/completed_locked.json').read_text())
assert all(sha(P/k)==v for k,v in oldlock.items())
prefix_checks=[];inputs={original, P/'experiments/region_pose/iter_011/train/summary.json',P/'experiments/region_pose/iter_011/train/locked.json'}
for r in cases:
 ep=r['id'];f=P/'experiments/region_pose/iter_021/cases'/(ep+'.npz');inputs.add(f)
 with np.load(f) as z:
  mask=z['mask_after_erosion'].copy();old=z['initial_uv'].copy()
 uv=[p.copy() for p in old]
 v,u=np.where(mask>0)
 cells=sorted(set(zip((v//8).tolist(),(u//8).tolist())))
 for cy,cx in cells:
  ix=np.flatnonzero((v//8==cy)&(u//8==cx))
  dd=(u[ix]-(cx*8+3.5))**2+(v[ix]-(cy*8+3.5))**2
  ix=ix[np.lexsort((u[ix],v[ix],dd))][0]
  pt=np.array([u[ix],v[ix]],np.float32)
  if uv and np.min(np.linalg.norm(np.asarray(uv)-pt,axis=1))<3:continue
  if len(uv)>=300:break
  uv.append(pt)
 uv=np.asarray(uv,np.float32).reshape(-1,2)
 assert np.array_equal(uv[:len(old)],old)
 if len(uv)>len(old):assert np.all(mask[uv[len(old):,1].astype(int),uv[len(old):,0].astype(int)]>0)
 np.savez_compressed(O/'seeds'/(ep+'.npz'),mask=mask,original_uv=old,initial_uv=uv,source_is_grid=np.arange(len(uv))>=len(old))
 prefix_checks.append({'episode':ep,'query':r['query'],'original':len(old),'added':len(uv)-len(old),'total':len(uv),'eligible':len(uv)>=8,'prefix_exact':True})
weight=P/'experiments/region_pose/iter_010/weights/scaled_offline.pth'
assert sha(weight)=='2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834'
inputs.add(weight)
# Include only prior training lock entries, including reused 010 training arrays stored under 011.
L=P/'experiments/region_pose/iter_011/train'
for k,h in json.loads((L/'locked.json').read_text()).items():
 f=L/k
 assert sha(f)==h
 inputs.add(f)
# Record underlying predetermined train episode files, no heldout lookup.
protocol=json.loads((P/'experiments/region_pose/iter_003/evidence/base/protocol.json').read_text())
refs={r['id']:r for r in protocol['episodes'] if r['split']=='train'}
for r in cases:
 ref=refs[r['id']]
 for k in ['component','rgbd']: inputs.add(Path(ref[k]))
 inputs.add(P/('experiments/region_pose/iter_003/evidence/motion/'+r['id']+'.npz'))
save(O/'input_locked.json',{str(f):sha(f) for f in sorted(inputs)})
save(O/'source_locked.json',{str(f):sha(f) for f in [O/'设计方案.md',O/'config.json',O/'cases_locked.json',*sorted((O/'src').glob('*.py'))]})
save(O/'audits/preflight.json',{'passed':True,'cases':prefix_checks,'old_021_lock_checked':len(oldlock),
 'quality_functions_unchanged':True,'old_011_train_lock_checked':True,'weight_hash_passed':True,'no_validation_test_access':True})
print(json.dumps(prefix_checks,ensure_ascii=False))
