import json,hashlib,ast,subprocess,sys,datetime,traceback
from pathlib import Path
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';D=B/'iter_041';E=D/'full_train_new15'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def state(s,**kw):save(D/'stage_state.json',{'stage':s,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw})
try:
 state('preparing_full_train_inputs',next='hash new training inputs then extract actual finite-volume observations')
 E.mkdir();(E/'cases').mkdir();(E/'audits').mkdir();(E/'src').mkdir()
 oldcases=json.loads((B/'iter_033/cases.json').read_text());prior={x['id'] for x in oldcases}
 cases=[];train=[]
 for p in sorted((B/'iter_003/evidence/motion').glob('*.json')):
  obj=json.loads(p.read_text())
  if not isinstance(obj,dict) or obj.get('split')!='train' or 'frame_diagnostics' not in obj:continue
  name=obj['id'];train.append(name)
  if name in prior:continue
  asset,num=name.rsplit('_demo_',1)
  options=[P/f'data/v3/trainval_rgbd_448/{asset}/demo_{num}.h5',P/f'data/v2/stage1_rgbd/{asset}/demo_{num}.h5']
  present=[x for x in options if x.is_file()]
  assert present
  # Same preference as existing replay data organization; actual source recorded and narrow-point equality required.
  cases.append({'id':name,'split':'train','rgbd':str(present[0]),'frames':obj['frames']})
 assert len(train)==20 and len(cases)==15 and prior.issubset(set(train))
 save(E/'cases.json',cases);save(D/'training_scope.json',{'all_train':train,'reuse_033':sorted(prior),'extract_new':[x['id'] for x in cases],'validation':[],'test':[]})
 original=B/'iter_033/src/compare_sources_v3.py'
 tree=ast.parse(original.read_text())
 class Adapter(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value=='iter_033':return ast.copy_location(ast.Constant(value='iter_041/full_train_new15'),n)
   if n.value=='5 training episodes, 6 attempts; frames not independent grasps':return ast.copy_location(ast.Constant(value='15 training episodes, 15 attempts; frames not independent grasps'),n)
   return n
 tree=Adapter().visit(tree);ast.fix_missing_locations(tree)
 target=E/'src/extract_full_training.py';target.write_text(ast.unparse(tree)+'\n',encoding='utf-8')
 # Verify reverse transformation restores exact AST, excluding formatting.
 reverse=ast.parse(target.read_text())
 class Reverse(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value=='iter_041/full_train_new15':return ast.copy_location(ast.Constant(value='iter_033'),n)
   if n.value=='15 training episodes, 15 attempts; frames not independent grasps':return ast.copy_location(ast.Constant(value='5 training episodes, 6 attempts; frames not independent grasps'),n)
   return n
 assert ast.dump(Reverse().visit(reverse),include_attributes=False)==ast.dump(ast.parse(original.read_text()),include_attributes=False)
 save(E/'audits/method_adapter.json',{'original':str(original.relative_to(P)),'original_sha256':sha(original),'adapter_sha256':sha(target),'reverse_ast_equal':True,'only_output_directory_and_summary_unit_changed':True})
 inputs=json.loads((B/'iter_033/input_locked.json').read_text())
 # Old input hashes remain authoritative; never replace them with current values to hide changes.
 for path,h in inputs.items():assert sha(path)==h
 for case in cases:
  for p in [Path(case['rgbd']),B/f"iter_003/evidence/motion/{case['id']}.json",B/f"iter_003/evidence/motion/{case['id']}.npz"]:
   assert p.is_file();inputs[str(p)]=sha(p)
 inputs[str(E/'cases.json')]=sha(E/'cases.json')
 save(E/'input_locked.json',inputs)
 for f in ['source_locked.json','source_addendum_033_1.json','source_addendum_033_2.json']:
  source=json.loads((B/'iter_033'/f).read_text())
  for path,h in source.items():assert sha(path if Path(path).is_absolute() else P/path)==h
  if f=='source_locked.json':
   for p in [target,D/'设计方案.md',D/'config.json',D/'src/run_full_train.py',D/'src/apply_full_train.py']:
    source[str(p)]=sha(p)
  save(E/f,source)
 state('extracting_full_train_15',next='apply new rule to all 20 training episodes after extraction; old validation remains unread')
 with (E/'extract.log').open('w') as log:
  result=subprocess.run([sys.executable,str(target)],cwd=P,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError('full_train_extraction_failed')
 state('applying_full_train_rule')
 with (D/'full_train_rule.log').open('w') as log:
  result=subprocess.run([sys.executable,str(D/'src/apply_full_train.py')],cwd=P,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError('full_train_rule_failed')
except Exception as e:
 save(D/'full_train_failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
 state('full_train_failed',failure_type=type(e).__name__,next='inspect safe failure metadata; preserve original source and preregister repair')
