from pathlib import Path
import ast,json,hashlib,datetime,subprocess
R=Path('/mnt/nvme1/drh/D_VLA');E=R/'experiments/region_pose';I=E/'iter_038'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
class Adapt(ast.NodeTransformer):
 def visit_Constant(self,n):
  if n.value=='iter_033':return ast.copy_location(ast.Constant('iter_038'),n)
  if n.value=='train':return ast.copy_location(ast.Constant('val'),n)
  if n.value in ['source_addendum_033_1.json','source_addendum_033_2.json']:return ast.copy_location(ast.Constant('source_locked.json'),n)
  return n
 def visit_Dict(self,n):
  self.generic_visit(n)
  for j,k in enumerate(n.keys):
   if isinstance(k,ast.Constant) and k.value=='validation_read':n.values[j]=ast.copy_location(ast.Constant(True),n.values[j])
  return n
methods={}
for orig,new in [('compare_sources_v3.py','compare_sources.py'),('independent_audit_v2.py','independent_audit.py')]:
 t=ast.parse((E/'iter_033/src'/orig).read_text());oldfun={n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)}
 adapted=ast.fix_missing_locations(Adapt().visit(t))
 for n in adapted.body:
  if isinstance(n,ast.FunctionDef):assert ast.dump(n,include_attributes=False)==oldfun[n.name]
 text=ast.unparse(adapted)+'\n';(I/'src'/new).write_text(text)
 methods[new]={'source':str(E/'iter_033/src'/orig),'original_sha256':sha(E/'iter_033/src'/orig),'adapted_sha256':sha(I/'src'/new),'all_functions_AST_equal':True,'changes':['iteration output path','train split to original val','source lock references consolidated','validation_read metadata true'],'full_adapted_AST_hash':hashlib.sha256(ast.dump(adapted,include_attributes=False).encode()).hexdigest()}
save(I/'audits/method_adapter.json',methods)
protocol=json.loads((E/'iter_002/evidence/base/protocol.json').read_text())
def find(x):
 if isinstance(x,dict):
  if x.get('id')=='tomato_sauce_demo_4':return [x]
  return sum((find(v) for v in x.values()),[])
 if isinstance(x,list):return sum((find(v) for v in x),[])
 return []
cases=find(protocol);assert len(cases)==1 and cases[0]['split']=='val'
save(I/'cases.json',cases)
save(I/'config.json',{'episode':'tomato_sauce_demo_4','frames':'all t>=1, total132','attempts':[[50,113]],'voxel_m':.001,'min_voxels':3,'min_run':5,'validation_read':True,'test_read':False,'new_training':False,'new_labels':False,'method_frozen_iteration':33})
oldinputs=json.loads((E/'iter_033/input_locked.json').read_text())
inputs=[Path(k) for k in oldinputs if not any(s in k for s in ['/evidence/motion/','/data/v2/','/data/v3/']) and 'iter_032/completed' not in k]
inputs += [Path(cases[0]['rgbd']),E/'iter_003/evidence/motion/tomato_sauce_demo_4.json',E/'iter_003/evidence/motion/tomato_sauce_demo_4.npz',E/'iter_033/src/compare_sources_v3.py',E/'iter_033/src/independent_audit_v2.py',E/'iter_033/completed_locked.json',E/'iter_037/completed_locked.json',E/'iter_037/pre_validation_freeze.json',E/'iter_037/summary.json']
save(I/'input_locked.json',{str(p):sha(p) for p in inputs})
src=[I/'设计方案.md',I/'config.json',I/'cases.json',I/'src/prepare.py',I/'src/compare_sources.py',I/'src/independent_audit.py',I/'audits/method_adapter.json']
save(I/'source_locked.json',{str(p):sha(p) for p in src})
for name in ['033','037']:
 d=json.loads((E/f'iter_{name}/completed_locked.json').read_text())
 assert all(sha(R/k)==v for k,v in d.items())
save(I/'pre_validation_freeze.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_freeze':json.loads((E/'iter_037/pre_validation_freeze.json').read_text()),'method_source_sha256':methods,'validation_episode':'tomato_sauce_demo_4','before_validation_RGBD_computation':True})
p=subprocess.Popen(['/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python',str(I/'src/compare_sources.py')],cwd=R,stdout=open(I/'compare.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
save(I/'launch.json',{'pid':p.pid});print(json.dumps({'pid':p.pid,'inputs':len(inputs),'sources':len(src)}))
