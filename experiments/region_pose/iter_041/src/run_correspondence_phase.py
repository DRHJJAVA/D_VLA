import ast,json,hashlib,subprocess,sys,datetime,traceback,shutil
from pathlib import Path
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';D=B/'iter_041';E=D/'full_train_new15';T=D/'tracking'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def state(s,**kw):save(D/'stage_state.json',{'stage':s,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw})
def checked_lock(path,only=None):
 locked=json.loads(Path(path).read_text());count=0
 for k,v in locked.items():
  p=Path(k);p=p if p.is_absolute() else P/p
  if only is not None and p not in only:continue
  assert sha(p)==(v if isinstance(v,str) else v['sha256']);count+=1
 return count
try:
 state('auditing_full_train_sources')
 original=B/'iter_033/src/independent_audit_v2.py'
 tree=ast.parse(original.read_text())
 class AuditAdapter(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value=='iter_033':return ast.copy_location(ast.Constant(value='iter_041/full_train_new15'),n)
   return n
 tree=AuditAdapter().visit(tree);ast.fix_missing_locations(tree);p=E/'src/independent_audit.py';p.write_text(ast.unparse(tree)+'\n')
 reverse=ast.parse(p.read_text())
 class Reverse(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value=='iter_041/full_train_new15':return ast.copy_location(ast.Constant(value='iter_033'),n)
   return n
 assert ast.dump(Reverse().visit(reverse),include_attributes=False)==ast.dump(ast.parse(original.read_text()),include_attributes=False)
 save(D/'source_independent_audit_addendum.json',{str(q.relative_to(P)):sha(q) for q in [p,original,D/'实施范围041_2.md']})
 with (E/'independent_audit.log').open('w') as log:
  result=subprocess.run([sys.executable,str(p)],cwd=P,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError('independent_full_training_source_audit_failed')
 assert (E/'audits/independent_reconstruction.json').is_file()
 state('preparing_fixed_training_correspondences')
 T.mkdir();(T/'src').mkdir();(T/'seeds').mkdir();(T/'train').mkdir();(T/'audits').mkdir()
 old={x['id']:x for x in json.loads((B/'iter_022/cases_runtime_022_1.json').read_text())}
 old.update({x['id']:x for x in json.loads((B/'iter_035/case.json').read_text())})
 refs={x['id']:x for x in json.loads((B/'iter_033/cases.json').read_text())+json.loads((E/'cases.json').read_text())}
 rows=json.loads((D/'train_all/summary.json').read_text())['rows']
 cases=[];reuse=[];seed_records=[];inputs={}
 for r in rows:
  for c in r['candidates']:
   name=r['episode'];o=old[name];query=r['attempt_start']-1
   assert query==o['query'] and o['split']=='train'
   if name=='white_yellow_mug_demo_6':
    assert o['selected_held_frames']==c['endpoint_frames']
    reused=checked_lock(B/'iter_035/completed_locked.json')
    reuse.append({'id':name,'query':query,'endpoints':c['endpoint_frames'],'source_iteration':35,'verified_locked_files':reused,'stable_target_available':False,'stable_positive_query_sources':1,'first_quality_failure_frame':65})
    continue
   seed=B/f'iter_022/seeds/{name}.npz'
   assert checked_lock(B/'iter_022/completed_locked.json',{seed})==1
   shutil.copyfile(seed,T/f'seeds/{name}.npz')
   ref=refs[name];ref['component']=str(P/f'data/v3/proposals/foreground3d_components/train/{name}.npz');assert Path(ref['component']).is_file()
   cases.append({'id':name,'event':f"{name}_attempt{r['attempt']}_iter041",'split':'train','query':query,'bridges':[{'frame':i} for i in c['endpoint_frames']],'selected_held_frames':c['endpoint_frames'],'early_support_start':c['endpoint_frames'][0],'source_kind':'041_split_motion_and_visibility_candidate_not_label'})
   seed_records.append({'id':name,'query':query,'initial_seed_source':str(seed.relative_to(P)),'seed_sha256':sha(seed),'clip_frames':c['endpoint_frames'][-1]-query+1,'same_initial_seed_as_022':True})
   for q in [seed,Path(ref['component']),Path(ref['rgbd']),B/f'iter_003/evidence/motion/{name}.json',B/f'iter_003/evidence/motion/{name}.npz']:
    inputs[str(q)]=sha(q)
 assert len(cases)==16 and len(reuse)==1
 save(T/'case.json',cases);save(T/'refs.json',{x['id']:refs[x['id']] for x in cases})
 save(T/'reuse.json',reuse);save(T/'seed_and_clip_audit.json',seed_records)
 source=B/'iter_035/src/track.py';text=source.read_text()
 oldbase=str(B/'iter_035');newbase=str(T)
 tree=ast.parse(text)
 class TrackAdapter(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value==oldbase:return ast.copy_location(ast.Constant(value=newbase),n)
   return n
 tree=TrackAdapter().visit(tree);ast.fix_missing_locations(tree)
 target=T/'src/track.py';target.write_text(ast.unparse(tree)+'\n')
 class TrackReverse(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value==newbase:return ast.copy_location(ast.Constant(value=oldbase),n)
   return n
 assert ast.dump(TrackReverse().visit(ast.parse(target.read_text())),include_attributes=False)==ast.dump(ast.parse(text),include_attributes=False)
 helper=B/'iter_035/src/sampling_helpers.py';shutil.copyfile(helper,T/'src/sampling_helpers.py')
 weights=B/'iter_010/weights/scaled_offline.pth'
 assert sha(weights)=='2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834'
 for q in [weights,source,helper,T/'case.json',T/'refs.json',D/'train_all/summary.json']:
  inputs[str(q)]=sha(q)
 save(T/'input_locked.json',inputs)
 save(T/'source_locked.json',{str(q):sha(q) for q in [target,T/'src/sampling_helpers.py',D/'设计方案.md',D/'config.json',D/'实施范围041_2.md',D/'src/run_correspondence_phase.py']})
 save(T/'seed_locked.json',{str(q):sha(q) for q in (T/'seeds').glob('*.npz')})
 save(T/'audits/method_identity.json',{'reverse_ast_equal_to_035_except_BASE':True,'helper_byte_equal':sha(helper)==sha(T/'src/sampling_helpers.py'),'same_query_and_seeds':True,'train_only_cases':16,'reuse_cases':1,'clip_lengths':[x['clip_frames'] for x in seed_records],'new_clip_context_not_old_prediction_reuse':True})
 state('tracking_training_fixed_endpoints',cases=16,reuse=1,next='after correspondence: audit tracks then actual full-volume remapping, LOO and stable targets; do not stop at candidate count')
 with (T/'track.log').open('w') as log:
  result=subprocess.run([sys.executable,str(target)],cwd=P,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError('training_correspondence_failed')
 assert all(sha(k)==v for k,v in inputs.items())
 state('training_correspondences_computed',next='audit tracking numerics and source identity then LOO and stable target generation for qualifying training windows; validation not yet accessed')
except Exception as e:
 save(D/'correspondence_phase_failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
 state('correspondence_phase_failed',failure_type=type(e).__name__,next='inspect safe failure metadata and preregister targeted repair')
