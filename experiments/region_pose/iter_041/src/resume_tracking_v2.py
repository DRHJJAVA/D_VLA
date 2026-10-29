import ast,json,hashlib,subprocess,sys,datetime,traceback,shutil
from pathlib import Path
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';D=B/'iter_041';O=D/'tracking';T=D/'tracking_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def state(s,**kw):save(D/'stage_state.json',{'stage':s,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw})
try:
 state('preparing_tracking_path_repair')
 assert not T.exists();(T/'src').mkdir(parents=True);(T/'seeds').mkdir();(T/'train').mkdir();(T/'audits').mkdir()
 inputs=json.loads((O/'input_locked.json').read_text())
 for path,h in inputs.items():assert sha(path)==h
 for name in ['case.json','refs.json','reuse.json','seed_and_clip_audit.json']:
  shutil.copyfile(O/name,T/name)
 for seed in (O/'seeds').glob('*.npz'):shutil.copyfile(seed,T/'seeds'/seed.name)
 source=B/'iter_035/src/track.py';tree=ast.parse(source.read_text())
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE' for t in node.targets):
   node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(value=str(T))],keywords=[])
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OLD' for t in node.targets):
   node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(value=str(B/'iter_003'))],keywords=[])
 ast.fix_missing_locations(tree);target=T/'src/track.py';target.write_text(ast.unparse(tree)+'\n')
 # Compare complete AST after replacing only the two registered path assignments with original nodes.
 original=ast.parse(source.read_text());reverse=ast.parse(target.read_text())
 assignments={n.targets[0].id:n for n in original.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['BASE','OLD']}
 reverse.body=[assignments[n.targets[0].id] if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in assignments else n for n in reverse.body]
 assert ast.dump(reverse,include_attributes=False)==ast.dump(original,include_attributes=False)
 shutil.copyfile(O/'src/sampling_helpers.py',T/'src/sampling_helpers.py')
 cases=json.loads((T/'case.json').read_text());refs=json.loads((T/'refs.json').read_text())
 for c in cases:
  assert (B/f"iter_003/evidence/motion/{c['id']}.npz").is_file()
  for field in ['rgbd','component']:assert Path(refs[c['id']][field]).is_file()
  assert (T/f"seeds/{c['id']}.npz").is_file()
 for p in [T/'case.json',T/'refs.json',*list((T/'seeds').glob('*.npz'))]:inputs[str(p)]=sha(p)
 save(T/'input_locked.json',inputs)
 save(T/'source_locked.json',{str(p):sha(p) for p in [target,T/'src/sampling_helpers.py',D/'实施修复041_3.md',D/'src/resume_tracking_v2.py']})
 save(T/'audits/preflight.json',{'same_AST_except_BASE_and_OLD':True,'motion_paths_exist':len(cases),'refs_and_seed_paths_exist':True,'old_inputs_hash_unchanged':True,'original_failed_run_preserved':True})
 state('tracking_training_fixed_endpoints_v2',cases=16,reuse=1,next='audit new tracking results then full-volume remapping/LOO/stable supervision')
 with (T/'track.log').open('w') as log:
  p=subprocess.run([sys.executable,str(target)],cwd=P,stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError('training_correspondence_v2_failed')
 assert all(sha(path)==h for path,h in inputs.items())
 save(D/'audits/resolved_041_3.json',{'original_failure':'tracking/track.log','reason':'nested BASE.parent made OLD point to iter_041/iter_003','repair':'explicit original iter_003 path, separate tracking_v2 output','v2_completed':True,'old_failure_preserved':True})
 state('training_correspondences_computed',results='tracking_v2/train',next='audit new tracking numerics then remap actual full regions, LOO and stable labels; training only')
except Exception as e:
 save(D/'correspondence_v2_failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
 state('correspondence_v2_failed',failure_type=type(e).__name__,next='diagnose before repair; do not rerun completed full-volume extraction')
