import json,hashlib,subprocess,sys,datetime,traceback,ast
from pathlib import Path
import numpy as np
P=Path('/mnt/nvme1/drh/D_VLA');D=P/'experiments/region_pose/iter_041';T=D/'tracking_v4';I=T/'train'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def state(s,**kw):save(D/'stage_state.json',{'stage':s,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw})
try:
 for lock in ['input_locked.json','source_locked.json']:
  for p,h in json.loads((T/lock).read_text()).items():assert sha(p)==h
 for name in ['trajectories','correspondences','audits','filter_masks','case_locks']:(I/name).mkdir(exist_ok=True)
 name='alphabet_soup_demo_0_attempt1_iter041_query41'
 rec=json.loads((I/f'audits/{name}.json').read_text())
 assert rec['selected_held_frames']==[53,55,58,63,127]
 for p in [I/f'trajectories/{name}.npz',I/f'correspondences/{name}.npz']:
  with np.load(p) as z:
   assert len(z.files)>0
   for key in z.files:a=z[key]
 preserved={str(p):sha(p) for p in [I/f'audits/{name}.json',I/f'trajectories/{name}.npz',I/f'correspondences/{name}.npz',*list((I/'filter_masks').glob('*.npz'))]}
 tree=ast.parse((T/'src/track.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');loop=next(n for n in main.body if isinstance(n,ast.For))
 # Complete the interrupted bookkeeping using the original exact casefiles and case_lock statements.
 assert isinstance(loop.body[-3],ast.Assign) and isinstance(loop.body[-2],ast.Expr)
 env={'I':I,'name':name,'case_lock':I/f'case_locks/{name}.json','json':json,'hashlib':hashlib}
 exec(compile(ast.Module(body=loop.body[-3:-1],type_ignores=[]),'original_case_lock_tail','exec'),env)
 assert (I/f'case_locks/{name}.json').is_file()
 assert all(sha(p)==h for p,h in preserved.items())
 save(T/'audits/resume_first_case.json',{'completed_case':name,'case_lock_rebuilt_by_original_AST':True,'existing_case_results_hashes_unchanged':preserved,'all_output_directories_precreated':True,'skip_existing_case_via_original_resume_logic':True})
 state('tracking_training_resumed_with_case_reuse',results='tracking_v4/train',next='continue remaining cases; soup0 first result preserved and skipped via case lock')
 with (T/'track_resumed_v2.log').open('w') as log:
  r=subprocess.run([sys.executable,str(T/'src/track.py')],cwd=P,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('tracking_second_resume_failed')
 assert all(sha(p)==h for p,h in preserved.items())
 for p,h in json.loads((T/'input_locked.json').read_text()).items():assert sha(p)==h
 save(D/'audits/resolved_041_3_to_7.json',{'all_adapter_and_directory_failures_resolved':True,'tracking_completed':True,'first_completed_case_preserved':True,'original_sources_and_logs_retained':True,'input_hashes_unchanged':True})
 state('training_correspondences_computed',results='tracking_v4/train',next='audit numerics and first-failure truncation then full-region remapping, LOO and stable supervision; no validation yet')
except Exception as e:
 save(D/'tracking_second_resume_failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
 state('tracking_second_resume_failed',failure_type=type(e).__name__,next='inspect safe metadata and resume from completed case locks')
