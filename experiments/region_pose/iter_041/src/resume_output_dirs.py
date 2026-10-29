import json,hashlib,subprocess,sys,datetime,traceback,shutil
from pathlib import Path
P=Path('/mnt/nvme1/drh/D_VLA');D=P/'experiments/region_pose/iter_041';T=D/'tracking_v4'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def state(s,**kw):save(D/'stage_state.json',{'stage':s,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw})
try:
 state('preparing_tracking_output_directories')
 for lock in ['input_locked.json','source_locked.json']:
  for p,h in json.loads((T/lock).read_text()).items():assert sha(p)==h
 assert not list((T/'train').rglob('*.npz'))
 snapshots=T/'failed_execution_041_5';snapshots.mkdir()
 for p in (T/'train').glob('*.json'):shutil.copyfile(p,snapshots/p.name)
 for name in ['trajectories','correspondences','audits','filter_masks']:(T/'train'/name).mkdir(exist_ok=True)
 save(T/'audits/output_directory_preflight.json',{'directories_exist':all((T/'train'/n).is_dir() for n in ['trajectories','correspondences','audits','filter_masks']),'original_source_unchanged':True,'original_failed_log_retained':'track.log','actual_16_query_seed_preflight_previously_passed':True,'npz_before_resume':0,'one_unsaved_first_case_forward_must_be_repeated':True})
 state('tracking_training_fixed_endpoints_v4_resumed',next='audit completed trajectories then full-region remapping/LOO/stable supervision; no original extraction rerun')
 with (T/'track_resumed.log').open('w') as log:r=subprocess.run([sys.executable,str(T/'src/track.py')],cwd=P,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('tracking_resumed_execution_failed')
 for p,h in json.loads((T/'input_locked.json').read_text()).items():assert sha(p)==h
 save(D/'audits/resolved_041_3_to_6.json',{'motion_path_repaired':True,'helper_seed_path_repaired':True,'real_image_preflight_fixed':True,'output_directories_created':True,'original_failed_sources_logs_preserved':True,'tracking_completed':True,'all_input_hashes_unchanged':True})
 state('training_correspondences_computed',results='tracking_v4/train',next='audit numerics and first-failure truncation then actual full region remapping/LOO/stable labels; do not stop at candidate count')
except Exception as e:
 save(D/'tracking_resumed_failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
 state('tracking_resumed_failed',failure_type=type(e).__name__,next='inspect safe failure summary and repair only affected execution')
