from pathlib import Path
import time,json,subprocess,hashlib
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018')
P='/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python'
V=str(R/'runtime/bin/python')
def stage(name,extra=None):
 x={'stage':name,'time':time.strftime('%Y-%m-%d %H:%M:%S%z')}
 if extra:x.update(extra)
 (R/'fine_state.json').write_text(json.dumps(x,ensure_ascii=False))
def run(name,cmd):
 log=R/(name+'.log')
 with log.open('ab') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:
  stage(name+'_failed',{'exit_code':r.returncode,'log_file':log.name})
  raise RuntimeError(name+' failed; remote log')
stage('waiting_fixed_windows')
while True:
 p=R/'inference_state.json'
 x=json.loads(p.read_text()) if p.exists() else {}
 if x.get('stage')=='fixed_windows_frozen':break
 if str(x.get('stage','')).endswith('_failed') or x.get('stage') in ['smoke_parse_review','blocked_download']:
  stage('blocked_fixed_windows',{'upstream_stage':x.get('stage')})
  raise RuntimeError('upstream blocked')
 time.sleep(10)
assert json.loads((R/'input_preflight.json').read_text())['stage']=='complete'
assert (R/'prepared_summary_fine.json').exists()
stage('planning_shared_windows')
run('fine_plan',[P,str(R/'src/plan_fine_v2.py')])
plan=json.loads((R/'fine_windows.json').read_text())
expected=4*sum(len(x['windows']) for x in plan['cases'])
stage('fine_inference',{'expected_calls':expected})
if expected:
 run('fine_infer',[V,str(R/'src/infer_fine.py'),'--phase','fine'])
files=sorted((R/'responses_fine').glob('*.json')) if (R/'responses_fine').exists() else []
assert len(files)==expected,(len(files),expected)
lock={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(R/'fine_predictions_locked.json').write_text(json.dumps({'count':len(files),'source_fixed_lock_sha256':hashlib.sha256((R/'predictions_locked.json').read_bytes()).hexdigest(),'files':lock},ensure_ascii=False,indent=2))
stage('fine_frozen',{'response_count':len(files),'lock_sha256':hashlib.sha256((R/'fine_predictions_locked.json').read_bytes()).hexdigest()})
