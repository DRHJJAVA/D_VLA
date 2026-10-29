from pathlib import Path
import subprocess,time,json,hashlib
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018')
P=R/'runtime/bin/python'
SRC=R/'src/infer_video.py'
PAGE=R/'src/build_review.py'
def stage(name,extra=None):
 x={'stage':name,'time':time.strftime('%Y-%m-%d %H:%M:%S%z')}
 if extra:x.update(extra)
 (R/'inference_state.json').write_text(json.dumps(x,ensure_ascii=False))
def run(name,args):
 log=R/(name+'.log')
 with log.open('ab') as f:
  q=subprocess.run([str(P),*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
 if q.returncode!=0:
  stage(name+'_failed',{'exit_code':q.returncode,'log_file':log.name})
  raise RuntimeError(name+' failed; see remote log')
stage('waiting_weights')
while not (R/'weights_locked.json').exists():
 x=json.loads((R/'stage_state.json').read_text())
 if x['stage']=='download_failed':
  stage('blocked_download',{'reason':x.get('error_type')})
  raise RuntimeError('weights failed')
 time.sleep(10)
stage('smoke')
run('smoke',[SRC,'--smoke'])
smoke=R/'smoke/alphabet_soup_demo_0__A__smoke__all.json'
obj=json.loads(smoke.read_text())
if obj['parse_status']=='unparsed':
 stage('smoke_parse_review',{'reason':'model response not valid JSON','smoke_file':str(smoke)})
 raise RuntimeError('smoke parse review required')
stage('global')
run('global',[SRC,'--phase','global'])
run('page_global',[PAGE])
stage('windows')
run('windows',[SRC,'--phase','window'])
run('page_full',[PAGE])
files=sorted((R/'responses').glob('*.json'))
expected=32+152
assert len(files)==expected,(len(files),expected)
lock={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(R/'predictions_locked.json').write_text(json.dumps({'count':len(files),'files':lock},ensure_ascii=False,indent=2))
stage('fixed_windows_frozen',{'response_count':len(files),'prediction_lock_sha256':hashlib.sha256((R/'predictions_locked.json').read_bytes()).hexdigest(),'next':'shared fine windows before human review'})
