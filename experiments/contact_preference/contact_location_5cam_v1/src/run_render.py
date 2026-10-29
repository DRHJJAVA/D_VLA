from pathlib import Path
import json,sys,subprocess,hashlib,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
B=Path('/mnt/nvme1/drh/D_VLA');D=B/'experiments/contact_preference/contact_location_5cam_v1'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 cfg=json.loads((D/'config.json').read_text());inp={c['raw']:sha(c['raw']) for c in cfg['cases']};save(D/'raw_input_locked.json',inp)
 for c in cfg['cases']:
  folder=D/'rgbd'/c['asset'];folder.mkdir(exist_ok=True);out=folder/(c['demo']+'.h5')
  save(D/'stage_state.json',{'time':now(),'stage':'rendering','case':c['asset'],'cameras':cfg['cameras']})
  cmd=[sys.executable,str(D/'src/render_five.py'),'--input',c['raw'],'--output',str(out),'--libero-root','/home/drh/project/ECV-VLA/vendor/LIBERO','--bddl-subdir',c['suite'],'--episode',c['demo'],'--frames',','.join(map(str,c['render_frames'])),'--size','448']
  with (D/(c['asset']+'_render.log')).open('w') as log:
   p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=str(B))
  if p.returncode:raise RuntimeError('render_failed_'+c['asset']+'_exit_'+str(p.returncode))
 assert all(sha(p)==h for p,h in inp.items())
 save(D/'stage_state.json',{'time':now(),'stage':'rendered_pending_consistency','cameras':cfg['cameras']})
 print('rendered_all',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(D/'failure.json',{'time':now(),'type':type(e).__name__,'message':str(e),'stack':[(x.filename,x.lineno) for x in traceback.extract_tb(e.__traceback__)]})
  save(D/'stage_state.json',{'time':now(),'stage':'failed','type':type(e).__name__})
  traceback.print_exc(file=(D/'failure.log').open('w'))
