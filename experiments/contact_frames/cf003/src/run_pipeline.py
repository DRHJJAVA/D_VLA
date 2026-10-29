from pathlib import Path
import subprocess,json,datetime,traceback
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/contact_frames/cf003';PY='/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python'
def save(p,x):p.write_text(json.dumps(x,indent=2))
def main():
 cfg=json.loads((B/'config.json').read_text())
 for ep in cfg['new_preparations']:
  save(B/'stage_state.json',{'stage':'preparing','episode':ep,'time':datetime.datetime.now().isoformat(timespec='seconds')})
  c=B/'data'/ep
  with (c/'prepare.log').open('w') as f:ret=subprocess.run([PY,str(c/'src/prepare.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  if ret.returncode:raise RuntimeError('prepare failed '+ep)
 save(B/'stage_state.json',{'stage':'training','time':datetime.datetime.now().isoformat(timespec='seconds')})
 with (B/'train.log').open('w') as f:ret=subprocess.run([PY,str(B/'src/train_and_evaluate.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT)
 if ret.returncode:raise RuntimeError('train/evaluate failed')
except_placeholder=0
if __name__=='__main__':
 try:main()
 except Exception as e:save(B/'failure.json',{'type':type(e).__name__,'message':str(e),'stack':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
