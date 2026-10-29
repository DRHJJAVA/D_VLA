import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,datetime,hashlib
from pathlib import Path
ROOT=Path('/mnt/nvme1/drh/D_VLA');BASE=ROOT/'experiments/contact_frames/cf016';OLD=ROOT/'experiments/contact_frames/cf010'
CFG=json.loads((BASE/'config.json').read_text())
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def lock(ps):return {str(p):sha(p) for p in ps}
def verify(d):
 bad=[p for p,h in d.items() if sha(p)!=h];assert not bad,('hash_changed',bad);return len(d)
def stage(s,**kw):save(BASE/'stage_state.json',dict(stage=s,time=now(),**kw))
def dataset(ep):
 if ep in CFG['train_episodes']:return OLD/'data'/ep/'artifacts/dataset.npz'
 return OLD/'evaluation'/ep/'artifacts/dataset.npz'
def episode_config(ep):
 if ep in CFG['train_episodes']:return ROOT/'experiments/contact_frames/cf005/data'/ep/'config.json'
 return dataset(ep).resolve().parent.parent/'config.json'
