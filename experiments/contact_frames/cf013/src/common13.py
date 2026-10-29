import os,sys,json,hashlib,datetime
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path('/mnt/nvme1/drh/D_VLA')
BASE=ROOT/'experiments/contact_frames/cf013'
PREV=ROOT/'experiments/contact_frames/cf011'
CFG=json.loads((BASE/'config.json').read_text())
sys.path.insert(0,str(PREV/'src'))
import train as old
import torch,numpy as np
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for v in iter(lambda:f.read(8*1024*1024),b''):h.update(v)
 return h.hexdigest()
def lock(paths):return {str(p):sha(p) for p in paths}
def verify(d):
 bad=[p for p,h in d.items() if not Path(p).exists() or sha(p)!=h]
 assert not bad,('hash_mismatch',bad)
 return len(d)
def stage(s,**kw):save(BASE/'stage_state.json',dict(stage=s,time=now(),**kw))
