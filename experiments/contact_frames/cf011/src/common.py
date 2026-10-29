import os,json,hashlib,traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/mnt/nvme1/drh/D_VLA')
BASE=ROOT/'experiments/contact_frames/cf011'
CFG=json.loads((BASE/'config.json').read_text())
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False))
 tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def lock(paths):return {str(p):sha(p) for p in sorted(set(map(Path,paths)))}
def verify(d):
 bad=[p for p,h in d.items() if not Path(p).exists() or sha(p)!=h]
 if bad:raise RuntimeError('input_hash_changed')
 return len(d)
def stage(s,**kw):save(BASE/'stage_state.json',dict(stage=s,time=now(),**kw))
def failure(e):
 (BASE/'failure_traceback.log').write_text(traceback.format_exc())
 save(BASE/'failure.json',dict(time=now(),type=type(e).__name__,frames=[dict(file=x.filename,line=x.lineno) for x in traceback.extract_tb(e.__traceback__)]))
 stage('failed',exception_type=type(e).__name__)
