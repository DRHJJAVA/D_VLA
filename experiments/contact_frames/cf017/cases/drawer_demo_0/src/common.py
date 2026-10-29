from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,os,traceback
ROOT=Path("/mnt/nvme1/drh/D_VLA")
RUN=ROOT/"/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf017/cases/drawer_demo_0"
DOC=ROOT/"docs/小型验证方案_夹持运动证据与区域表示_v1.md"
def now():return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S UTC+08:00")
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(2**20),b""):h.update(b)
 return h.hexdigest()
def stage(name,**kw):
 save(RUN/"stage_state.json",dict(stage=name,time=now(),pid=os.getpid(),**kw))
 print(json.dumps(dict(stage=name,time=now(),**kw),ensure_ascii=False),flush=True)
def fail(e):
 save(RUN/"failure.json",{"time":now(),"type":type(e).__name__,"message":str(e)[:500],"stack":[{"file":x.filename,"line":x.lineno,"function":x.name} for x in traceback.extract_tb(e.__traceback__)]})
 stage("failed",error_type=type(e).__name__)
