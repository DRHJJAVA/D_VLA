import sys,json,hashlib,traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/mnt/nvme1/drh/D_VLA')
BASE=ROOT/'experiments/contact_frames/cf012'
PREV=ROOT/'experiments/contact_frames/cf011'
sys.path.insert(0,str(PREV/'src'))
import train as old
CFG=json.loads((BASE/'config.json').read_text())
save=old.save;sha=old.sha;lock=old.lock;verify=old.verify
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def stage(s,**kw):save(BASE/'stage_state.json',dict(stage=s,time=now(),**kw))
def failure(e):
 (BASE/'failure_traceback.log').write_text(traceback.format_exc())
 save(BASE/'failure.json',dict(time=now(),type=type(e).__name__,frames=[dict(file=x.filename,line=x.lineno) for x in traceback.extract_tb(e.__traceback__)]))
 stage('failed',exception_type=type(e).__name__)
