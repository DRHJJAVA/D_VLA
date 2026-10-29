from pathlib import Path
import subprocess,sys,json
from datetime import datetime
from zoneinfo import ZoneInfo
D=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_preference/contact_location_5cam_v1')
def state(stage):
 (D/'stage_state.json').write_text(json.dumps({'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),'stage':stage}))
for name in ['fuse.py','refine_sources.py']:
 state(name)
 with (D/(name+'.log')).open('w') as log:p=subprocess.run([sys.executable,str(D/'fusion/src'/name)],stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:
  state('failed_'+name);break
else:state('fusion_complete')
