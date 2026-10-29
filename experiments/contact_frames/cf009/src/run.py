from pathlib import Path
import subprocess,json,os,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf009')
PY='/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def state(s,**kw):(R/'stage_state.json').write_text(json.dumps(dict(stage=s,time=now(),**kw),indent=2))
assert (R/'train_supervision_summary.json').exists()
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
for phase,script in [('preflight',R/'src/preflight.py'),('positive_evidence',R/'positive_evidence/src/train.py'),('persistent_evidence',R/'persistent_evidence/src/train.py')]:
 state('running_'+phase)
 with (R/(phase+'.log')).open('w') as f:
  p=subprocess.run([PY,str(script)],cwd='/mnt/nvme1/drh/D_VLA',stdout=f,stderr=f,env=env)
 if p.returncode:
  state('failed',phase=phase,returncode=p.returncode)
  raise SystemExit(p.returncode)
state('training_evaluation_complete')
