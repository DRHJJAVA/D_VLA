"""Wait for all original-sequence features, then run fixed Stage-2 comparisons."""
from pathlib import Path
import json,time,subprocess,sys
R=Path('/mnt/nvme1/drh/D_VLA');F=R/'data/v3/features/dinov2_all_frames'
rows=[json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]
deadline=time.monotonic()+8*3600;last=0
while True:
 complete=sum((F/x['asset_group']/(Path(x['rgbd_path']).stem+'.h5')).exists() for x in rows)
 if complete==len(rows):break
 if time.monotonic()>deadline:raise TimeoutError('Feature extraction did not finish in eight hours')
 if time.monotonic()-last>600:
  print(json.dumps(dict(event='waiting_features',complete=complete,total=len(rows))),flush=True);last=time.monotonic()
 time.sleep(30)
print(json.dumps(dict(event='features_ready',count=complete)),flush=True)
# A coarse full-image pilot is evaluated before scaling to all variants.
variant='A';seed=2026
out=R/'outputs/v3_stage2'/variant/f'seed_{seed}'
if not (out/'done.json').exists():
 print(json.dumps(dict(event='train_start',variant=variant,seed=seed)),flush=True)
 cmd=[sys.executable,'-u',str(R/'src/v3/stage2_crossview_teacher.py'),'--variant',variant,'--seed',str(seed),'--epochs','8']
 subprocess.run(cmd,check=True)
print(json.dumps(dict(event='coarse_teacher_pilot_complete',variant=variant,seed=seed)),flush=True)
