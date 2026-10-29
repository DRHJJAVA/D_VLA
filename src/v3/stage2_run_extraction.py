"""Resource-aware all-frame feature extraction; resumes only completed episode outputs."""
from pathlib import Path
import json,time,subprocess,sys
from stage2_extract_dino import R,OUT,model,extract
rows=[json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]
deadline=time.monotonic()+6*3600
def gpu_state():
 p=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 free,util=[int(x.strip()) for x in p.stdout.splitlines()[0].split(',')]
 return free,util
def ready():
 free,util=gpu_state()
 return free>=20000 and util<85,free,util
print(json.dumps(dict(event='started',episodes=len(rows),policy='wait for GPU free >= 20 GiB and utilisation < 85%; official data read only')),flush=True)
net=None;last_log=0
for row in rows:
 output=OUT/row['asset_group']/(Path(row['rgbd_path']).stem+'.h5')
 if output.exists():
  with __import__('h5py').File(output,'r') as f:
   assert f.attrs['status']=='complete' and int(f.attrs['frames'])==int(row['frames'])
  print(json.dumps(dict(event='reused',id=row['id'])),flush=True);continue
 while True:
  okay,free,util=ready()
  if okay:break
  if time.monotonic()>deadline:raise TimeoutError('Shared GPU unavailable for six hours')
  if time.monotonic()-last_log>600:
   print(json.dumps(dict(event='waiting_gpu',free_mib=free,util_pct=util,id=row['id'])),flush=True);last_log=time.monotonic()
  time.sleep(30)
 if net is None:net=model()
 extract(row,net)
print(json.dumps(dict(event='all_features_complete',episodes=len(rows))),flush=True)
