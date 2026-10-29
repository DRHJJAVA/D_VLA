"""Resource-aware high-resolution crop extraction; completed episodes are reused."""
import json,subprocess,time
from pathlib import Path
import h5py
from stage2_extract_highres_crops import R,OUT,model,extract
entries=[json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]
deadline=time.monotonic()+6*3600;last=0.;net=None
for e in entries:
 target=OUT/e['asset_group']/(Path(e['rgbd_path']).stem+'.h5')
 if target.exists():
  with h5py.File(target,'r') as f:
   assert f.attrs['status']=='complete' and f.attrs['source_rgbd']==e['rgbd_path'] and int(f.attrs['frames'])==e['frames']
  print(json.dumps(dict(event='reused',id=e['id'])),flush=True);continue
 while True:
  p=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  free,util=map(int,p.stdout.splitlines()[0].split(','))
  if free>=20000 and util<85:break
  if time.monotonic()>deadline:raise TimeoutError('shared GPU unavailable for six hours')
  if time.monotonic()-last>600:
   print(json.dumps(dict(event='waiting_gpu',id=e['id'],free_mib=free,util_pct=util)),flush=True);last=time.monotonic()
  time.sleep(30)
 if net is None:net=model()
 extract(e,net)
print(json.dumps(dict(event='all_highres_crops_complete',episodes=len(entries))),flush=True)
