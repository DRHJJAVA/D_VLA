"""Wait for complete raw-crop features and shared GPU before isolated A_highres pilot."""
from pathlib import Path
import json,subprocess,time,h5py
from stage2_highres_teacher import R,CROPS,OUTPUT,MANIFEST,main
entries=[json.loads(x) for x in MANIFEST.read_text().splitlines() if x]
deadline=time.monotonic()+8*3600;last=0.
if (OUTPUT/'done.json').exists():
 print(json.dumps(dict(event='already_complete')),flush=True)
else:
 while True:
  complete=0
  for e in entries:
   path=CROPS/e['asset_group']/(Path(e['rgbd_path']).stem+'.h5')
   if not path.exists():continue
   with h5py.File(path,'r') as f:
    assert f.attrs['status']=='complete' and int(f.attrs['frames'])==e['frames']
   complete+=1
  p=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  free,util=map(int,p.stdout.splitlines()[0].split(','))
  if complete==len(entries):break
  if time.monotonic()>deadline:raise TimeoutError('high-resolution features unavailable for eight hours')
  if time.monotonic()-last>600:
   print(json.dumps(dict(event='waiting_features',features_complete=complete,episodes=len(entries),free_mib=free,util_pct=util)),flush=True);last=time.monotonic()
  time.sleep(30)
 print(json.dumps(dict(event='ready',features_complete=complete,free_mib=free,util_pct=util)),flush=True)
 main()
