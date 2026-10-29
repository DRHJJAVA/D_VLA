"""Wait for shared GPU headroom, then run the isolated local-crop pilot once."""
from pathlib import Path
import json,subprocess,time
from stage2_local_teacher import R,OUTPUT,main
deadline=time.monotonic()+6*3600
last=0.
if (OUTPUT/'done.json').exists():
 print(json.dumps(dict(event='already_complete')),flush=True)
else:
 while True:
  p=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  free,util=map(int,p.stdout.splitlines()[0].split(','))
  if free>=20000 and util<85:break
  if time.monotonic()>deadline:raise TimeoutError('shared GPU unavailable for six hours')
  if time.monotonic()-last>600:
   print(json.dumps(dict(event='waiting_gpu',free_mib=free,util_pct=util)),flush=True);last=time.monotonic()
  time.sleep(30)
 print(json.dumps(dict(event='gpu_ready',free_mib=free,util_pct=util)),flush=True)
 main()
