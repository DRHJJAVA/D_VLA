import subprocess,time,json
from pathlib import Path
ROOT=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018')
PY='/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python'
V=ROOT/'runtime'
try:
 subprocess.run([PY,'-m','venv','--system-site-packages',str(V)],check=True)
 subprocess.run([str(V/'bin/python'),'-m','pip','install','--upgrade','transformers==4.57.3','huggingface_hub==0.36.0','accelerate==1.12.0'],check=True)
 import os
 r=subprocess.run([str(V/'bin/python'),'-c','import transformers,torch,huggingface_hub,accelerate; print(transformers.__version__,torch.__version__,huggingface_hub.__version__,accelerate.__version__)'],capture_output=True,text=True,check=True)
 (ROOT/'runtime_state.json').write_text(json.dumps({'stage':'ready','versions':r.stdout.strip(),'time':time.strftime('%Y-%m-%d %H:%M:%S%z')},ensure_ascii=False))
except Exception as e:
 (ROOT/'runtime_state.json').write_text(json.dumps({'stage':'failed','error_type':type(e).__name__,'message':str(e)[:300]},ensure_ascii=False))
 raise
