import os,json,pathlib,traceback,time,hashlib
from huggingface_hub import snapshot_download,HfApi
R=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r2/v0_v1/v001')
os.environ['HTTPS_PROXY']='http://127.0.0.1:7898'
os.environ['HTTP_PROXY']='http://127.0.0.1:7898'
try:
 info=HfApi().model_info('lerobot/smolvla_base')
 out=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/weights/smolvla_base')
 (R/'download_state.json').write_text(json.dumps({'stage':'downloading','revision':info.sha,'started':time.time()}))
 snapshot_download('lerobot/smolvla_base',revision=info.sha,local_dir=out,allow_patterns=['config.json','*.safetensors','*.json'],max_workers=4)
 files=[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in out.iterdir() if p.is_file()]
 (R/'download_complete.json').write_text(json.dumps({'revision':info.sha,'files':files,'path':str(out),'finished':time.time()},indent=2))
except Exception as e:
 (R/'download_failure.json').write_text(json.dumps({'type':type(e).__name__,'summary':str(e).splitlines()[0][:300],'frames':[{'file':pathlib.Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]}))
