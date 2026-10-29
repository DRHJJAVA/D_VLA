import os,json,time,hashlib
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018')
revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
(ROOT/'stage_state.json').write_text(json.dumps({'stage':'downloading_weights','time':time.strftime('%Y-%m-%d %H:%M:%S%z'),'revision':revision},ensure_ascii=False))
try:
 path=snapshot_download('Qwen/Qwen3-VL-8B-Instruct',revision=revision,local_dir=str(ROOT/'weights'),endpoint='https://hf-mirror.com',max_workers=4,ignore_patterns=['*.md','.gitattributes'])
 files=[]
 for p in sorted(Path(path).rglob('*')):
  if p.is_file() and '.cache' not in p.parts:
   files.append({'path':str(p.relative_to(path)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 (ROOT/'weights_locked.json').write_text(json.dumps({'revision':revision,'files':files},ensure_ascii=False,indent=2))
 (ROOT/'stage_state.json').write_text(json.dumps({'stage':'weights_ready','time':time.strftime('%Y-%m-%d %H:%M:%S%z'),'files':len(files)},ensure_ascii=False))
except Exception as e:
 (ROOT/'stage_state.json').write_text(json.dumps({'stage':'download_failed','error_type':type(e).__name__,'message':str(e)[:400]},ensure_ascii=False))
 raise
