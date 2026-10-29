from pathlib import Path
import requests,hashlib,json,os
P=Path(__file__).resolve().parents[1]
url='https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth'
dest=P/'weights/scaled_offline.pth'
dest.parent.mkdir(exist_ok=True)
assert not dest.exists()
partial=dest.with_suffix('.pth.partial')
assert not partial.exists()
with requests.get(url,stream=True,timeout=(15,60)) as response:
 response.raise_for_status()
 with partial.open('xb') as f:
  for chunk in response.iter_content(1024*1024):
   if chunk:f.write(chunk)
  f.flush();os.fsync(f.fileno())
 expected=response.headers.get('Content-Length')
 if expected:assert partial.stat().st_size==int(expected)
partial.rename(dest)
(P/'weights_locked.json').write_text(json.dumps({'source_url':url,'path':str(dest),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))
print('WEIGHTS_COMPLETE',dest.stat().st_size,flush=True)
