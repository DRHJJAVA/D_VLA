import sys,json,time,hashlib,os,requests
from pathlib import Path
from urllib.parse import urlsplit
from datetime import datetime
from zoneinfo import ZoneInfo
r=Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003')
d=r/'weights/dinov3_vits16plus'
d.mkdir(parents=True,exist_ok=True)
name='dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
p=d/name
state=r/'p0/dinov3_direct_download.json'
def save(v):
 v['timestamp_bjt']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
 state.write_text(json.dumps(v,indent=2))
try:
 u=sys.stdin.read().strip()
 a=urlsplit(u)
 assert a.scheme=='https' and a.hostname=='dinov3.llamameta.net' and a.path.endswith('/'+name)
 t=time.monotonic()
 if not p.exists():
  z=requests.get(u,stream=True,timeout=(20,60),proxies={'http':'http://127.0.0.1:7898','https':'http://127.0.0.1:7898'})
  if z.status_code!=200:
   save({'status':'download_failed','http_status':z.status_code,'source_host':a.hostname})
   print(json.dumps({'status':'download_failed','http_status':z.status_code}));sys.exit(1)
  total=int(z.headers.get('content-length',0));n=0
  save({'status':'downloading','bytes':0,'expected_bytes':total,'source_host':a.hostname})
  part=p.with_suffix('.pth.partial')
  with part.open('wb') as f:
   for chunk in z.iter_content(4*1024*1024):
    if chunk:f.write(chunk);n+=len(chunk)
  if total:assert n==total
  os.replace(part,p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 sha=h.hexdigest()
 assert sha.startswith('4057cbaa')
 result={'status':'downloaded_hash_verified','path':str(p),'bytes':p.stat().st_size,'sha256':sha,'official_hash_prefix_verified':True,'source_host':a.hostname,'source_path':a.path,'authorization_url_saved':False,'seconds':time.monotonic()-t}
 save(result);print(json.dumps(result))
except Exception as e:
 result={'status':'failed','exception_type':type(e).__name__}
 save(result);print(json.dumps(result));sys.exit(1)
