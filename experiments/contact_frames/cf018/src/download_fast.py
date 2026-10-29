import os,time,json,hashlib,requests,concurrent.futures,threading
from pathlib import Path
from huggingface_hub import snapshot_download
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018')
W=R/'weights'
REV='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
DIGEST=['d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa','8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5','83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192','0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5']
names=[f'model-{i:05d}-of-00004.safetensors' for i in range(1,5)]
def write_stage(stage,extra=None):
 x={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S%z'),'model_revision':REV}
 if extra:x.update(extra)
 (R/'stage_state.json').write_text(json.dumps(x,ensure_ascii=False))
def one(item):
 name,want=item
 url='https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct/resolve/master/'+name
 meta=requests.get(url,headers={'Range':'bytes=0-0'},timeout=30,stream=True)
 assert meta.status_code==206,(name,meta.status_code)
 linked=meta.headers.get('X-Linked-Etag','').strip('"')
 assert linked==want,(name,'modelscope sha differs')
 total=int(meta.headers['Content-Range'].rsplit('/',1)[1])
 meta.close()
 old=list((W/'.cache/huggingface/download').glob(name+'.*.incomplete'))
 part=W/(name+'.partial')
 if old and not part.exists():
  os.replace(old[0],part)
 if part.exists() and part.stat().st_size>total:raise RuntimeError('partial larger than expected')
 for retry in range(8):
  start=part.stat().st_size if part.exists() else 0
  if start==total:break
  try:
   res=requests.get(url,headers={'Range':f'bytes={start}-'},timeout=(20,90),stream=True)
   assert res.status_code==206,(name,res.status_code,start)
   assert res.headers.get('Content-Range','').startswith(f'bytes {start}-')
   with part.open('ab') as f:
    for chunk in res.iter_content(8*1024*1024):
     if chunk:f.write(chunk)
   res.close()
  except Exception:
   if retry==7:raise
   time.sleep(min(2**retry,20))
 assert part.stat().st_size==total,(name,part.stat().st_size,total)
 h=hashlib.sha256()
 with part.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==want,(name,'sha256 mismatch')
 os.replace(part,W/name)
 return {'name':name,'bytes':total,'sha256':want}
try:
 write_stage('fast_download')
 os.environ['HF_ENDPOINT']='https://hf-mirror.com'
 snapshot_download('Qwen/Qwen3-VL-8B-Instruct',revision=REV,local_dir=str(W),endpoint='https://hf-mirror.com',max_workers=4,ignore_patterns=['*.safetensors','*.md','.gitattributes'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  weightfiles=list(pool.map(one,zip(names,DIGEST)))
 assert (W/'model.safetensors.index.json').is_file()
 files=[]
 for p in sorted(W.iterdir()):
  if p.is_file():
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
   files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
 (R/'weights_locked.json').write_text(json.dumps({'revision':REV,'files':files,'source':'Qwen official SHA256; ModelScope CDN byte-identical shards'},ensure_ascii=False,indent=2))
 write_stage('weights_ready',{'files':len(files),'weight_bytes':sum(x['bytes'] for x in weightfiles)})
except Exception as e:
 write_stage('download_failed',{'error_type':type(e).__name__,'message':str(e)[:400]})
 raise
