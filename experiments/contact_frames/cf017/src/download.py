import pathlib,json,urllib.request,subprocess,hashlib,concurrent.futures,datetime
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf017')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
op=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:17897','http':'http://127.0.0.1:17897'}))
cfg=json.loads((B/'config.json').read_text());entries=[]
for c in cfg['cases']:
 rows=json.load(op.open('https://huggingface.co/api/datasets/yifengzhu-hf/LIBERO-datasets/tree/main/'+c['suite'],timeout=30))
 x=next(x for x in rows if x['path']==c['suite']+'/'+c['name']);entries.append(dict(**c,metadata=x))
(B/'download_sources.json').write_text(json.dumps(dict(time=now(),repo='yifengzhu-hf/LIBERO-datasets',entries=entries),ensure_ascii=False,indent=2))
def one(e):
 out=B/'raw'/e['name'];part=out.with_suffix('.part')
 url='https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/'+e['metadata']['path']
 with open(B/'raw'/(e['id']+'.log'),'w') as log:
  p=subprocess.run(['curl','--proxy','http://127.0.0.1:17897','-fL','--retry','3','--connect-timeout','20','--max-time','1200','-C','-','-o',str(part),url],stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:return dict(id=e['id'],error='curl_exit_'+str(p.returncode))
 h=hashlib.sha256()
 with part.open('rb') as f:
  for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
 assert part.stat().st_size==e['metadata']['size'] and h.hexdigest()==e['metadata']['lfs']['oid']
 part.rename(out)
 return dict(id=e['id'],path=str(out),size=out.stat().st_size,sha256=h.hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(one,entries))
(B/'download_result.json').write_text(json.dumps(dict(time=now(),results=result),indent=2))
