
import os,pathlib,json,hashlib,fcntl
os.environ['HTTP_PROXY']='http://127.0.0.1:7898'
os.environ['HTTPS_PROXY']='http://127.0.0.1:7898'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60'
from huggingface_hub import HfApi,snapshot_download,get_token
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003');P=B/'p0/weights_state.json'
lock=(B/'p0/dinov3_download.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(s):
 q=P.with_suffix('.json.tmp');q.write_text(json.dumps(s,indent=2));q.replace(P)
s=json.loads(P.read_text());token=get_token()
if not token:
 s['dinov3'].update({'status':'blocked_access','credentials_present':False,'weights_downloaded':False});save(s)
 print(json.dumps({'status':'blocked_access','reason':'no_remote_hf_credential'}));raise SystemExit(2)
try:
 repo='facebook/dinov3-vits16plus-pretrain-lvd1689m';info=HfApi().model_info(repo,token=token)
 folder=B/'weights/dinov3_vits16plus';folder.mkdir(parents=True,exist_ok=True)
 s['dinov3'].update({'status':'downloading','credentials_present':True,'revision':info.sha,'pid':os.getpid()});save(s)
 snapshot_download(repo,revision=info.sha,local_dir=str(folder),token=token,allow_patterns=['config.json','model.safetensors','preprocessor_config.json'],max_workers=2)
 files=[]
 for f in folder.iterdir():
  if f.is_file():
   h=hashlib.sha256()
   with f.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
   files.append({'file':f.name,'bytes':f.stat().st_size,'sha256':h.hexdigest()})
 assert (folder/'model.safetensors').is_file()
 s=json.loads(P.read_text());s['dinov3'].update({'status':'downloaded','path':str(folder),'files':files,'weights_downloaded':True,'http_status':200,'error_code':None});save(s)
 print(json.dumps({'status':'downloaded','path':str(folder),'files':len(files)}))
except Exception as e:
 r=getattr(e,'response',None);s=json.loads(P.read_text());s['dinov3'].update({'status':'download_failed','exception_type':type(e).__name__,'http_status':getattr(r,'status_code',None)})
 if r is not None:s['dinov3']['error_code']=r.headers.get('X-Error-Code')
 save(s);print(json.dumps({'status':'download_failed','exception_type':type(e).__name__,'http_status':getattr(r,'status_code',None)}))
