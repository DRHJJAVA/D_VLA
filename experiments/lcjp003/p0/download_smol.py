
import os,pathlib,json,datetime,hashlib,fcntl
os.environ['HTTP_PROXY']='http://127.0.0.1:7898'
os.environ['HTTPS_PROXY']='http://127.0.0.1:7898'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60'
os.environ['HF_HUB_ETAG_TIMEOUT']='20'
from huggingface_hub import HfApi,snapshot_download
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003')
P=B/'p0/weights_state.json'
lock=(B/'p0/smol_download.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(data):
 tmp=P.with_suffix('.json.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(P)
s=json.loads(P.read_text());s['smolvlm2']['status']='downloading';s['smolvlm2']['pid']=os.getpid();save(s)
try:
 repo=s['smolvlm2']['repo'];info=HfApi().model_info(repo)
 rev=info.sha
 folder=B/'weights/smolvlm2';folder.mkdir(parents=True,exist_ok=True)
 allow=['config.json','model.safetensors','model.safetensors.index.json','model-*.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','preprocessor_config.json','processor_config.json','chat_template.json','chat_template.jinja','added_tokens.json','*.model']
 snapshot_download(repo_id=repo,revision=rev,local_dir=str(folder),allow_patterns=allow,max_workers=3)
 files=[]
 for f in folder.iterdir():
  if f.is_file():
   h=hashlib.sha256()
   with f.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
   files.append({'file':f.name,'bytes':f.stat().st_size,'sha256':h.hexdigest()})
 assert (folder/'config.json').is_file() and list(folder.glob('*.safetensors'))
 s=json.loads(P.read_text());s['smolvlm2'].update({'status':'downloaded','path':str(folder),'revision':rev,'files':files})
 save(s);print(json.dumps({'status':'downloaded','repo':repo,'path':str(folder),'files':len(files)}))
except Exception as e:
 s=json.loads(P.read_text());response=getattr(e,'response',None)
 s['smolvlm2'].update({'status':'download_failed','exception_type':type(e).__name__,'http_status':getattr(response,'status_code',None)})
 save(s);print(json.dumps({'status':'download_failed','exception_type':type(e).__name__,'http_status':getattr(response,'status_code',None)}))
