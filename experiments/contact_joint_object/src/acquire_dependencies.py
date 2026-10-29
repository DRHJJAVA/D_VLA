from pathlib import Path
import requests, subprocess,json,hashlib,traceback,os,time
B=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object')
out={}
proxy={'https':'http://127.0.0.1:17897','http':'http://127.0.0.1:17897'}
for name,url in [('dino3','https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/resolve/main/model.safetensors'),('sam21_tiny','https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt')]:
 try:
  r=requests.get(url,proxies=proxy,timeout=(15,90),stream=True)
  out[name]={'status':r.status_code}
  if r.status_code==200:
   p=B/'dependencies'/(name+('.safetensors' if name=='dino3' else '.pt'));tmp=p.with_suffix('.partial')
   with tmp.open('wb') as f:
    for b in r.iter_content(1024*1024):f.write(b)
   tmp.replace(p);out[name].update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  r.close()
 except Exception as e:out[name]={'error':type(e).__name__}
 (B/'dependencies/status.json').write_text(json.dumps(out,indent=2))
target=B/'dependencies/sam2'
if not target.exists():
 r=subprocess.run(['git','-c','http.proxy=http://127.0.0.1:17897','clone','--depth','1','https://github.com/facebookresearch/sam2.git',str(target)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (B/'logs/sam2_clone.log').write_text(r.stdout);out['sam2_clone_returncode']=r.returncode
if target.exists():
 out['sam2_commit']=subprocess.run(['git','rev-parse','HEAD'],cwd=target,capture_output=True,text=True).stdout.strip()
(B/'dependencies/status.json').write_text(json.dumps(out,indent=2))
