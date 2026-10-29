import sys,pathlib,json,hashlib,time,os,traceback
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r5/joint/v001')
E=B/'engineering/history_throughput'
sys.path.insert(0,str(B/'src'));sys.path.insert(0,str(E))
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 lock=json.loads((E/'deployment_lock.json').read_text())
 for name,sha in lock['files'].items():assert digest(name)==sha, 'engineering_dependency_hash'
 assert digest(B/'source_lock.json')==lock['inherited_source_lock_sha256']
 assert digest(B/'config.json')==lock['config_sha256']
 import train,io_runtime as io
 from optimized_history import install
 ck=io.torch.load(B/'checkpoints/last.pt',map_location='cpu',weights_only=False)
 assert ck['step']==lock['resume_step'] and digest(B/'checkpoints/last.pt')==lock['resume_checkpoint_sha256']
 assert not (B/'STOP').exists()
 install(train,lock['history_batch_size'])
 original_checkpoint=train.checkpoint
 def checkpoint(model,opt,step,extra=None):
  addition=dict(extra or {})
  addition['engineering_history_throughput']={'lock_sha256':digest(E/'deployment_lock.json'),'history_batch_size':lock['history_batch_size'],'cache_scope':'one_microbatch_only','inherited_resume_step':lock['resume_step']}
  return original_checkpoint(model,opt,step,addition)
 train.checkpoint=checkpoint
 io.atomic('engineering/history_throughput/resumed.json',{'pid':os.getpid(),'step':ck['step'],'time':time.time(),'checkpoint_sha256':lock['resume_checkpoint_sha256'],'engineer_lock_sha256':digest(E/'deployment_lock.json')})
 del ck
 train.main()
if __name__=='__main__':
 try:main()
 except Exception as e:
  data={'type':type(e).__name__,'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)],'time':time.time()}
  (E/'resume_failure.json').write_text(json.dumps(data,indent=2))
  raise SystemExit(1)
