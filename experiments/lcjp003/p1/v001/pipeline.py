import pathlib,json,time,subprocess,os,hashlib,fcntl,datetime
P=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/p1/v001')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(d):
 tmp=P/'pipeline_state.tmp';tmp.write_text(json.dumps(dict(time=now(),pid=os.getpid(),**d)));tmp.replace(P/'pipeline_state.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 f=(P/'pipeline.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write({'stage':'waiting_for_inputs'})
 while not (P/'inputs.json').exists():
  if (P/'failure.json').exists():write({'stage':'failed','reason':'input_preparation_failure'});return
  time.sleep(15)
 d=json.loads((P/'inputs.json').read_text());assert d['status']=='complete'
 c=json.loads((P/'cached_window_contract.json').read_text())['checks']
 assert all(v is True for k,v in c.items() if k!='semantic_max_abs')
 assert json.loads((P/'frozen_precision_probe.json').read_text())['cached_vs_same_frame_max']==0
 assert not (P/'source_lock.json').exists()
 files=[P/'config.json',P/'common.py',P/'prepare.py',P/'geometry_cache.py',P/'train.py',P/'pipeline.py',P/'p0/visibility_v3_config.json']+list((P/'source').glob('*.py'))
 (P/'source_lock.json').write_text(json.dumps({str(p.relative_to(P)):sha(p) for p in files},indent=2))
 env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
 proc=subprocess.Popen(['/mnt/nvme1/drh/D_VLA/experiments/lcjp003/runtime/venv/bin/python',str(P/'train.py')],cwd=P,env=env,stdout=(P/'train.stdout.log').open('a'),stderr=(P/'train.stderr.log').open('a'))
 write({'stage':'training_worker','worker_pid':proc.pid})
 rc=proc.wait();write({'stage':'training_worker_finished' if rc==0 else 'failed','worker_pid':proc.pid,'returncode':rc})
if __name__=='__main__':
 try:main()
 except Exception as e:write({'stage':'failed','type':type(e).__name__})
