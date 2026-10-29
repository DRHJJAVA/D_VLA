import pathlib,json,time,os,subprocess,hashlib,shutil,statistics,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r5/joint/v001');E=B/'engineering/history_throughput'
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA');PY=ROOT/'experiments/lcjp003/runtime/venv/bin/python'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def alive(pid):
 p=pathlib.Path('/proc')/str(pid)
 return p.exists() and p.joinpath('stat').read_text().split(') ')[1][0]!='Z'
def command(pid):return pathlib.Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
def main():
 assert not (E/'deployment_lock.json').exists(),'already_deployed'
 assert not (B/'STOP').exists(),'existing_stop'
 for p in [E/'qualification_first.json',E/'results.json']:
  r=json.loads(p.read_text());assert r['done'] and all(x.get('qualified',True) for x in r['reports'])
  assert all(v['max_abs']==0 for x in r['reports'] for v in x.get('comparison',{}).values()),'require_exact_observed_equivalence'
 r=json.loads((E/'throughput.json').read_text());assert r['done']
 med={k:statistics.median(x['seconds'] for x in r['rows'] if x['batch']==k) for k in [0,8,16]}
 batch=min([8,16],key=lambda k:med[k]);assert med[batch]<med[0]
 state=json.loads((B/'state.json').read_text());pid=state['pid']
 assert alive(pid) and str(B/'src/train.py').encode() in command(pid)
 env=os.environ.copy()
 for item in pathlib.Path('/proc',str(pid),'environ').read_bytes().split(b'\0'):
  if b'=' in item:
   k,v=item.split(b'=',1)
   if k.decode() in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG']:
    env[k.decode()]=v.decode()
 with (ROOT/'docs/LCJP003_状态条件三维部位评分与DriftingVLA设计方案.md').open('a') as f:
  f.write('\n\n### '+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')+' R5 工程优化接续\n两组各8窗口、固定第500次检查点的一致性核对：输入几何/历史摘要/动作/损失/梯度/一次AdamW后模型与优化器最大差异均为0；第二组Torch及CUDA随机状态相同。独立测试优化器状态曾因CPU步数张量共享导致比较器污染，已隔离副本重跑，原证据保留，正式训练未受影响。交错三次中位耗时：'+json.dumps(med)+'秒；选择历史批量'+str(batch)+'，网络、损失、有效批量8及FP32精度不变。现在请求正式训练在更新边界保存全状态再从该步接续。\n')
 write(B/'STOP',{'reason':'authorized_engineering_history_optimization_handoff','time':time.time(),'training_pid':pid})
 deadline=time.time()+90
 while alive(pid) and time.time()<deadline:time.sleep(1)
 assert not alive(pid),'graceful_stop_timeout'
 st=json.loads((B/'state.json').read_text());assert st['status']=='stopped'
 import torch
 ck=torch.load(B/'checkpoints/last.pt',map_location='cpu',weights_only=False)
 assert ck['step']==st['completed_updates']
 shutil.copy2(B/'checkpoints/last.pt',E/'handoff_checkpoint.pt')
 assert sha(E/'handoff_checkpoint.pt')==sha(B/'checkpoints/last.pt')
 assert ck['source_lock']==sha(B/'source_lock.json')
 lock={'resume_step':ck['step'],'resume_checkpoint_sha256':sha(B/'checkpoints/last.pt'),'inherited_source_lock_sha256':sha(B/'source_lock.json'),'config_sha256':sha(B/'config.json'),'history_batch_size':batch,'benchmark_medians_seconds':med,'paired_speedup':med[0]/med[batch],'time':time.time(),'files':{str(p):sha(p) for p in [E/'optimized_history.py',E/'resume_optimized.py',E/'qualification_first.json',E/'results.json',E/'throughput.json']}}
 write(E/'deployment_lock.json',lock)
 (B/'STOP').rename(E/'handoff_stop_request.json')
 log=open(E/'training.log','ab')
 p=subprocess.Popen([str(PY),str(E/'resume_optimized.py')],cwd=str(ROOT),env=env,stdout=log,stderr=log,start_new_session=True)
 write(E/'launch.json',{'training_pid':p.pid,'old_training_pid':pid,'resume_step':ck['step'],'time':time.time()})
 time.sleep(3);assert alive(p.pid),'optimized_process_early_exit'
 executor=str(B/'closed_loop_object50/executor.py')
 active=[]
 for d in pathlib.Path('/proc').glob('[0-9]*'):
  try:
   k=int(d.name)
   if alive(k) and executor.encode() in command(k):active.append(k)
  except (OSError,ValueError):pass
 assert len(active)<=1
 if not active:
  f=open(B/'logs/closed_loop.log','ab')
  q=subprocess.Popen([str(PY),executor],cwd=str(ROOT),env=env,stdout=f,stderr=f,start_new_session=True);active=[q.pid]
 write(E/'handoff_complete.json',{'training_pid':p.pid,'executor_pid':active[0],'resume_step':ck['step'],'history_batch':batch,'checkpoint_preserved':True,'time':time.time()})
try:main()
except Exception as e:
 write(E/'handoff_failure.json',{'type':type(e).__name__,'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)],'time':time.time()})
 raise SystemExit(1)
