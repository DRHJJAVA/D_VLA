import pathlib,json,time,os,fcntl,subprocess,hashlib,datetime
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA');RUN=ROOT/'experiments/lcjp003/r4/joint/v001'
lock=open(RUN/'launch.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
start=time.time()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
while not (RUN/'checks/input_contract.json').exists():
 if (RUN/'STOP').exists():raise SystemExit('paused')
 p=RUN/'failure.json'
 if p.exists():
  err=json.loads(p.read_text())
  if err.get('stage')=='data_contract':raise SystemExit('input gate failed')
 time.sleep(3)
for n in ['checks/input_contract.json','checks/network_and_gradient.json']:
 assert json.loads((RUN/n).read_text())['passed']
if (RUN/'launch.json').exists():raise SystemExit('already launched; inspect actual process before recovery')
cfg=json.loads((RUN/'config.json').read_text());cfg.update(status='formal_locked',deterministic_algorithms=True,cublas_workspace=':4096:8',sam_grid_offline_only=True)
(RUN/'config.json').write_text(json.dumps(cfg,indent=2))
gate=json.loads((RUN/'checks/network_and_gradient.json').read_text())
for name,value in gate['source_hashes'].items():
 assert sha(RUN/'src'/name)==value,'source_changed_since_validation:'+name
paths=list((RUN/'src').glob('*.py'))+[RUN/n for n in ['config.json','episodes.json','normalization.json','input_manifest.json','evaluation_windows.json','sampling_schedule.json','supervision_protocol.json']]
for base,names in [('r3/joint/v001/src',['geometry.py','supervision.py','visibility.py']),('r2/v2/v001/src',['common.py'])]:
 paths += [ROOT/'experiments/lcjp003'/base/n for n in names]
files={str(p.relative_to(ROOT)):sha(p) for p in paths}
(RUN/'source_lock.json').write_text(json.dumps({'time':time.time(),'files':files,'design_version':'LCJP003-R4-design-v1','no_R3_trained_weights':True},indent=2))
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false')
log=open(RUN/'logs/train.log','a')
p=subprocess.Popen([str(ROOT/'experiments/lcjp003/runtime/venv/bin/python'),str(RUN/'src/train.py')],env=env,stdout=log,stderr=log,cwd=ROOT,start_new_session=True)
(RUN/'launch.json').write_text(json.dumps({'pid':p.pid,'time':time.time(),'source_lock_sha256':sha(RUN/'source_lock.json'),'automatic_closed_loop':False}))
ts=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
with (ROOT/'docs/LCJP003_状态条件三维部位评分与DriftingVLA设计方案.md').open('a') as f:
 f.write('\n'+ts+'：R4输入与梯度/恢复检查通过，锁定config、source、采样日程及监督协议，启动正式训练进程。先保存step0000并完成80固定窗口预测掩膜基线，随后3000次联合更新；第20/50次保留进度并测时，每250次评估。未启动闭环。正式启动PID '+str(p.pid)+'。\n')
print(json.dumps({'pid':p.pid,'started':True}))
