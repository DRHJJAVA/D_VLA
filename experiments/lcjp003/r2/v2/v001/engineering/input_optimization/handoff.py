import sys,os,json,time,signal,shutil,subprocess,fcntl,traceback
from pathlib import Path
sys.path.insert(0,"/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r2/v2/v001/src")
from common import *
WORK=RUN/"engineering/input_optimization"
def command(pid,path):
 p=Path("/proc")/str(pid)
 return p.exists() and p.stat().st_uid==os.getuid() and str(path).encode() in (p/"cmdline").read_bytes().split(b"\0")
def writing_h5(pid):
 p=Path("/proc")/str(pid)
 for fd in (p/"fd").iterdir():
  try:
   name=os.readlink(fd)
   if name.startswith(str(RUN/"cache")) and name.endswith(".h5"):
    info=(p/"fdinfo"/fd.name).read_text()
    flags=int(next(s.split()[1] for s in info.splitlines() if s.startswith("flags:")),8)
    if flags&3:return True
  except (FileNotFoundError,ProcessLookupError):pass
 return False
def main():
 lock=open(WORK/"handoff.lock","w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (WORK/"handoff.json").exists():raise RuntimeError("Handoff already recorded")
 while True:
  gates=["input_equivalence.json","training_equivalence.json","concurrency_equivalence.json"]
  if all((WORK/n).exists() for n in gates):
   assert all(json.loads((WORK/n).read_text())["passed"] for n in gates);break
  if (WORK/"failure.json").exists():raise RuntimeError("Engineering qualification failed")
  time.sleep(2)
 files=["fast_inputs.py","prefetch.py","resume.py","engineering_config.json"]
 save("engineering/input_optimization/source_engineering_lock.json",{"time":now(),"original_source_lock_sha256":sha(RUN/"source_lock.json"),"files":{n:sha(WORK/n) for n in files},"qualification":{n:sha(WORK/n) for n in gates},"unchanged":["32x32 segmentation","SAM batch64","VLM forward batch8 and order","network","loss","effective batch8","FP32","sampler sequence"]})
 oldtrain=2273893;oldparent=2273861
 assert command(oldtrain,RUN/"src/train.py") and command(oldparent,RUN/"src/pipeline.py")
 while True:
  state=json.loads((RUN/"state.json").read_text())
  n=state.get("completed_updates",-1)
  if state.get("status")=="training" and n>0 and n%10==0 and state.get("current_update")==n+1 and state.get("microbatch",8)<8:
   assert command(oldtrain,RUN/"src/train.py") and command(oldparent,RUN/"src/pipeline.py")
   os.kill(oldtrain,signal.SIGSTOP);os.kill(oldparent,signal.SIGSTOP)
   state=json.loads((RUN/"state.json").read_text())
   if state.get("completed_updates")!=n or state.get("current_update")!=n+1 or state.get("microbatch",8)>=8 or writing_h5(oldtrain):
    os.kill(oldtrain,signal.SIGCONT);os.kill(oldparent,signal.SIGCONT);time.sleep(1);continue
   c=torch.load(RUN/"checkpoints/last.pt",map_location="cpu",weights_only=False)
   assert c["completed_updates"]==n
   assert c["source_lock_sha256"]==sha(RUN/"source_lock.json") and c["config_sha256"]==sha(RUN/"config.json")
   tail=json.loads((RUN/"runtime_timings.jsonl").read_text().splitlines()[-1]);assert tail["update"]==n
   backup=WORK/("handoff_step"+str(n).zfill(4)+".pt");shutil.copy2(RUN/"checkpoints/last.pt",backup)
   digest=sha(backup);assert digest==sha(RUN/"checkpoints/last.pt")
   save("engineering/input_optimization/handoff_state_original.json",state)
   append_doc("工程输入优化资格全部通过；在第"+str(n)+"个完整更新检查点接续。新一轮尚无optimizer更新，已备份完整model/LoRA/EMA/optimizer/全部RNG及累计更新数。3个独立输入准备进程、每个4线程预处理，提前按原Python RNG采样顺序准备缓存并异步预读；原网络、损失、32×32与有效批量8不变。原src未覆盖，继承哈希见engineering/input_optimization/source_engineering_lock.json。")
   for pid in [oldparent,oldtrain]:os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
   for _ in range(50):
    if not command(oldtrain,RUN/"src/train.py"):break
    time.sleep(.2)
   assert not command(oldtrain,RUN/"src/train.py")
   log=open(WORK/"resume.log","ab")
   env=dict(os.environ,USE_TF="0",OMP_NUM_THREADS="4",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",TF_CPP_MIN_LOG_LEVEL="3")
   child=subprocess.Popen([sys.executable,str(WORK/"resume.py")],stdout=log,stderr=log,start_new_session=True,env=env)
   save("engineering/input_optimization/handoff.json",{"time":now(),"completed_updates":n,"backup":str(backup),"checkpoint_sha256":digest,"old_training_pid":oldtrain,"old_parent_pid":oldparent,"new_training_pid":child.pid,"network_loss_config_unchanged":True})
   save("pipeline_state.json",{"status":"engineering_prefetch_training_running","time":now(),"child":child.pid,"entry":str(WORK/"resume.py"),"continued_from":n})
   break
  save("engineering/input_optimization/continuation_state.json",{"status":"waiting_complete_checkpoint","time":now(),"observed_update":n})
  time.sleep(2)
if __name__=="__main__":
 try:main()
 except Exception as e:
  save("engineering/input_optimization/handoff_failure.json",{"time":now(),"type":type(e).__name__,"message":str(e)[:400],"frames":[{"file":v.filename,"line":v.lineno} for v in traceback.extract_tb(e.__traceback__)]})
  # Do not leave verified original jobs stopped if handoff failed before replacement.
  if not (WORK/"handoff.json").exists():
   for pid,path in [(2273893,RUN/"src/train.py"),(2273861,RUN/"src/pipeline.py")]:
    if command(pid,path):
     try:os.kill(pid,signal.SIGCONT)
     except ProcessLookupError:pass
