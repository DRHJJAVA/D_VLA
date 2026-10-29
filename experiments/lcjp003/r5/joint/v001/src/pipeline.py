import pathlib,json,time,os,sys,hashlib,subprocess,fcntl,traceback
ROOT=pathlib.Path("/mnt/nvme1/drh/D_VLA")
RUN=ROOT/"experiments/lcjp003/r5/joint/v001"
PY=ROOT/"experiments/lcjp003/runtime/venv/bin/python"
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def atomic(name,x):
 p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def main():
 lk=open(RUN/"pipeline.lock","w");fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
 atomic("pipeline_state.json",{"stage":"waiting_for_preflight","pid":os.getpid(),"time":time.time()})
 while True:
  if (RUN/"STOP").exists():return
  p=RUN/"checks/integration.json"
  ready=p.exists() and (RUN/"priority_ready.json").exists()
  if ready:
   check=json.loads(p.read_text())
   checked=["r5_model.py","r5_modules.py","r5_action.py","r5_geometry.py","io_runtime.py","losses.py","drifting.py"]
   ready=check.get("passed",False) and all(check["source_files"].get("src/"+n)==digest(RUN/"src"/n) for n in checked)
  if ready:break
  time.sleep(2)
 files=list((RUN/"src").glob("*.py"))+[RUN/n for n in ["config.json","episodes.json","evaluation_windows.json","normalization.json","sampling_schedule.json","priority_windows.json"]]
 old=ROOT/"experiments/lcjp003/r4/joint/v001"
 files+=list((old/"src").glob("*.py"))+[old/"input_manifest.json"]
 files+=[ROOT/"experiments/lcjp003/r3/joint/v001/src"/n for n in ["geometry.py","supervision.py","visibility.py"]]
 source={"version":"LCJP003_R5_joint_v001","files":{str(p.relative_to(ROOT)):digest(p) for p in files},"training_initializer":"official_expert_and_new_R5_modules","no_engineering_weights":True,"time":time.time()}
 lockpath=RUN/"source_lock.json"
 if lockpath.exists():assert json.loads(lockpath.read_text())["files"]==source["files"],"source_lock_changed"
 else:atomic("source_lock.json",source)
 cl=RUN/"closed_loop_object50"
 atomic("closed_loop_object50/build_lock.json",{"files":{str(p.relative_to(ROOT)):digest(p) for p in list(cl.glob("*.py"))+[cl/"task_words.pt"]},"no_video":True,"time":time.time()})
 # Do not launch another training process if a verified instance already exists.
 existing=[]
 for proc in pathlib.Path("/proc").iterdir():
  if not proc.name.isdigit():continue
  try:
   args=(proc/"cmdline").read_bytes().split(b"\0")
   if str(RUN/"src/train.py").encode() in args:existing.append(int(proc.name))
  except OSError:pass
 assert not existing,"existing_training_process"
 log=(RUN/"logs/train.log").open("ab")
 p=subprocess.Popen([str(PY),str(RUN/"src/train.py")],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launch={"training_pid":p.pid,"pipeline_pid":os.getpid(),"started":time.time(),"source_lock_sha256":digest(lockpath),"planned_updates":3000,"post_training_trials":50,"generate_video":False}
 atomic("launch.json",launch);atomic("pipeline_state.json",{"stage":"training","pid":os.getpid(),**launch})
 c=subprocess.Popen([str(PY),str(cl/"executor.py")],cwd=str(ROOT),stdout=(RUN/"logs/closed_loop.log").open("ab"),stderr=subprocess.STDOUT,start_new_session=True)
 atomic("closed_loop_object50/launch.json",{"pid":c.pid,"waits_for_update":3000,"video":False,"time":time.time()})
 rc=p.wait()
 atomic("pipeline_state.json",{"stage":"training_finished" if rc==0 else "training_failed","returncode":rc,"training_pid":p.pid,"test_executor_pid":c.pid,"time":time.time()})
if __name__=="__main__":
 try:main()
 except BaseException as ex:
  atomic("pipeline_failure.json",{"type":type(ex).__name__,"message":str(ex)[:200],"frames":[{"file":f.filename,"line":f.lineno} for f in traceback.extract_tb(ex.__traceback__)],"time":time.time()})
  raise SystemExit(1)
