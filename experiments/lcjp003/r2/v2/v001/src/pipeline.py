from common import *
import time,subprocess,fcntl
def main():
 lock=open(RUN/"pipeline.lock","w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=RUN/"engineering/validation.json"
 while not p.exists():
  if (RUN/"failure.json").exists():raise RuntimeError("Implementation validation failed; automatic training launch blocked")
  save("pipeline_state.json",{"status":"waiting_for_network_validation","time":now(),"pid":os.getpid()})
  time.sleep(5)
 v=json.loads(p.read_text());assert v["passed"] and v["resume_parameter_max"]==0 and v["formal_optimizer_updates"]==0
 files=list((RUN/"src").glob("*.py"))+[V1/"v0_corrected_check.py",V1/"smolvlm_with_expert_official.py",V1/"fusion_interface/fusion_module.py",V1/"segmentation_coverage.py",V1/"v1_check.py",ROOT/"experiments/lcjp003/src/data.py",ROOT/"experiments/lcjp003/src/visibility.py"]
 manifest={"time":now(),"files":{str(f.relative_to(ROOT)):sha(f) for f in files},"config_sha256":sha(RUN/"config.json"),"episodes_sha256":sha(RUN/"episodes.json"),"normalization_sha256":sha(RUN/"normalization.json"),"validation_sha256":sha(p),"frozen_weights":{}}
 for f in [ROOT/"experiments/lcjp003/weights/smolvla_base/model.safetensors",ROOT/"experiments/contact_joint_object/dependencies/sam21_tiny.pt"]:
  manifest["frozen_weights"][str(f.relative_to(ROOT))]=sha(f)
 for f in (ROOT/"experiments/lcjp003/weights/smolvlm2").glob("*.safetensors"):
  manifest["frozen_weights"][str(f.relative_to(ROOT))]=sha(f)
 save("source_lock.json",manifest)
 # The OS lock is authoritative; PID metadata is informational only.
 log=open(RUN/"logs/train.log","ab")
 env=dict(os.environ,OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",USE_TF="0",TF_CPP_MIN_LOG_LEVEL="3")
 worker=subprocess.Popen([sys.executable,str(RUN/"src/train.py")],stdout=log,stderr=log,env=env)
 save("launch.json",{"time":now(),"pipeline_pid":os.getpid(),"training_pid":worker.pid,"python":sys.executable,"entry":str(RUN/"src/train.py"),"config_sha256":manifest["config_sha256"],"source_lock_sha256":sha(RUN/"source_lock.json")})
 save("pipeline_state.json",{"status":"training_running","time":now(),"pid":os.getpid(),"child":worker.pid})
 code=worker.wait()
 save("pipeline_state.json",{"status":"training_process_completed" if code==0 else "training_process_failed","time":now(),"returncode":code,"pid":os.getpid()})
if __name__=="__main__":
 try:main()
 except Exception as e:fail("V2_pipeline",e);raise SystemExit(1)
