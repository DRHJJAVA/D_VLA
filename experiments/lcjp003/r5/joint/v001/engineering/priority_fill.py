import sys,pathlib,json,os,time,concurrent.futures,fcntl
ROOT=pathlib.Path("/mnt/nvme1/drh/D_VLA")
RUN=ROOT/"experiments/lcjp003/r5/joint/v001"
sys.path.insert(0,str(RUN/"src"))
import prepare
DEST=RUN
TEMP=RUN/"engineering/priority_materialization"
def job(pair):
 e,t=pair;actual=DEST/"teacher"/e["id"]/(str(t)+".pt")
 if actual.exists():return 0
 prepare.RUN=TEMP
 prepare.episode(e,[t])
 source=TEMP/"teacher"/e["id"]/(str(t)+".pt")
 actual.parent.mkdir(parents=True,exist_ok=True)
 try:os.link(source,actual);return 1
 except FileExistsError:return 0
def main():
 lk=open(RUN/"engineering/priority_fill.lock","w");fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
 es={e["id"]:e for e in json.loads((RUN/"episodes.json").read_text())}
 schedule=json.loads((RUN/"sampling_schedule.json").read_text())
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  for step in range(json.loads((RUN/"state.json").read_text())["completed_updates"],3000):
   if (RUN/"STOP").exists():break
   if (RUN/"teacher_complete.json").exists():break
   pairs=[(es[x["episode"]],x["frame"]) for x in schedule[step] if not (DEST/"teacher"/x["episode"]/(str(x["frame"])+".pt")).exists()]
   if pairs:
    n=sum(pool.map(job,pairs))
    prepare.atomic(RUN/"engineering/priority_fill_state.json",{"step_inputs_ready":step+1,"new_windows":n,"pid":os.getpid(),"time":time.time()})
if __name__=="__main__":
 try:main()
 except BaseException as ex:
  prepare.atomic(RUN/"engineering/priority_fill_failure.json",{"type":type(ex).__name__,"time":time.time()})
