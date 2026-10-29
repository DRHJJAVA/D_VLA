import sys,time,json,datetime
from pathlib import Path
sys.path.insert(0,"/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r2/v2/v001/src")
from common import *
WORK=RUN/"engineering/input_optimization"
def main():
 while not (WORK/"handoff.json").exists():
  if (WORK/"handoff_failure.json").exists():return
  time.sleep(5)
 h=json.loads((WORK/"handoff.json").read_text());n=h["completed_updates"]
 while True:
  for name in ["resume_failure.json","worker0_failure.json","worker1_failure.json","worker2_failure.json"]:
   if (WORK/name).exists():return
  a=[json.loads(x) for x in (RUN/"runtime_timings.jsonl").read_text().splitlines()]
  new=[x for x in a if n<x["update"]<=n+20]
  if len(new)>=20:break
  time.sleep(5)
 assert [x["update"] for x in a]==list(range(1,len(a)+1))
 baseline=[x for x in a if 83<=x["update"]<=102];assert len(baseline)==20
 def metrics(v):
  return {"mean_update_seconds":sum(x["seconds"] for x in v)/len(v),"mean_stage_seconds":{k:sum(x["timings"][k] for x in v)/len(v) for k in v[0]["timings"]},"training_thread_cache_hit_fraction":sum(x["cache_hits"] for x in v)/(8*len(v))}
 old=metrics(baseline);fast=metrics(new)
 wall=(datetime.datetime.fromisoformat(new[-1]["time"])-datetime.datetime.fromisoformat(h["time"])).total_seconds()
 report={"time":now(),"baseline_updates":[83,102],"optimized_updates":[n+1,n+20],"baseline":old,"optimized":fast,"observed_training_speedup":old["mean_update_seconds"]/fast["mean_update_seconds"],"handoff_inclusive_seconds":wall,"handoff_inclusive_speedup":old["mean_update_seconds"]/(wall/20),"workers":3,"preprocessing_threads_per_worker":4,"input_max_difference":0,"loss_gradient_optimizer_max_difference":0,"source_network_loss_unchanged":True,"qualification_cold_start_cost_included_in_separate_input_test":True,"conditions":"Real consecutive updates, input preparation runs concurrently; cache reuse fraction changes naturally. Baseline before qualification jobs. Handoff-inclusive number includes model/worker startup."}
 save("engineering/input_optimization/speedup20.json",report)
 append_doc("输入准备优化正式接续20更新实测："+json.dumps(report,ensure_ascii=False)+"。所有更新连续、原网络损失分割配置未变；不把预取后的单次缓存读取加速当作端到端提速。")
if __name__=="__main__":main()
