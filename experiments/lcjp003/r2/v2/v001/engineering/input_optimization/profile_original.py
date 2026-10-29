import sys,time,cProfile,pstats,json,os,copy
from pathlib import Path
sys.path.insert(0,"/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r2/v2/v001/src")
from common import *
import inputs
def main():
 seed(CONFIG["seed"]);expert=build_expert()
 work=RUN/"engineering/input_optimization";work.mkdir(exist_ok=True)
 bench=work/"profile_original";(bench/"cache").mkdir(parents=True,exist_ok=True)
 inputs.RUN=bench
 def put(name,d):
  p=bench/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(".tmp");torch.save(d,q);q.replace(p)
 inputs.atomic_torch=put
 data=inputs.Inputs(expert);data.load_sam();data.load_prior()
 e=next(e for e in ROWS if "bbq_sauce" in e["task"] and e["episode"]=="demo_0")
 prof=cProfile.Profile();torch.cuda.synchronize();start=time.perf_counter();prof.enable()
 actual=data.cache_frame(e,20)
 prof.disable();torch.cuda.synchronize();elapsed=time.perf_counter()-start
 stats=pstats.Stats(prof);prof.dump_stats(str(work/"original_profile.pstats"))
 calls=[]
 for (file,line,name),(cc,nc,tt,ct,callers) in stats.stats.items():
  if ct>.03:calls.append({"file":file,"line":line,"function":name,"calls":nc,"self_seconds":tt,"cumulative_seconds":ct})
 calls=sorted(calls,key=lambda x:x["cumulative_seconds"],reverse=True)[:45]
 old=torch.load(RUN/"cache"/e["id"]/"20.pt",map_location="cpu",weights_only=False)
 comparison={k:float((actual[k].float()-old[k].float()).abs().max()) for k in actual if torch.is_tensor(actual[k])}
 save("engineering/input_optimization/profile.json",{"time":now(),"seconds":elapsed,"comparison_to_formal_cache":comparison,"calls":calls,"concurrent_training":True})
if __name__=="__main__":
 try:main()
 except Exception as e:
  save("engineering/input_optimization/failure.json",{"time":now(),"type":type(e).__name__,"message":str(e)[:400],"frames":[{"file":x.filename,"line":x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
