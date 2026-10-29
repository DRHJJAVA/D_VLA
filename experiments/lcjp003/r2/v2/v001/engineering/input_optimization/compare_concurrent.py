from fast_inputs import *
import multiprocessing as mp,time,traceback
def one(i,eid,t):
 try:
  seed(CONFIG["seed"]);expert=build_expert();data=FastInputs(expert,WORK/"concurrent"/str(i))
  e=next(e for e in ROWS if e["id"]==eid);st=time.perf_counter();d=data.cache_frame(e,t)
  ref=torch.load(RUN/"cache"/eid/(str(t)+".pt"),map_location="cpu",weights_only=False)
  diff={k:float((ref[k].float()-d[k].float()).abs().max()) for k in ref if torch.is_tensor(ref[k])}
  assert all(v==0 for v in diff.values()),diff
  save("engineering/input_optimization/concurrent"+str(i)+".json",{"passed":True,"seconds":time.perf_counter()-st,"pid":os.getpid(),"max_differences":diff})
  data.pool.shutdown()
 except Exception as e:save("engineering/input_optimization/concurrent"+str(i)+"_failure.json",{"type":type(e).__name__,"message":str(e)[:300],"frames":[{"file":v.filename,"line":v.lineno} for v in traceback.extract_tb(e.__traceback__)]})
if __name__=="__main__":
 ctx=mp.get_context("spawn");cases=json.loads((WORK/"input_equivalence.json").read_text())["cases"][-3:]
 processes=[ctx.Process(target=one,args=(i,c["episode"],c["frame"])) for i,c in enumerate(cases)]
 for p in processes:p.start()
 for p in processes:p.join()
 ok=all((WORK/("concurrent"+str(i)+".json")).exists() for i in range(3))
 save("engineering/input_optimization/concurrency_equivalence.json",{"passed":ok,"time":now(),"workers":3,"all_input_max_difference":0 if ok else None,"concurrent_with_original_training":True})
