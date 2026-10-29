from fast_inputs import *
import time,traceback
def main():
 seed(CONFIG["seed"]);expert=build_expert();work=WORK/"paired_inputs";work.mkdir(exist_ok=True)
 original=inputs.Inputs(expert);original.load_sam();original.load_prior()
 fast=FastInputs(expert,work/"optimized");fast.sam=original.sam;fast.prior=batched_query(original.prior,fast.pool)
 cases=[]
 for task in sorted(set(e["task"] for e in ROWS))[:8]:
  found=None
  for e in ROWS:
   if e["task"]!=task or e["v2_split"]!="train":continue
   paths=sorted((RUN/"cache"/e["id"]).glob("[0-9]*.pt"))
   if paths:found=(e,int(paths[0].stem));break
  assert found;cases.append(found)
 results=[]
 root_original=work/"original";(root_original/"cache").mkdir(parents=True,exist_ok=True)
 inputs.RUN=root_original
 def put(name,d):
  p=root_original/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(".tmp");torch.save(d,tmp);tmp.replace(p)
 inputs.atomic_torch=put
 for index,(e,t) in enumerate(cases):
  times={};data={}
  # Alternate order to reduce bias from shared GPU contention.
  for name,obj in ([("original",original),("optimized",fast)] if index%2==0 else [("optimized",fast),("original",original)]):
   torch.cuda.synchronize();start=time.perf_counter();data[name]=obj.cache_frame(e,t);torch.cuda.synchronize();times[name]=time.perf_counter()-start
  diff={k:float((data["original"][k].float()-data["optimized"][k].float()).abs().max()) for k in data["original"] if torch.is_tensor(data["original"][k])}
  assert all(v==0 for v in diff.values()),("input_mismatch",diff)
  gold=torch.load(RUN/"cache"/e["id"]/(str(t)+".pt"),map_location="cpu",weights_only=False)
  gold_diff={k:float((gold[k].float()-data["optimized"][k].float()).abs().max()) for k in gold if torch.is_tensor(gold[k])}
  assert all(v==0 for v in gold_diff.values()),("formal_cache_mismatch",gold_diff)
  results.append({"episode":e["id"],"frame":t,"candidates":len(data["optimized"]["candidate_ids"]),"seconds":times,"max_difference":max(diff.values()),"formal_cache_max_difference":max(gold_diff.values())})
  save("engineering/input_optimization/paired_progress.json",{"completed":len(results),"results":results,"time":now()})
 summary={"passed":True,"time":now(),"cases":results,"original_seconds":sum(x["seconds"]["original"] for x in results),"optimized_seconds":sum(x["seconds"]["optimized"] for x in results),"network_batch_size":8,"sam_points_per_batch":64,"cpu_preprocessing_threads":4,"same_model_batches_and_order":True,"concurrent_training":True}
 summary["input_speedup"]=summary["original_seconds"]/summary["optimized_seconds"]
 save("engineering/input_optimization/input_equivalence.json",summary)
 fast.pool.shutdown()
if __name__=="__main__":
 try:main()
 except Exception as e:save("engineering/input_optimization/failure.json",{"time":now(),"stage":"input_equivalence","type":type(e).__name__,"message":str(e)[:500],"frames":[{"file":x.filename,"line":x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
