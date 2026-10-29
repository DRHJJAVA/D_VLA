import time,json,torch,numpy as np
from io_runtime import RUN,load_record,move,atomic
def evaluate(model,step):
 records=json.loads((RUN/"evaluation_windows.json").read_text())
 norm=json.loads((RUN/"normalization.json").read_text())
 scale=torch.tensor(norm["std"],device="cuda")
 rows=[];start=time.perf_counter();model.eval()
 with torch.no_grad():
  for idx,rec in enumerate(records):
   b=move(load_record(rec));e=model.encode(b["d"],b["history"],b["history_valid"])
   gen=torch.Generator(device="cpu").manual_seed(20261029+idx)
   noise=torch.randn(8,16,6,generator=gen).cuda()
   motion,grip=model.generate(e,noise);err=(motion-b["y"]["target"][None]).abs()
   valid=e["neighbor_valid"].any(-1)[None];score=e["score"]
   changed={}
   for name,s in [("uniform",torch.where(valid,score.sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1),0)),("reverse",torch.where(valid,1-score,0)),("shuffle",score.flip(-1)*valid)]:
    alt=model.rescore(e,s);am,ag=model.generate(alt,noise)
    changed[name]={"motion_delta":float((am-motion).abs().mean()),"motion_mae":float((am-b["y"]["target"][None]).abs().mean()),"grip_flip_rate":float((ag.argmax(-1)!=grip.argmax(-1)).float().mean()),"grip_error":float((ag.argmax(-1)!=b["y"]["grip"][None]).float().mean())}
   rows.append({**rec,"motion_mae":float(err.mean()),"translation_raw_mae":float((err[...,:3]*scale[:3]).mean()),"rotation_raw_mae":float((err[...,3:]*scale[3:]).mean()),"grip_error":float((grip.argmax(-1)!=b["y"]["grip"][None]).float().mean()),"target_null":int(e["target_index"])==16,"valid_points":int(e["points"]["valid"].sum()),"score_mean":float(score.mean()),"score_std":float(score.std()),"score_min":float(score.min()),"score_max":float(score.max()),"lambda_mean":float(e["lambda"].mean()),"interventions":changed})
   atomic("evaluation_state.json",{"step":step,"completed":idx+1,"total":len(records),"time":time.time()})
 summary={}
 for split in ["train","dev"]:
  rr=[x for x in rows if x["split"]==split or (split=="dev" and x["split"]=="development")]
  if not rr:continue
  keys=["motion_mae","translation_raw_mae","rotation_raw_mae","grip_error","target_null","valid_points","score_mean","score_std","score_min","score_max","lambda_mean"]
  summary[split]={k:float(np.mean([x[k] for x in rr])) for k in keys};summary[split]["windows"]=len(rr)
  summary[split]["interventions"]={name:{k:float(np.mean([x["interventions"][name][k] for x in rr])) for k in ["motion_delta","motion_mae","grip_flip_rate","grip_error"]} for name in ["uniform","reverse","shuffle"]}
 result={"step":step,"fixed_windows":len(rows),"summary":summary,"rows":rows,"seconds":time.perf_counter()-start,"time":time.time(),"success_rate":None,"note":"offline action errors and interventions; not closed-loop success"}
 atomic("eval%04d.json"%step,result);model.train();return result
