from prefetch import *
import train as training
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
def restored(front,expert,ema,opt,c,completed):
 checks={"front":equal(front.state_dict(),c["front"]),"lora":all(torch.equal(dict(expert.named_parameters())[k],v) for k,v in c["lora"].items()),"ema":equal(ema.state_dict(),c["ema"]),"optimizer":equal(opt.state_dict(),c["optimizer"]),"rng":equal(rng_state(),c["rng"]),"updates":int(completed)==int(c["completed_updates"])}
 assert all(checks.values()),checks
 save("engineering/input_optimization/resumed.json",{"time":now(),"pid":os.getpid(),"completed_updates":completed,"exact_restore":checks,"checkpoint_sha256":sha(RUN/"checkpoints/last.pt")})
def main():
 lock=json.loads((WORK/"source_engineering_lock.json").read_text())
 for name,h in lock["files"].items():assert sha(WORK/name)==h,("engineering_source_changed",name)
 assert json.loads((WORK/"input_equivalence.json").read_text())["passed"]
 assert json.loads((WORK/"training_equivalence.json").read_text())["passed"]
 original_check=training.check_sources
 def check():
  original_check()
  for name,h in lock["files"].items():assert sha(WORK/name)==h,("engineering_source_changed",name)
 training.check_sources=check;training.Inputs=PrefetchedInputs
 original_checkpoint=training.checkpoint
 def checkpoint(*a,**kw):
  c=original_checkpoint(*a,**kw);c["input_engineering_sha256"]=sha(WORK/"source_engineering_lock.json");return c
 training.checkpoint=checkpoint;training.assert_restored=restored
 tree=ast.parse((RUN/"src/train.py").read_text());fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main"))
 oldloop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="step")
 before=ast.dump(oldloop,include_attributes=False)
 index=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.If) and any(isinstance(x,ast.Name) and x.id=="last" for x in ast.walk(n.test)))
 fn.body.insert(index+1,ast.parse("assert_restored(front,expert,ema,opt,c,completed)").body[0])
 assert ast.dump(next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="step"),include_attributes=False)==before
 fn.name="engineering_resume_main"
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),"<original_train_loop_with_verified_prefetch>", "exec"),training.__dict__)
 training.engineering_resume_main()
if __name__=="__main__":
 try:main()
 except Exception as e:
  save("engineering/input_optimization/resume_failure.json",{"time":now(),"type":type(e).__name__,"message":str(e)[:500],"frames":[{"file":v.filename,"line":v.lineno} for v in traceback.extract_tb(e.__traceback__)]})
  save("state.json",{"status":"engineering_resume_failed","time":now(),"failure":"engineering/input_optimization/resume_failure.json"})
  raise SystemExit(1)
