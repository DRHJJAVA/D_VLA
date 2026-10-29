import sys,pathlib,json,time,os,datetime,hashlib,traceback,subprocess,ast,multiprocessing as mp,queue,fcntl
W=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(W))
import prefetch as p
R=p.RUN
def save(n,d):
 t=W/(n+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(W/n)
def qualify(data):
 old=data.rootdir
 q=json.loads((W/'input_equivalence.json').read_text())['cases'][1]
 eid,t=q['episode'],q['frame']
 ref=p.torch.load(R/'cache'/eid/(str(t)+'.pt'),map_location='cpu',weights_only=False)
 e=next(x for x in json.loads((R/'episodes.json').read_text()) if x['id']==eid)
 data.rootdir=W/'four_worker_qualification'
 (data.rootdir/'cache').mkdir(parents=True,exist_ok=True)
 got=data.cache_frame(e,t)
 data.rootdir=old
 def eq(a,b):
  if isinstance(a,p.torch.Tensor):return p.torch.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
  if isinstance(a,(tuple,list)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
  return a==b
 assert eq(ref,got),'concurrent input mismatch'
 save('four_worker_equivalence.json',{'passed':True,'time':p.now(),'all_input_max_difference':0,'episode':eid,'frame':t})
def child(normal,urgent,stop):
 try:
  tree=ast.parse((W/'prefetch.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='worker')
  class Insert(ast.NodeTransformer):
   def visit_Assign(self,node):
    if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='FastInputs':
     return [node,ast.Expr(ast.Call(ast.Name('qualify',ast.Load()),[ast.Name('data',ast.Load())],[]))]
    return node
  fn=Insert().visit(fn);mod=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));p.__dict__['qualify']=qualify;exec(compile(mod,str(W/'prefetch.py'),'exec'),p.__dict__)
  p.worker(3,normal,urgent,stop)
 except Exception as e:
  save('four_worker_failure.json',{'type':type(e).__name__,'frames':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]});stop.set()
def main():
 lk=(W/'four_worker_trial.lock').open('w');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
 mem=float(subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip())
 assert mem>12000,'insufficient memory margin'
 c=p.torch.load(R/'checkpoints/last.pt',map_location='cpu',weights_only=False)
 state=json.loads((R/'state.json').read_text());start=state['completed_updates']
 requests=list(p.forecast(c['rng']['python'],c['completed_updates']))
 ctx=mp.get_context('spawn');normal=ctx.Queue(24);urgent=ctx.Queue();stop=ctx.Event()
 ch=ctx.Process(target=child,args=(normal,urgent,stop));ch.start()
 begin=time.time()
 save('four_worker_trial_launch.json',{'time':p.now(),'parent_pid':os.getpid(),'worker_pid':ch.pid,'start_update':start,'end_update':start+20,'memory_free_before_MiB':mem,'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()})
 i=0;seen=set()
 try:
  while time.time()-begin<1200:
   state=json.loads((R/'state.json').read_text())
   if state['completed_updates']>=start+20 or stop.is_set() or not ch.is_alive():break
   while i<len(requests):
    eid,t,step,micro,role=requests[i]
    if step<=state['completed_updates'] or (eid,t) in seen or (R/'cache'/eid/(str(t)+'.pt')).exists():i+=1;continue
    try:normal.put_nowait((eid,t));seen.add((eid,t));i+=1
    except queue.Full:break
   time.sleep(.5)
 finally:
  stop.set();ch.join(35)
  if ch.is_alive():ch.terminate();ch.join(10)
 rows=[json.loads(x) for x in (R/'runtime_timings.jsonl').read_text().splitlines()]
 before=[x for x in rows if start-19<=x['update']<=start]
 after=[x for x in rows if start<x['update']<=start+20]
 def summary(xs):
  return {'n':len(xs),'mean_seconds':sum(x['seconds'] for x in xs)/len(xs),'mean_stages':{k:sum(x['timings'][k] for x in xs)/len(xs) for k in xs[0]['timings']}}
 out={'time':p.now(),'start_update':start,'end_update':state['completed_updates'],'before':summary(before),'after':summary(after) if after else None,'worker_exitcode':ch.exitcode,'worker_stopped':not ch.is_alive(),'network_loss_inputs_unchanged':True,'external_load_not_controlled':True}
 if after:out['observed_speedup']=out['before']['mean_seconds']/out['after']['mean_seconds']
 save('four_worker_trial_result.json',out)
if __name__=='__main__':
 try:main()
 except Exception as e:save('four_worker_controller_failure.json',{'type':type(e).__name__,'frames':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
