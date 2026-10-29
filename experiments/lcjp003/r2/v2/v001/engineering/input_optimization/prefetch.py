from fast_inputs import *
import multiprocessing as mp,queue,time,fcntl,atexit,traceback,random,pickle
def forecast(state,completed):
 rr=random.Random();rr.setstate(state)
 tasks=sorted(set(e["task"] for e in ROWS))
 train={t:[e for e in ROWS if e["task"]==t and e["v2_split"]=="train"] for t in tasks}
 for step in range(completed+1,CONFIG["updates"]+1):
  for micro in range(CONFIG["effective_batch"]):
   e=rr.choice(train[rr.choice(tasks)]);t=rr.randint(0,e["frames"]-17)
   yield e["id"],t,step,micro,"current"
   if t>0 and rr.random()<.5*min((step-1)/500,1)*.8:yield e["id"],t-1,step,micro,"previous_plan"
def worker(worker_id,normal,urgent,stop):
 try:
  seed(CONFIG["seed"]);expert=build_expert();data=FastInputs(expert)
  rows={e["id"]:e for e in ROWS};count=0
  while not stop.is_set():
   try:job=urgent.get_nowait()
   except queue.Empty:
    try:job=normal.get(timeout=.25)
    except queue.Empty:continue
   eid,t=job;assert eid in rows and 0<=t<rows[eid]["frames"]-16
   p=RUN/"cache"/eid/(str(t)+".pt")
   if p.exists():continue
   lp=WORK/"locks"/(eid+"_"+str(t)+".lock");lp.parent.mkdir(exist_ok=True)
   with lp.open("a") as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:continue
    if p.exists():continue
    st=time.perf_counter();data.cache_frame(rows[eid],t);elapsed=time.perf_counter()-st;count+=1
    save("engineering/input_optimization/worker"+str(worker_id)+"_state.json",{"pid":os.getpid(),"time":now(),"prepared":count,"last_seconds":elapsed,"episode":eid,"frame":t})
  data.pool.shutdown()
 except Exception as e:
  save("engineering/input_optimization/worker"+str(worker_id)+"_failure.json",{"time":now(),"type":type(e).__name__,"message":str(e)[:300],"frames":[{"file":v.filename,"line":v.lineno} for v in traceback.extract_tb(e.__traceback__)]})
  stop.set()
class PrefetchedInputs(inputs.Inputs):
 def __init__(self,expert):
  super().__init__(expert)
  cfg=json.loads((WORK/"engineering_config.json").read_text());self.settings=cfg
  context=mp.get_context("spawn");self.normal=context.Queue(maxsize=cfg["lookahead"]);self.urgent=context.Queue();self.stop=context.Event()
  self.workers=[context.Process(target=worker,args=(i,self.normal,self.urgent,self.stop),name="r2_input_"+str(i),daemon=True) for i in range(cfg["workers"])]
  for p in self.workers:p.start()
  checkpoint=torch.load(RUN/"checkpoints/last.pt",map_location="cpu",weights_only=False)
  self.begin_update=checkpoint["completed_updates"];self.forecast=list(forecast(checkpoint["rng"]["python"],self.begin_update))
  del checkpoint
  self.io=ThreadPoolExecutor(max_workers=2,thread_name_prefix="cache_read")
  self.ready={};self.io_lock=threading.Lock();self.cursor=0
  self.scheduler=threading.Thread(target=self.schedule,daemon=True);self.scheduler.start()
  atexit.register(self.close)
  save("engineering/input_optimization/prefetch_launch.json",{"time":now(),"parent_pid":os.getpid(),"workers":[p.pid for p in self.workers],"from_update":self.begin_update,"forecast_requests":len(self.forecast),"settings":cfg})
 def schedule(self):
  seen=set()
  try:
   for eid,t,*_ in self.forecast:
    if self.stop.is_set():break
    key=(eid,t)
    if key in seen:continue
    seen.add(key)
    if (RUN/"cache"/eid/(str(t)+".pt")).exists():continue
    while not self.stop.is_set():
     try:self.normal.put(key,timeout=.5);break
     except queue.Full:pass
  except Exception as e:
   save("engineering/input_optimization/scheduler_failure.json",{"time":now(),"type":type(e).__name__,"message":str(e)[:300]});self.stop.set()
 def load(self,key):
  p=RUN/"cache"/key[0]/(str(key[1])+".pt")
  return torch.load(p,map_location="cpu",weights_only=False)
 def cache_frame(self,e,t):
  begin=time.perf_counter();key=(e["id"],t);p=RUN/"cache"/e["id"]/(str(t)+".pt")
  if not p.exists():self.urgent.put(key)
  while not p.exists():
   if self.stop.is_set() or not all(w.is_alive() for w in self.workers):raise RuntimeError("Input preparation worker stopped; preserve training checkpoint")
   time.sleep(.05)
  # The training thread receives the same immutable tensor bundle and retains RNG ownership.
  with self.io_lock:future=self.ready.pop(key,None)
  result=future.result() if future is not None else self.load(key)
  # Read upcoming already-prepared bundles in parallel with forward/backward.
  state=json.loads((RUN/"state.json").read_text())
  if state.get("status")=="training" and self.cursor<len(self.forecast) and tuple(self.forecast[self.cursor][:2])==key:self.cursor+=1
  upcoming=list(dict.fromkeys(tuple(v[:2]) for v in self.forecast[self.cursor:self.cursor+16]))
  with self.io_lock:
   for old in list(self.ready):
    if old not in upcoming:self.ready.pop(old).cancel()
   for item in upcoming:
    if item not in self.ready and (RUN/"cache"/item[0]/(str(item[1])+".pt")).exists():self.ready[item]=self.io.submit(self.load,item)
  with (WORK/"consumer_timings.jsonl").open("a") as f:f.write(json.dumps({"time":now(),"episode":e["id"],"frame":t,"wait_and_read_seconds":time.perf_counter()-begin})+"\n")
  return result
 def close(self):
  if self.stop.is_set() and not any(p.is_alive() for p in self.workers):return
  self.stop.set()
  for p in self.workers:p.join(timeout=15)
  for p in self.workers:
   if p.is_alive():p.terminate();p.join(timeout=5)
  self.io.shutdown(wait=False,cancel_futures=True)
