import pathlib,sys,json,os,time,fcntl,concurrent.futures,traceback
RUN=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r4/joint/v001')
sys.path.insert(0,str(RUN/'src'))
from data import prepare_one,obs_file,ROOT
def run(job):
 e,t=job
 try:
  if not (ROOT/'experiments/lcjp003/r2/v2/v001/cache'/e['id']/(str(t)+'.pt')).exists():return 'unavailable'
  obs_file(e,t);prepare_one((e,t));return 'ready'
 except FileNotFoundError:return 'unavailable'
if __name__=='__main__':
 lock=open(RUN/'engineering/prefetch.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ep={e['id']:e for e in json.loads((RUN/'episodes.json').read_text())}
 schedule=json.loads((RUN/'sampling_schedule.json').read_text())
 start=json.loads((RUN/'state.json').read_text())['completed_updates']
 jobs=list(dict.fromkeys((x['episode'],x['frame']-1) for batch in schedule[start:] for x in batch if x['frame']>0))
 count=0;skipped=0;t0=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
  for status in pool.map(run,[(ep[e],t) for e,t in jobs],chunksize=1):
   count+=1;skipped+=status=='unavailable'
   if count%20==0:
    p=RUN/'engineering/prefetch_v2_state.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps({'pid':os.getpid(),'processed':count,'total':len(jobs),'missing_observation':skipped,'seconds':time.time()-t0,'source_and_sampling_unchanged':True}));tmp.replace(p)
   if (RUN/'STOP').exists():break
