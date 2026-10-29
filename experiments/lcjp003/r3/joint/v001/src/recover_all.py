import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,pathlib,json,time,shutil,traceback,h5py,fcntl
R=pathlib.Path('/mnt/nvme1/drh/D_VLA');N=R/'experiments/lcjp003/r3/joint/v001';OLD=R/'experiments/lcjp003/r2/v2/v001';REC=N/'recovered_observations'
def save(name,data):
 p=N/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(p)
def main():
 lock=(N/'recovery.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 sys.dont_write_bytecode=True;sys.path.insert(0,str(OLD/'src'))
 import common,inputs
 common.RUN=REC;inputs.RUN=REC
 from recover_observation import protect
 sys.addaudithook(protect)
 data=inputs.Inputs(None)
 class ObservationReady(Exception):pass
 def stop():raise ObservationReady()
 data.load_prior=stop
 episodes=json.loads((N/'episodes.json').read_text());missing=[]
 for e in episodes:
  for p in (OLD/'cache'/e['id']).glob('*.pt'):
   if not p.stem.isdigit():continue
   dest=REC/'cache'/e['id']/('observations_'+p.stem+'.h5')
   if not (p.parent/('observations_'+p.stem+'.h5')).exists() and not dest.exists():missing.append((e,int(p.stem)))
 start=time.time();times=[]
 for j,(e,t) in enumerate(missing):
  if (N/'STOP').exists():save('recovery_state.json',{'status':'stopped_by_request','done':j,'total':len(missing)});return
  tt=time.time();folder=REC/'cache'/e['id'];folder.mkdir(parents=True,exist_ok=True)
  try:data.cache_frame(e,t)
  except ObservationReady:pass
  with h5py.File(folder/'observations.h5','r') as h,h5py.File(folder/('observations_'+str(t)+'.h5'),'w') as w:
   w.create_group('frames');h.copy('frames/'+str(t),w['frames'],name=str(t))
  times.append(time.time()-tt)
  save('recovery_state.json',{'status':'recovering_missing_observations','done':j+1,'total':len(missing),'episode':e['id'],'frame':t,'elapsed':time.time()-start,'last_seconds':times[-1],'mean_seconds':sum(times)/len(times),'eta_seconds':(len(missing)-j-1)*sum(times)/len(times),'pid':os.getpid(),'old_run_read_only':True})
 save('checks/input_completeness.json',{'passed':True,'recovered_frames':len(missing),'seconds':time.time()-start,'existing_observations_unchanged':True,'episode_count':len(episodes)})
 save('recovery_state.json',{'status':'complete','total':len(missing),'done':len(missing),'seconds':time.time()-start})
if __name__=='__main__':
 try:main()
 except Exception as e:
  save('recovery_all_failure.json',{'type':type(e).__name__,'summary':str(e)[:300],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]})
  sys.exit(1)
