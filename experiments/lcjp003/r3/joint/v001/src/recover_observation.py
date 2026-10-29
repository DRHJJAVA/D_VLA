import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,pathlib,json,shutil,traceback,time
R=pathlib.Path('/mnt/nvme1/drh/D_VLA');N=R/'experiments/lcjp003/r3/joint/v001';OLD=R/'experiments/lcjp003/r2/v2/v001';REC=N/'recovered_observations'
REC.mkdir(exist_ok=True)
for f in ['config.json','episodes.json','normalization.json']:
    if not (REC/f).exists():shutil.copy2(OLD/f,REC/f)
def protect(event,args):
    if event=='open':
        p=args[0]
        if isinstance(p,(str,bytes)):
            mode=args[1];flags=args[2];write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
            q=pathlib.Path(os.fsdecode(p)).resolve()
            if write and q.is_relative_to(R) and not q.is_relative_to(N):raise PermissionError('write_outside_new_run_blocked')
def main():
    sys.dont_write_bytecode=True;sys.path.insert(0,str(OLD/'src'))
    import common,inputs
    common.RUN=REC;inputs.RUN=REC
    sys.addaudithook(protect)
    data=inputs.Inputs(None)
    class ObservationReady(Exception):pass
    def stop_before_region_prior():raise ObservationReady()
    data.load_prior=stop_before_region_prior
    episodes=json.loads((N/'episodes.json').read_text());e=episodes[0]
    t=int(sys.argv[1]) if len(sys.argv)>1 else 118
    (REC/'cache'/e['id']).mkdir(parents=True,exist_ok=True)
    try:data.cache_frame(e,t)
    except ObservationReady:pass
    result={'observation_only':True}
    import h5py
    folder=REC/'cache'/e['id'];src=folder/'observations.h5';out=folder/('observations_'+str(t)+'.h5')
    with h5py.File(src,'r') as h,h5py.File(out,'w') as w:
        w.create_group('frames');h.copy('frames/'+str(t),w['frames'],name=str(t))
    report={'passed':True,'episode':e['id'],'frame':t,'observation':str(out),'old_run_read_only':True,'sam_grid':32,'returned_keys':list(result)}
    (N/'checks/recovery.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':
    try:main()
    except Exception as e:
        report={'type':type(e).__name__,'summary':str(e)[:300],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]}
        (N/'checks/recovery_failure.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
