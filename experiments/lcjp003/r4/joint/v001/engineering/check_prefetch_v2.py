import pathlib,sys,json,time,torch,hashlib
RUN=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r4/joint/v001')
sys.path.insert(0,str(RUN/'src'))
from data import get_frame,load_sample,atomic_json,ROOT
ep={e['id']:e for e in json.loads((RUN/'episodes.json').read_text())}
records=json.loads((RUN/'evaluation_windows.json').read_text())
out=[]
for rec in [next(x for x in records if x['split']=='train'),next(x for x in records if x['split']=='dev')]:
 stored=load_sample(rec);fresh=get_frame(ep[rec['episode']],rec['frame']);diff={};equal=True
 for group in ['inputs','labels']:
  for k,a in fresh[group].items():
   b=stored[group][k];same=torch.equal(a,b);equal &= same
   diff[group+'.'+k]=0 if same else float((a.float()-b.float()).abs().nan_to_num(nan=float('inf')).max())
 out.append({'episode':rec['episode'],'frame':rec['frame'],'all_arrays_exact':equal,'maximum_difference':max(diff.values())})
locked=json.loads((RUN/'source_lock.json').read_text())
unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in locked['files'].items())
report={'time':time.time(),'samples':out,'source_lock_unchanged':unchanged,'parallelism_only':True,'passed':all(x['all_arrays_exact'] for x in out) and unchanged}
atomic_json('engineering/prefetch_v2_consistency.json',report)
print(json.dumps(report))
