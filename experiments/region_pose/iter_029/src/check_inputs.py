from pathlib import Path
import sys,json,hashlib,numpy as np
B=Path(__file__).resolve().parents[1];R=B.parents[2]
sys.path.insert(0,str(R/'src'))
from build_point_graphs import fps
from supervision import targets
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for lock in ['source_locked.json','input_locked.json']:
 for f,h in json.loads((B/lock).read_text()).items():assert sha(Path(f) if Path(f).is_absolute() else R/f)==h,f
m=json.loads((B/'manifest.json').read_text());checks=[]
for row in m['rows']:
 z=np.load(B/'samples'/(row['key']+'.npz'));a=np.load(row['target_source'])
 sel=fps(a['query_xyz'],2048) if len(a['query_xyz'])>2048 else np.arange(len(a['query_xyz']))
 assert np.array_equal(sel,z['causal_fps_indices'])
 y,k,c,idx=targets(row,a)
 for name,arr in [('target',y),('known',k),('confidence',c),('reference_source_index',idx)]:
  assert np.array_equal(arr[sel],z[name]),(row['key'],name)
 assert np.all(z['confidence'][~z['known']]==0)
 assert np.array_equal(z['input_rgb_pixel'],a['query_pixel'])
 assert np.array_equal(z['pixel'],a['query_pixel'][sel])
 assert np.array_equal(z['xyz_world'],a['query_xyz'][sel].astype(np.float32))
 assert np.isfinite(z['dino_feat']).all() and np.isfinite(z['pose']).all()
 assert row['query_frame']<row['closed_command_start_frame']
 assert len(np.unique(z['reference_source_index'][z['target']]))>=3
 assert len(np.unique(z['reference_source_index'][z['known']&~z['target']]))>=16
 checks.append({'key':row['key'],'offset':row['offset'],'fps_causal':True,'confidence_recomputed':True,'query_strict_preclose':True})
for sp in ['train','val']:
 for ev in {r['event'] for r in m['rows'] if r['split']==sp}:
  assert sorted(r['offset'] for r in m['rows'] if r['event']==ev)==[1,5,10,15]
tr={r['episode'] for r in m['rows'] if r['split']=='train'};va={r['episode'] for r in m['rows'] if r['split']=='val'};assert not tr&va
(B/'audits/preflight.json').write_text(json.dumps({'checks':checks,'train_events':len(tr),'val_events':len(va),'training_gate':m['training_gate_passed']},indent=2))
print('PREFLIGHT_PASSED',len(checks),flush=True)

import torch,random
import train as trainer
from model import RegionPoseNet
cfg=json.loads((B/'config.json').read_text())
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
net=RegionPoseNet().cuda()
state_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
torch.save({'state':net.state_dict(),'seed':cfg['seed']},B/'initialization/initial.pt')
(B/'initialization/locked.json').write_text(json.dumps({'state_sha256':state_hash,'checkpoint_sha256':sha(B/'initialization/initial.pt'),'before_optimization':True},indent=2))
sample=np.load(B/'samples'/(m['rows'][0]['key']+'.npz'))
d={k:torch.as_tensor(sample[k],device='cuda',dtype=torch.float32) for k in ['target','confidence']}
d['known']=torch.as_tensor(sample['known'],device='cuda',dtype=torch.bool)
logit=torch.zeros(len(d['target']),device='cuda',requires_grad=True)
trainer.loss(logit,d).backward()
assert torch.count_nonzero(logit.grad[~d['known']])==0
plans={}
for run in cfg['runs']:
 chosen=[r for r in m['rows'] if r['split']=='train' and (run['group']=='expanded10' or r['episode'] in cfg['control_episodes'])]
 events=sorted({r['event'] for r in chosen});assert len(events)==run['events']
 rng=random.Random(cfg['seed']);steps=[]
 for ep in range(1,run['epochs']+1):
  order=events.copy();rng.shuffle(order)
  for ev in order:
   off=[15,10,5,1][(ep-1+events.index(ev))%4]
   row=next(r for r in chosen if r['event']==ev and r['offset']==off)
   steps.append(dict(epoch=ep,event=ev,offset=off,key=row['key']))
 assert len(steps)==240
 for ev in events:
  for off in [15,10,5,1]:assert sum(s['event']==ev and s['offset']==off for s in steps)==run['epochs']//4
 name=run['group']+'_'+run['variant'];plans[name]=steps
 (B/'plans'/(name+'.json')).write_text(json.dumps(steps,indent=2))
(B/'audits/optimizer_preflight.json').write_text(json.dumps({'initialization_sha256':state_hash,'unknown_gradient_zero':True,'runs':{n:len(v) for n,v in plans.items()},'same_initialization_all_runs':True,'equal_event_exposure_between_groups':False},indent=2))
print('OPTIMIZER_PREFLIGHT_PASSED',flush=True)
