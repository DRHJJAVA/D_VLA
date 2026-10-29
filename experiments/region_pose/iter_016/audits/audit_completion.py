from pathlib import Path
import json,hashlib,collections
import torch,numpy as np
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose/iter_016'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
for name,base in [('source_locked.json',P),('input_locked.json',P),('training_inputs_locked.json',B)]:
 d=json.loads((B/name).read_text());bad=[k for k,v in d.items() if sha(Path(k) if Path(k).is_absolute() else base/k)!=v]
 assert not bad,bad;checks[name]={'files':len(d),'mismatches':bad}
add=json.loads((B/'source_addendum_016_1.json').read_text())
assert add['original_source_lock_sha256']==sha(B/'source_locked.json')
for path,h in add['files'].items():assert sha(P/path)==h,path
checks['source_addendum_016_1']={'files':len(add['files']),'mismatches':[]}
m=json.loads((B/'manifest.json').read_text());events=sorted({r['event'] for r in m['rows'] if r['split']=='train'})
assert m['training_gate_passed']
runs=[];inits=[]
for run in ['late_object_only','late_pose_conditioned','multi_object_only','multi_pose_conditioned']:
 path=B/'runs'/run;h=json.loads((path/'history.json').read_text());done=json.loads((path/'done.json').read_text())
 assert len(h)==32 and [x['epoch'] for x in h]==list(range(1,33))
 assert all(x['optimizer_steps']==x['epoch']*len(events) for x in h)
 counts=collections.defaultdict(collections.Counter)
 for row in h:
  assert {x['event'] for x in row['selected_queries']}==set(events)
  assert len(row['selected_queries'])==len(events)
  for x in row['selected_queries']:counts[x['event']][x['offset']]+=1
 expected={1:32} if run.startswith('late_') else {15:8,10:8,5:8,1:8}
 assert all(dict(v)==expected for v in counts.values())
 init=json.loads((path/'initialization.json').read_text())['state_sha256'];inits.append(init)
 assert sha(path/'best.pt')==done['checkpoint_sha256']
 assert done['best_epoch']==min(h,key=lambda x:x['val_loss'])['epoch']
 ck=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
 assert ck['epoch']==32 and all(x in ck for x in ['optimizer','rng_numpy','rng_python','rng_cpu','rng_cuda'])
 perquery=[]
 for r in m['rows']:
  if r['split']!='val':continue
  z=np.load(B/'samples'/(r['key']+'.npz'));pred=np.load(path/'val_predictions'/(r['key']+'.npz'))
  assert np.array_equal(pred['target'],z['target']) and np.array_equal(pred['known'],z['known'])
  assert np.isfinite(pred['score']).all()
  perquery.append({'key':r['key'],'offset':r['offset'],'positive':int(z['target'].sum())})
 runs.append({'run':run,'best_epoch':done['best_epoch'],'optimizer_steps':done['optimizer_steps'],'validation':done['validation'],'queries':perquery,'schedule_verified':True})
assert len(set(inits))==1
out={'locks':checks,'all_initializations_identical':True,'matched_steps_per_run':32*len(events),'train_events':len(events),'train_queries':sum(r['split']=='train' for r in m['rows']),'val_events':m['val_episodes'],'val_queries':sum(r['split']=='val' for r in m['rows']),'excluded':m['skipped'],'runs':runs,'test_read':False,'physical_contact_ground_truth':False}
(B/'audits/completion_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({'train_events':len(events),'steps':32*len(events),'initializations_identical':True,'run_metrics':[(r['run'],r['best_epoch'],r['validation']['ap'],r['validation']['coverage20']) for r in runs]}))
