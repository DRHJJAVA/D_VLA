from pathlib import Path
import hashlib,json,numpy as np,torch
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf009')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((R/'config.json').read_text())
rows=[];lockchecks={}
for name in ['input_locked.json','source_locked.json','source_addendum_locked.json']:
 v=json.loads((R/name).read_text());assert all(sha(p)==s for p,s in v.items());lockchecks[name]=len(v)
for name in ['cf005','cf005_frozen_review_003','cf006','cf007','cf008']:
 v=json.loads((R.parent/name/'completed_locked.json').read_text());assert all(sha(p)==s for p,s in v.items());lockchecks[name]=len(v)
for arm in cfg['arms']:
 a=R/arm
 for n in ['training_inputs_locked.json','evaluation_weights_locked.json']:
  v=json.loads((a/n).read_text());assert all(sha(p)==s for p,s in v.items());lockchecks[arm+'/'+n]=len(v)
 for seed in cfg['seeds']:
  dest=a/'runs/evidence'/('seed_'+str(seed))
  init=torch.load(dest/'initial.pt',map_location='cpu',weights_only=False)
  prev=torch.load(R.parent/'cf008/runs/evidence'/('seed_'+str(seed))/'initial.pt',map_location='cpu',weights_only=False)
  assert init['model'].keys()==prev['model'].keys()
  assert all(torch.equal(v,prev['model'][k]) for k,v in init['model'].items())
  last=torch.load(dest/'last.pt',map_location='cpu',weights_only=False)
  hist=json.loads((dest/'history.json').read_text());assert len(hist)==600 and last['steps']==600
  steps=[int(v['step'].item()) if hasattr(v['step'],'item') else int(v['step']) for v in last['optimizer']['state'].values()]
  assert max(steps)==600
  assert (a/'plans'/('seed_'+str(seed)+'.json')).read_bytes()==(R.parent/'cf008/plans'/('seed_'+str(seed)+'.json')).read_bytes()
  rows.append({'arm':arm,'seed':seed,'steps':600,'optimizer_steps':max(steps),'all_initial_parameters_equal_cf008':True,'plan_equal_cf008':True,'last_sha256':sha(dest/'last.pt')})
 for ep in cfg['train_episodes']:
  d=np.load(a/'data'/ep/'artifacts/dataset.npz');o=np.load(R.parent/'cf008/data'/ep/'artifacts/dataset.npz')
  for k in ['images','points','states','near_target','near_weight','dy','dynvalid','visual_transforms']:
   assert np.array_equal(d[k],o[k],equal_nan=True)
 # ensure scored states exactly reconstruct from original decision policy
 for ep in cfg['train_episodes']+cfg['fixed_review_episodes']:
  p=np.load(a/'evaluation'/ep/'artifacts/frame_predictions.npz');d=np.load(a/'evaluation'/ep/'artifacts/dataset.npz')
  sep=d['geometry_reliable']&d['state_valid']&(d['distance']-2*d['sigma']>.006)
  pos=(p['probabilities'].min(0)>=.8)&d['geometry_reliable']&d['state_valid']&~sep
  neg=(p['proximity_probabilities'].max(0)<=.2)&sep
  assert np.array_equal(pos,p['contact_candidate']) and np.array_equal(neg,p['observed_separation_candidate'])
  assert np.array_equal(~(pos|neg),p['unknown'])
audit={'passed':True,'training_runs':rows,'lock_checks':lockchecks,'inputs_equal_except_registered_supervision':True,'all_decisions_reconstructed':True,'human_ranges_training_labels':False}
(R/'audits/completion.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({'passed':True,'runs':len(rows),'locks':lockchecks}))
