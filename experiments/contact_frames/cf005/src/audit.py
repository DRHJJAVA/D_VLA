from pathlib import Path
import json,hashlib,numpy as np,torch,datetime
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/contact_frames/cf005';C=B.parent/'cf003'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=json.loads((B/'config.json').read_text());lock=json.loads((B/'training_inputs_locked.json').read_text());assert all(sha(Path(p))==h for p,h in lock.items())
assert (B/'src/model_all_train_frames.py').read_bytes()==(C/'src/model_no_world.py').read_bytes()
runs=[]
for seed in cfg['seeds']:
 out=B/f'runs/all_train_frames/seed_{seed}';h=json.loads((out/'history.json').read_text());plan=json.loads((B/f'plans/seed_{seed}.json').read_text())
 assert len(h)==len(plan)==600 and all(row['step']==j+1 and row['episode']==plan[j]['episode'] for j,row in enumerate(h))
 ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);init=torch.load(C/f'runs/no_world/seed_{seed}/initial.pt',map_location='cpu',weights_only=False)
 assert ck['steps']==600 and ck['initial_tensor_hash']==init['initial_tensor_hash']
 steps=sorted(set(float(v['step']) for v in ck['optimizer']['state'].values()));assert steps==[600.]
 runs.append({'seed':seed,'steps':600,'initial_tensor_hash':ck['initial_tensor_hash'],'last_sha256':sha(out/'last.pt'),'optimizer_steps':steps})
rows=[]
for ep in cfg['fixed_review_episodes']:
 oldpath=C/'evaluation_no_world'/ep/'artifacts';newpath=B/'evaluation_all_train_frames'/ep/'artifacts'
 assert sha(oldpath/'dataset.npz')==sha(newpath/'dataset.npz')
 a=np.load(oldpath/'frame_predictions.npz');z=np.load(newpath/'frame_predictions.npz');d=np.load(newpath/'dataset.npz')
 assert np.array_equal(a['context_indices'],z['context_indices'])
 rel=d['geometry_reliable']&d['state_valid'];pr=z['probabilities']
 assert np.isfinite(pr).all() and np.all((pr>=0)&(pr<=1))
 assert np.array_equal(z['contact_candidate'],(pr.min(0)>=.8)&rel)
 assert np.array_equal(z['observed_separation_candidate'],(pr.max(0)<=.2)&rel&(d['distance']-2*d['sigma']>.006))
 rr={'episode':ep,'dataset_hash_equal':True,'context_equal':True,'counts':{},'weak_bce':{}}
 for name,p in [('CF003',a),('CF005',z)]:
  rr['counts'][name]=[int(p[k].sum()) for k in ['contact_candidate','observed_separation_candidate','unknown']]
  w=d['weight'];t=d['target'];m=np.clip(p['probabilities'].mean(0),1e-7,1-1e-7)
  rr['weak_bce'][name]=float((w*(-t*np.log(m)-(1-t)*np.log(1-m))).sum()/max(w.sum(),1e-9))
  if ep=='white_yellow_mug_demo_5':rr[name+'_44_52']={'candidate_frames':(np.flatnonzero(p['contact_candidate'][44:53])+44).tolist(),'mean_scores':p['probabilities'][:,44:53].mean(0).tolist(),'states':[0 if p['contact_candidate'][i] else 1 if p['observed_separation_candidate'][i] else 2 for i in range(44,53)]}
  if ep=='porcelain_mug_demo_9':rr[name+'_56_95']=[int(p[k][56:96].sum()) for k in ['contact_candidate','observed_separation_candidate','unknown']]
 rows.append(rr)
out={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),'input_lock_unchanged':True,'model_same_as_cf003_no_world':True,'runs':runs,'comparison':rows,'source_code_sha':sha(B/'src/train.py'),'changed_only_training_anchor_eligibility_and_resulting_plan':True,'old_internal_audit_now_training':True}
(B/'audits/final_numeric.json').write_text(json.dumps(out,indent=2))
print(json.dumps(rows))
