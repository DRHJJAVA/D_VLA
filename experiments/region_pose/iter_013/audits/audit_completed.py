from pathlib import Path
import json,hashlib,sys
import numpy as np,torch
B=Path('/mnt/nvme1/drh/D_VLA/experiments/region_pose/iter_013');R=B.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def verify(d,root):
 for f,h in d.items():assert sha(root/f)==h,(f,'hash mismatch')
inputs=json.loads((B/'training_inputs_locked.json').read_text());verify(inputs,B)
verify(json.loads((B/'source_locked.json').read_text()),R);verify(json.loads((B/'input_locked.json').read_text()),R)
for s in ['train','val']:verify(json.loads((B/'targets'/s/'locked.json').read_text()),B/'targets'/s)
m=json.loads((B/'manifest.json').read_text());rows=[];summaries=[]
for variant in ['object_only','pose_conditioned']:
 folder=B/'runs'/variant;done=json.loads((folder/'done.json').read_text());hist=json.loads((folder/'history.json').read_text())
 assert len(hist)==30 and [x['epoch'] for x in hist]==list(range(1,31))
 ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
 assert ck['input_lock']==inputs and last['input_lock']==inputs and last['epoch']==30
 assert sha(folder/'best.pt')==done['checkpoint_sha256']
 assert ck['epoch']==min(hist,key=lambda h:h['val_loss'])['epoch']
 assert json.loads((folder/'pose_swap_probe.json').read_text())==[]
 for row in [r for r in m['rows'] if r['split']=='val']:
  z=np.load(B/'samples'/(row['key']+'.npz'));p=np.load(folder/'val_predictions'/(row['key']+'.npz'))
  assert np.array_equal(p['target'],z['target']) and np.array_equal(p['known'],z['known'])
  score=p['score'];known=z['known'];y=z['target'].astype(bool)
  rr={'variant':variant,'key':row['key'],'episode':row['episode'],'known_positive':int(y.sum()),'input_points':len(y)}
  for budget in [.1,.2,.3]:
   k=int(np.ceil(budget*len(y)));sel=np.argsort(-score,kind='stable')[:k]
   rr[str(int(100*budget))]={'observed_positive_coverage':float(y[sel].sum()/y.sum()),'unknown_selected_fraction':float((~known[sel]).mean()),'known_negative_selected_fraction':float((known[sel]&~y[sel]).mean()),'random_expected_positive_coverage':k/len(y)}
  rows.append(rr)
 summaries.append({'variant':variant,'best_epoch':ck['epoch'],'validation':done['validation'],'baselines':json.loads((folder/'baselines.json').read_text())})
out={'hashes_verified':True,'both_30_epochs':True,'best_selection_verified':True,'same_event_pose_swap_not_available':True,'train_episodes':m['train_episodes'],'val_episodes':m['val_episodes'],'skipped':m['skipped'],'whole_input_budget':rows,'results':summaries,'physical_contact_ground_truth':False}
(B/'audits/completion_audit.json').write_text(json.dumps(out,indent=2))
for r in rows:print(r)
for s in summaries:
 print(s['variant'],'baseline',[(x['key'],x['center']['ap'],x['current_gripper_box']['ap'],x['current_gripper_box']['coverage20']) for x in s['baselines']])
