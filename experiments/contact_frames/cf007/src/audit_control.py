import pathlib,json,hashlib,numpy as np,torch
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA');BASE=ROOT/'experiments/contact_frames';RUN=BASE/'cf007';OLD=BASE/'cf005'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
cfg=json.loads((RUN/'config.json').read_text())
audit={'model_source_identical':sha(RUN/'src/model_all_train_frames.py')==sha(OLD/'src/model_all_train_frames.py'),'seeds':[],'inputs':{},'initialization_equal':True,'plans_equal':True}
assert audit['model_source_identical']
for fn in ['input_locked.json','source_locked.json','training_inputs_locked.json']:
 lock=json.loads((RUN/fn).read_text())
 assert all(sha(pathlib.Path(p))==h for p,h in lock.items()),fn
 audit['inputs'][fn]=len(lock)
for seed in cfg['seeds']:
 newdir=RUN/'runs/all_train_frames'/('seed_'+str(seed));oldir=OLD/'runs/all_train_frames'/('seed_'+str(seed))
 a=torch.load(newdir/'initial.pt',map_location='cpu',weights_only=False);b=torch.load(oldir/'initial.pt',map_location='cpu',weights_only=False)
 assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
 assert sha(RUN/'plans'/('seed_'+str(seed)+'.json'))==sha(OLD/'plans'/('seed_'+str(seed)+'.json'))
 h=json.loads((newdir/'history.json').read_text());assert len(h)==600
 ck=torch.load(newdir/'last.pt',map_location='cpu',weights_only=False)
 assert ck['steps']==600 and all(int(x['step'])==600 for x in ck['optimizer']['state'].values())
 audit['seeds'].append({'seed':seed,'steps':600,'initial_tensor_hash':ck['initial_tensor_hash'],'last_sha256':sha(newdir/'last.pt'),'parameters':sum(v.numel() for v in ck['model'].values())})
rows=[]
for ep in cfg['train_episodes']+cfg['fixed_review_episodes']:
 path=RUN/'evaluation_all_train_frames'/ep/'artifacts'
 oldpath=(OLD/'evaluation_all_train_frames'/ep/'artifacts' if ep not in ['tomato_sauce_demo_3','alphabet_soup_demo_1'] else BASE/'cf005_frozen_review_003'/ep/'artifacts')
 n=np.load(path/'frame_predictions.npz');o=np.load(oldpath/'frame_predictions.npz');d=np.load(path/'dataset.npz')
 def metric(z,a=0,b=None):
  b=len(z['mean'])-1 if b is None else b
  sl=slice(a,b+1);p=z['probabilities'][:,sl]
  return {'candidate':int(z['contact_candidate'][sl].sum()),'separation':int(z['observed_separation_candidate'][sl].sum()),'unknown':int(z['unknown'][sl].sum()),'mean_score':float(p.mean()),'seed_high_counts':(p>=.8).sum(1).tolist(),'unanimous_high':int((p.min(0)>=.8).sum())}
 row={'episode':ep,'training':ep in cfg['train_episodes'],'frames':len(n['mean']),'cf005':metric(o),'cf007':metric(n)}
 if ep in ['tomato_sauce_demo_3','alphabet_soup_demo_1']:
  ranges= [[52,69],[94,153]] if ep=='tomato_sauce_demo_3' else [[46,64],[80,158]]
  row['command_attempt_navigation_only']=[{'range':[a,b],'cf005':metric(o,a,b),'cf007':metric(n,a,b)} for a,b in ranges]
 ranges={'tomato_sauce_demo_7':[52,132],'porcelain_mug_demo_9':[56,95],'white_yellow_mug_demo_5':[44,52]}
 if ep in ranges:
  a,b=ranges[ep];row['human_review_only']={'range':[a,b],'cf005':metric(o,a,b),'cf007':metric(n,a,b)}
 weak=d['weight']>0;tar=d['target'];p=np.clip(n['mean'],1e-6,1-1e-6)
 row['weak_bce']=float(np.sum(d['weight']*(-tar*np.log(p)-(1-tar)*np.log(1-p)))/max(float(d['weight'].sum()),1e-8))
 rows.append(row)
save(RUN/'audits/matched_control.json',audit);save(RUN/'comparison.json',{'rows':rows,'human_feedback_train':False,'candidate_rule':'unchanged CF005; CF006 not applied','contact_ground_truth_metric':False})
print(json.dumps({'audit_ok':True,'rows':[x for x in rows if not x['training'] or x['episode']=='tomato_sauce_demo_7']},ensure_ascii=False))
