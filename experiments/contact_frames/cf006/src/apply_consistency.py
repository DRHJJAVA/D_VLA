import pathlib,json,hashlib,datetime,numpy as np
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA'); BASE=ROOT/'experiments/contact_frames';RUN=BASE/'cf006'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
cfg=json.loads((RUN/'config.json').read_text());inputs={};rows=[]
for ep in cfg['episodes']:
 if ep in ['tomato_sauce_demo_3','alphabet_soup_demo_1']:p=BASE/'cf005_frozen_review_003'/ep/'artifacts'
 else:p=BASE/'cf005/evaluation_all_train_frames'/ep/'artifacts'
 for fn in ['dataset.npz','frame_predictions.npz']:inputs[str(p/fn)]=sha(p/fn)
save(RUN/'input_locked.json',inputs)
for ep in cfg['episodes']:
 if ep in ['tomato_sauce_demo_3','alphabet_soup_demo_1']:p=BASE/'cf005_frozen_review_003'/ep/'artifacts'
 else:p=BASE/'cf005/evaluation_all_train_frames'/ep/'artifacts'
 d=np.load(p/'dataset.npz');old=np.load(p/'frame_predictions.npz')
 reliable=d['geometry_reliable']&d['state_valid']
 separated=reliable&np.isfinite(d['distance'])&np.isfinite(d['sigma'])&(d['distance']-2*d['sigma']>.006)
 conflict=old['contact_candidate']&separated
 candidate=old['contact_candidate']&~separated
 neg=old['observed_separation_candidate'].copy()
 unknown=~(candidate|neg)
 out=RUN/ep/'artifacts';out.mkdir(parents=True)
 (out/'dataset.npz').symlink_to((p/'dataset.npz').resolve())
 values={k:old[k] for k in old.files}
 values.update(original_contact_candidate=old['contact_candidate'],contact_candidate=candidate,unknown=unknown,evidence_conflict=conflict,geometry_separation_evidence=separated)
 np.savez_compressed(out/'frame_predictions.npz',**values)
 assert np.array_equal(values['probabilities'],old['probabilities'])
 assert not np.any(candidate&neg) and not np.any(neg&conflict)
 assert np.array_equal(unknown,old['unknown']|conflict)
 def spans(mask):
  ids=np.flatnonzero(mask);return [[int(g[0]),int(g[-1])] for g in np.split(ids,np.where(np.diff(ids)>1)[0]+1) if len(g)]
 row={'episode':ep,'frames':len(candidate),'before_candidates':int(old['contact_candidate'].sum()),'after_candidates':int(candidate.sum()),'separation':int(neg.sum()),'unknown':int(unknown.sum()),'conflict_frames':np.flatnonzero(conflict).tolist(),'candidate_ranges':spans(candidate),'raw_scores_unchanged':True}
 rows.append(row)
save(RUN/'summary.json',{'rows':rows,'trained':False,'success_truth_available':False,'source':'frozen CF005 predictions; this is evidence consistency correction, not new network inference'})
assert all(sha(pathlib.Path(p))==h for p,h in inputs.items())
save(RUN/'audits/completion.json',{'input_hashes_unchanged':True,'raw_scores_unchanged':True,'unknown_only_increases_by_conflict':True,'no_new_labels':True,'no_training':True})
print(json.dumps(rows,ensure_ascii=False))
