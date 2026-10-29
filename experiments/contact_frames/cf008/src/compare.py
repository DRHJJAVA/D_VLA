import pathlib,json,numpy as np
BASE=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames');R=BASE/'cf008'
cfg=json.loads((R/'config.json').read_text());rows=[]
def metric(p,a=0,b=None):
 b=len(p['mean'])-1 if b is None else b;sl=slice(a,b+1);v=p['probabilities'][:,sl]
 return {'frames':b-a+1,'candidate':int(p['contact_candidate'][sl].sum()),'separation':int(p['observed_separation_candidate'][sl].sum()),'unknown':int(p['unknown'][sl].sum()),'score_mean':float(v.mean()),'score_min':float(v.min()),'score_max':float(v.max()),'raw_all_high':int((v.min(0)>=.8).sum())}
def ranges(mask):
 a=np.flatnonzero(mask);return [[int(x[0]),int(x[-1])] for x in np.split(a,np.where(np.diff(a)>1)[0]+1) if len(x)]
for ep in cfg['train_episodes']+cfg['fixed_review_episodes']:
 p=np.load(R/'evaluation'/ep/'artifacts/frame_predictions.npz')
 oldbase=BASE/'cf005_frozen_review_003'/ep/'artifacts' if ep in ['tomato_sauce_demo_3','alphabet_soup_demo_1'] else BASE/'cf005/evaluation_all_train_frames'/ep/'artifacts'
 o=np.load(oldbase/'frame_predictions.npz');old6=dict(o)
 d=np.load(oldbase/'dataset.npz');sep=d['geometry_reliable']&d['state_valid']&(d['distance']-2*d['sigma']>.006)
 old6['contact_candidate']=o['contact_candidate']&~sep;old6['unknown']=~(old6['contact_candidate']|o['observed_separation_candidate'])
 row={'episode':ep,'training':ep in cfg['train_episodes'],'cf005':metric(o),'cf005_same_conflict_veto':metric(old6),'cf008':metric(p),'cf008_candidate_ranges':ranges(p['contact_candidate'])}
 rr={'tomato_sauce_demo_3':[(49,74,'user_failed_review'),(94,153,'second_command_attempt_navigation')],'alphabet_soup_demo_1':[(46,64,'first_command_attempt_navigation'),(80,158,'second_command_attempt_navigation')],'porcelain_mug_demo_9':[(56,95,'user_positive_review')],'white_yellow_mug_demo_5':[(44,52,'user_negative_review')],'tomato_sauce_demo_7':[(52,132,'user_positive_training_case')]}
 if ep in rr:
  row['ranges']=[{'range':[a,b],'role':role,'cf005':metric(o,a,b),'cf005_same_conflict_veto':metric(old6,a,b),'cf008':metric(p,a,b),'new_proximity_mean':float(p['proximity_probabilities'][:,a:b+1].mean())} for a,b,role in rr[ep]]
 rows.append(row)
(R/'comparison.json').write_text(json.dumps({'rows':rows,'feedback_used_only_for_reporting':True,'old_score_proximity_vs_new_evidence_different_semantics':True},ensure_ascii=False,indent=2))
for row in rows:
 if not row['training'] or row['episode']=='tomato_sauce_demo_7':print(json.dumps(row,ensure_ascii=False))
