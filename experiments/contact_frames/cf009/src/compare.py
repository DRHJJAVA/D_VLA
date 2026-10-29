from pathlib import Path
import json,numpy as np
B=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames');R=B/'cf009'
cfg=json.loads((R/'config.json').read_text())
roots={'CF005':B/'cf005/evaluation_all_train_frames','CF008':B/'cf008/evaluation','CF009_A':R/'positive_evidence/evaluation','CF009_B':R/'persistent_evidence/evaluation'}
def get(root,ep):return np.load(root/ep/'artifacts/frame_predictions.npz')
def stats(d,a,b):
 sl=slice(a,b+1);p=d['probabilities'][:,sl]
 return {'frames':b-a+1,'candidates':int(d['contact_candidate'][sl].sum()),'separation':int(d['observed_separation_candidate'][sl].sum()),'unknown':int(d['unknown'][sl].sum()),'mean':float(p.mean()),'all_seeds_high':int((p.min(0)>=.8).sum()),'candidate_frames':(np.flatnonzero(d['contact_candidate'][sl])+a).tolist()}
rows=[]
for ep in cfg['train_episodes']+cfg['fixed_review_episodes']:
 ds={}
 for name,root in roots.items():
  if ep in ['tomato_sauce_demo_3','alphabet_soup_demo_1'] and name=='CF005':root=B/'cf005_frozen_review_003'
  ds[name]=get(root,ep)
 N=len(ds['CF008']['mean'])
 ranges={'full':[0,N-1]}
 if ep=='tomato_sauce_demo_3':ranges.update(user_failed=[49,74],attempt2_navigation=[94,153])
 if ep=='alphabet_soup_demo_1':ranges.update(attempt1_navigation=[46,64],attempt2_navigation=[80,158])
 if ep=='white_yellow_mug_demo_5':ranges['user_noncontact']=[44,52]
 if ep=='porcelain_mug_demo_9':ranges['user_contact']=[56,95]
 if ep=='tomato_sauce_demo_7':ranges['user_contact_training_video']=[52,132]
 for region,(a,b) in ranges.items():rows.append({'episode':ep,'region':region,'range':[a,b],'trained':ep in cfg['train_episodes'],'results':{k:stats(v,a,b) for k,v in ds.items()}})
(R/'comparison.json').write_text(json.dumps({'rows':rows,'human_ranges_training':False},ensure_ascii=False,indent=2))
print(json.dumps({'ready':True,'rows':len(rows)}))
