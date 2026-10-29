from pathlib import Path
import numpy as np,json
from datetime import datetime
from zoneinfo import ZoneInfo
BASE=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames');R=BASE/'cf010'
assert (R/'pre_review_freeze.json').exists()
ranges={'tomato_sauce_demo_3':[98,160],'alphabet_soup_demo_1':[85,166],'white_yellow_mug_demo_5':[53,91],'alphabet_soup_demo_0':[47,134],'porcelain_mug_demo_9':[49,95],'tomato_sauce_demo_7':[53,132]}
negative={'tomato_sauce_demo_3':[49,74],'white_yellow_mug_demo_5':[44,52]}
review={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),'positive':ranges,'negative_explicit_only':negative,'outside_positive_unknown':True,'training_use':False,'last_duplicate_demo3_resolved_as':'tomato_sauce_demo_7'}
(R/'review_ranges.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
def get(name,ep):
 if name=='CF005':
  root=BASE/'cf005_frozen_review_003' if ep in ['tomato_sauce_demo_3','alphabet_soup_demo_1'] else BASE/'cf005/evaluation_all_train_frames'
 elif name=='CF008':root=BASE/'cf008/evaluation'
 elif name=='CF009B':root=BASE/'cf009/persistent_evidence/evaluation'
 else:root=R/'evaluation'
 return np.load(root/ep/'artifacts/frame_predictions.npz'),np.load(root/ep/'artifacts/dataset.npz')
def metric(p,d,a,b):
 s=slice(a,b+1)
 reliable=d['geometry_reliable']&d['state_valid']
 separated=reliable&(d['distance']-2*d['sigma']>.006)
 raw=p['probabilities'].min(0)>=.8
 predicted=raw&~separated
 neg=p['observed_separation_candidate']
 return {'frames':b-a+1,'strict_candidate':int(p['contact_candidate'][s].sum()),'predicted_contact':int(predicted[s].sum()),'observed_separation':int(neg[s].sum()),'predicted_unknown':int((~(predicted|neg))[s].sum()),'mean':float(p['probabilities'][:,s].mean()),'all_seeds_high':int(raw[s].sum()),'observation_reliable':int(reliable[s].sum()),'predicted_contact_without_direct_support':int((predicted&~reliable)[s].sum()),'predicted_frames':(np.flatnonzero(predicted[s])+a).tolist()}
rows=[]
for ep,bounds in ranges.items():
 ds={name:get(name,ep) for name in ['CF005','CF008','CF009B','CF010']}
 regions={'full':[0,len(ds['CF010'][0]['mean'])-1],'user_contact':bounds}
 if ep in negative:regions['user_noncontact_or_failed']=negative[ep]
 for region,(a,b) in regions.items():
  rows.append({'episode':ep,'region':region,'range':[a,b],'is_training_video':ep=='tomato_sauce_demo_7','results':{k:metric(p,d,a,b) for k,(p,d) in ds.items()}})
(R/'comparison.json').write_text(json.dumps({'rows':rows,'human_ranges_training':False,'outside_positive_not_negative':True},ensure_ascii=False,indent=2))
print(json.dumps({'ready':True,'rows':len(rows)}))
