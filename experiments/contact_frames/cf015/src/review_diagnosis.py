from pathlib import Path
import json,numpy as np,hashlib,ast
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/mnt/nvme1/drh/D_VLA');BASE=ROOT/'experiments/contact_frames/cf015';PREV=BASE.parent/'cf014'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def main():
 cfg=json.loads((BASE/'config.json').read_text())
 assert json.loads((BASE/'stage_state.json').read_text())['stage']=='frozen_diagnostics_complete'
 human_path=BASE.parent/'cf010/review_ranges.json'
 h=json.loads(human_path.read_text())
 save(BASE/'audits/manual_access.json',dict(time=now(),source=str(human_path),sha256=sha(human_path),usage='fixed post-prediction review only',no_fitting=True))
 rows=[]
 for ep,pos in h['positive'].items():
  with np.load(BASE/'probes'/f'{ep}.npz') as n:
   segments={'human_contact':pos}
   if ep in h['negative_explicit_only']:segments['explicit_no_contact']=h['negative_explicit_only'][ep]
   row=dict(episode=ep,split='train_review' if ep in cfg['train_episodes'] else 'held_review',segments={})
   for label,(a,b) in segments.items():
    sl=slice(a,b+1);probs=n['probabilities'][:,:,sl];logits=n['logits'][:,:,sl];near=n['proximity'][:,:,sl]
    raw=n['motion_raw'][sl];valid=n['motion_valid'][sl]
    vstats={}
    for vi,v in enumerate(cfg['variants']):
     p=probs[:,vi];avg=p.mean(0)
     vstats[v]=dict(mean_score=float(p.mean()),seed_mean=[float(x.mean()) for x in p],
       high_frames=int((avg>=.8).sum()),mean_abs_delta=float(np.abs(p-probs[:,0]).mean()),
       mean_signed_delta=float((p-probs[:,0]).mean()),mean_abs_logit_delta=float(np.abs(logits[:,vi]-logits[:,0]).mean()))
    row['segments'][label]=dict(frames=b-a+1,bounds=[a,b],variants=vstats,
      proximity_mean=float(near[:,0].mean()),proximity_high_frames=int((near[:,0].mean(0)>=.8).sum()),
      raw_motion_medians=np.median(raw,axis=0).tolist(),valid_motion_frames=int(valid.sum()),
      geometry_distance_median=float(np.median(n['distance'][sl])),
      measured_speed_mm_median=float(np.median(raw[:,15])*10),
      hand_expected_speed_mm_median=float(np.median(raw[:,18])*10),
      hand_residual_mm_median=float(np.median(raw[:,17])*10),
      measured_valid_weight_fraction_median=float(np.median(raw[:,20])))
   rows.append(row)
 train=json.loads((BASE/'train/summary.json').read_text())
 positive=sum(x['groups']['positive']['frames'] for x in train['supervision'])
 negative=sum(x['groups']['separated_negative']['frames'] for x in train['supervision'])
 unknown=sum(x['groups']['unknown']['frames'] for x in train['supervision'])
 targetlow=sum(z['frames'] for x in train['supervision'] for z in x['positive_soft_target_bands'] if z['high']<=.1)
 targethalf=sum(z['frames'] for x in train['supervision'] for z in x['positive_soft_target_bands'] if z['high']<=.5)
 ratios=[g['shared_gradient_norms'][1]/max(g['shared_gradient_norms'][0],1e-15) for g in train['gradients']]
 near_negmax=max(x['groups']['separated_negative']['near']['max'] for x in train['supervision'])
 correlations=[v for s in ['train','review'] for x in json.loads((BASE/s/'summary.json').read_text())['rows'] for v in x['main_near_probability_correlation'] if v is not None]
 out=dict(time=now(),rows=rows,training_findings=dict(positive=positive,negative=negative,unknown=unknown,
   positive_old_soft_below_point1=targetlow,positive_old_soft_below_point5=targethalf,
   maximum_negative_near_target=near_negmax,
   final_aux_near_to_main_shared_gradient_ratio_min=float(min(ratios)),
   final_aux_near_to_main_shared_gradient_ratio_median=float(np.median(ratios)),
   final_aux_near_to_main_shared_gradient_ratio_max=float(max(ratios))),
   main_proximity_correlation_range=[min(correlations),max(correlations)],
   restrictions=['no human fitting','perturbations are distribution shifts, not physical contact truth',
                 'last-checkpoint gradients cannot establish training-history causality'])
 save(BASE/'review/diagnostic_review.json',out)
 print(json.dumps(dict(training=out['training_findings'],main_near_corr=out['main_proximity_correlation_range']),ensure_ascii=False))
 for r in rows:
  if 'explicit_no_contact' in r['segments']:
   s=r['segments']['explicit_no_contact']
   print(json.dumps(dict(episode=r['episode'],negative=s),ensure_ascii=False))
if __name__=='__main__':main()
