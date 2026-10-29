import json,sys,subprocess,traceback
from common14 import *
import numpy as np
from evaluate14 import runs
CONTROL=ROOT/'experiments/contact_frames/cf014'
def stats(a,bounds):
 q=a[:,bounds[0]:bounds[1]+1];m=q.mean(0)
 return dict(frames=len(m),mean_score=float(q.mean()),ensemble_high_frames=int((m>=.8).sum()),ensemble_high_fraction=float((m>=.8).mean()),seed_high_fraction=(q>=.8).mean(1).tolist(),seed_mean_score=q.mean(1).tolist())
def main():
 assert (BASE/'weights_frozen.json').exists()
 p=subprocess.run([sys.executable,'-u',str(BASE/'src/evaluate14.py')],cwd=BASE,stdout=open(BASE/'evaluate.log','w'),stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError('evaluate_exit_'+str(p.returncode))
 human=json.loads((BASE/'evaluation/manual_review_source.json').read_text())
 payload=json.loads((ROOT/'docs/图示/cf014_contact/data.json').read_text())
 rows=[]
 for case in payload['cases']:
  ep=case['id'];a=np.load(BASE/'evaluation'/ep/'predictions.npz')['with_motion'];control=np.load(CONTROL/'evaluation'/ep/'predictions.npz')['with_motion']
  bounds=human['positive'][ep];negative=human['negative_explicit_only'].get(ep)
  row=dict(episode=ep,split='training_review' if ep in CFG['train_episodes'] else 'held_review',human_positive=bounds,human_negative=negative,models={})
  case['models']={}
  for arm,q in [('control',control),('no_point_mass',a)]:
   m=q.mean(0)
   row['models'][arm]=dict(positive=stats(q,bounds),candidate_ranges=runs(m>=.8))
   if negative is not None:row['models'][arm]['explicit_negative']=stats(q,negative)
   case['models'][arm]=dict(mean=m.tolist(),low=q.min(0).tolist(),high=q.max(0).tolist(),ranges=runs(m>=.8))
  case['metrics']=row['models'];rows.append(row)
 aggregate={}
 held=[x for x in rows if x['split']=='held_review']
 for arm in ['control','no_point_mass']:
  neg=[x['models'][arm]['explicit_negative']['ensemble_high_fraction'] for x in held if 'explicit_negative' in x['models'][arm]]
  aggregate[arm]=dict(positive_event_mean_coverage=float(np.mean([x['models'][arm]['positive']['ensemble_high_fraction'] for x in held])),explicit_negative_event_mean_false_high=float(np.mean(neg)),positive_events=len(held),explicit_negative_events=len(neg))
 out=dict(time=now(),threshold=.8,rows=rows,aggregate=aggregate,manual_use='post-freeze review only, no training or selection',outside_intervals='unknown')
 save(BASE/'evaluation/review_metrics.json',out)
 payload.update(time=now(),aggregate=aggregate)
 dest=ROOT/'docs/图示/cf016_contact';dest.mkdir(parents=True,exist_ok=True)
 save(dest/'data.json',payload)
 save(BASE/'comparison_frozen.json',dict(time=now(),files=lock([BASE/'evaluation/review_metrics.json',dest/'data.json']),weights_frozen=json.loads((BASE/'weights_frozen.json').read_text())['time']))
 stage('comparison_ready')
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(BASE/'failure_compare.json',dict(time=now(),type=type(e).__name__,message=str(e),stack=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(e.__traceback__)]));stage('compare_failed');raise SystemExit(1)
