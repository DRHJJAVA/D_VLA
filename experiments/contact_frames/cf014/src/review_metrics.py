from common14 import *
import numpy as np
from evaluate14 import runs

def stats(p,mask):
    a=p[:,mask];m=a.mean(0)
    return dict(frames=int(mask.sum()),mean_score=float(a.mean()),
        ensemble_high_frames=int((m>=.8).sum()),ensemble_high_fraction=float((m>=.8).mean()),
        seed_high_fraction=[float((x>=.8).mean()) for x in a],
        seed_mean_score=[float(x.mean()) for x in a])

def main():
    assert (BASE/'predictions_frozen.json').exists()
    h=json.loads((BASE/'evaluation/manual_review_source.json').read_text())
    rows=[]
    for ep,bounds in h['positive'].items():
        with np.load(BASE/'evaluation'/ep/'predictions.npz') as d:
            n=d['with_motion'].shape[1];mask=np.zeros(n,bool);mask[bounds[0]:bounds[1]+1]=True
            assert bounds[1]<n
            neg=np.zeros(n,bool)
            if ep in h['negative_explicit_only']:
                a,b=h['negative_explicit_only'][ep];neg[a:b+1]=True
            assert not (neg&mask).any()
            row=dict(episode=ep,split='training_review' if ep in CFG['train_episodes'] else 'held_review',
                human_positive=bounds,human_negative=h['negative_explicit_only'].get(ep),models={})
            for arm in ['without_motion','with_motion','legacy']:
                p=d[arm]
                out=dict(positive=stats(p,mask),candidate_ranges=runs(p.mean(0)>=.8))
                if neg.any():
                    out['explicit_negative']=stats(p,neg)
                    out['positive_minus_negative_mean']=float(p[:,mask].mean()-p[:,neg].mean())
                row['models'][arm]=out
            rows.append(row)
    held=[r for r in rows if r['split']=='held_review']
    aggregate={}
    for arm in ['without_motion','with_motion','legacy']:
        aggregate[arm]=dict(
            positive_event_mean_coverage=float(np.mean([r['models'][arm]['positive']['ensemble_high_fraction'] for r in held])),
            explicit_negative_event_mean_false_high=float(np.mean([r['models'][arm]['explicit_negative']['ensemble_high_fraction'] for r in held if 'explicit_negative' in r['models'][arm]])),
            positive_events=len(held),explicit_negative_events=2)
    out=dict(time=now(),threshold=.8,manual_use='review only after frozen weights and predictions',
        outside_intervals='unknown; not counted negative',rows=rows,aggregate=aggregate)
    save(BASE/'evaluation/review_metrics.json',out)
    stage('contact_comparison_ready')
if __name__=='__main__':main()
