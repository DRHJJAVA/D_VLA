from common14 import *
import numpy as np,torch,importlib.util
torch.set_num_threads(2)
from model14 import ContactMotionNet,m as oldmodule
def checklock(p,key=None):
    x=json.loads(p.read_text());x=x[key] if key else x
    mismatches=[str(f) for f,h in x.items() if not Path(f).exists() or sha(f)!=h]
    assert not mismatches,(p.name,mismatches)
    return len(x)
def main():
    counts={}
    for name,key in [('training_input_locked.json',None),('evaluation_input_locked.json',None),
                     ('weights_frozen.json','checkpoints'),('predictions_frozen.json','files')]:
        counts[name]=checklock(BASE/name,key)
    runs=[]
    for seed in CFG['seeds']:
        plan=json.loads((OLD/'plans'/f'seed_{seed}.json').read_text())
        initial=[];ends=[]
        for arm in CFG['arms']:
            path=BASE/'runs'/arm/f'seed_{seed}'
            h=json.loads((path/'history.json').read_text())
            assert len(h)==600 and [x['step'] for x in h]==list(range(1,601))
            assert [x['episode'] for x in h]==[x['episode'] for x in plan]
            assert np.isfinite([[x[k] for k in ['loss','main','near','dynamics']] for x in h]).all()
            ck=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)['model'];initial.append(ck)
            old=torch.load(OLD/'runs/full_history'/f'seed_{seed}'/'initial.pt',map_location='cpu',weights_only=False)['model']
            assert all(torch.equal(ck['base.'+k],v) for k,v in old.items())
            last=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
            assert last['steps']==600 and all(torch.isfinite(v).all() for v in last['model'].values())
            optimizer_steps=[int(v['step']) for v in last['optimizer']['state'].values()]
            assert min(optimizer_steps)==600 and max(optimizer_steps)==600
            ends.append(last['model'])
            runs.append(dict(arm=arm,seed=seed,updates=600,parameters=sum(v.numel() for v in ck.values()),
              final_motion_last_norm=float(last['model']['motion.2.weight'].norm()),loss=h[-1]['loss']))
        assert initial[0].keys()==initial[1].keys()
        assert all(torch.equal(initial[0][k],initial[1][k]) for k in initial[0])
        assert initial[0]['motion.2.weight'].abs().max()==0
    # Independent main-target reconstruction from frozen arrays.
    for ep in CFG['train_episodes']:
        with np.load(dataset(ep)) as d,np.load(OLD/'evaluation'/ep/'artifacts/frame_predictions.npz') as g,np.load(BASE/'features'/f'{ep}_supervision.npz') as t:
            sep=g['geometry_separation_evidence'].astype(bool)
            w=np.where(sep,d['near_weight'],d['weight']*d['interaction_support']*d['near_target']).astype('float32')
            assert np.array_equal(w,t['weight'])
            assert np.array_equal(np.where(sep,0,1).astype('float32'),t['target'])
            assert np.array_equal(w==0,t['unknown'])
    # Original-model output equivalence at shared initialization, including all heads.
    ep=CFG['train_episodes'][0]
    with np.load(dataset(ep)) as d:
        norm=json.loads((OLD.parent/'cf003/normalization.json').read_text())
        x=[torch.tensor(a,dtype=torch.float32)[None,:8] for a in [d['images'],d['points'],(d['states']-norm['mean'])/norm['std']]]
    old=oldmodule.ContactFrameNet(50).eval();net=ContactMotionNet(False).eval()
    ck=torch.load(BASE/'runs/without_motion/seed_2026/initial.pt',map_location='cpu',weights_only=False)['model']
    net.load_state_dict(ck);old.load_state_dict({k[5:]:v for k,v in ck.items() if k.startswith('base.')})
    with torch.no_grad():diff=max(float((a-b).abs().max()) for a,b in zip(old(*x),net(*x,torch.randn(1,8,24))))
    assert diff==0
    # Review numbers independently reconstructed with inclusive frame intervals.
    h=json.loads((BASE/'evaluation/manual_review_source.json').read_text())
    metrics=json.loads((BASE/'evaluation/review_metrics.json').read_text())
    for row in metrics['rows']:
        ep=row['episode'];a,b=h['positive'][ep]
        with np.load(BASE/'evaluation'/ep/'predictions.npz') as d:
            for arm,v in row['models'].items():
                mean=d[arm].mean(0)
                assert v['positive']['ensemble_high_frames']==int(np.sum(mean[a:b+1]>=.8))
                if ep in h['negative_explicit_only']:
                    l,r=h['negative_explicit_only'][ep]
                    assert v['explicit_negative']['ensemble_high_frames']==int(np.sum(mean[l:r+1]>=.8))
    assert set(CFG['train_episodes']).isdisjoint(CFG['review_episodes'])
    freeze=json.loads((BASE/'weights_frozen.json').read_text())['time']
    review=json.loads((BASE/'audits/review_access.json').read_text())['time']
    assert freeze<review
    out=dict(time=now(),passed=True,locks=counts,runs=runs,original_initial_output_difference=diff,
      train_review_disjoint=True,human_file_access_after_weights_frozen=True,
      supervision_reconstruction_equal=True,manual_metrics_reconstruction_equal=True,
      test_access=False,claims='controlled measured-motion input ablation; not CF013 learned-weight transfer')
    save(BASE/'audits/completion_audit.json',out)
    print(json.dumps(dict(passed=True,runs=len(runs),updates=sum(x['updates'] for x in runs),locks=counts)))
if __name__=='__main__':main()
