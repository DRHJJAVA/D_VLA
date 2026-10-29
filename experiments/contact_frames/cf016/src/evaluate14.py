from common14 import *
import numpy as np,torch,traceback
from model14 import ContactMotionNet
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.22)
def runs(mask):
    out=[];start=None
    for i,v in enumerate(list(mask)+[False]):
        if v and start is None:start=i
        elif not v and start is not None:out.append([start,i-1]);start=None
    return out
def main():
    frozen=json.loads((BASE/'weights_frozen.json').read_text());verify(frozen['checkpoints'])
    stage('evaluating_contact')
    mn=json.loads((BASE/'motion_normalization.json').read_text())
    norm=json.loads((OLD.parent/'cf003/normalization.json').read_text())
    episodes=CFG['train_episodes']+CFG['review_episodes']
    data={}
    inputs=[BASE/'weights_frozen.json',BASE/'motion_normalization.json',OLD.parent/'cf003/normalization.json']
    for ep in episodes:
        dp=dataset(ep);fp=BASE/'features'/f'{ep}.npz';inputs.extend([dp,fp])
        with np.load(dp) as d,np.load(fp) as f:
            mo=(f['features']-np.asarray(mn['mean']))/np.asarray(mn['std']);mo[~f['valid']]=0
            arr=[d['images'],d['points'],(d['states']-np.asarray(norm['mean']))/np.asarray(norm['std']),mo]
            data[ep]=[torch.tensor(a,dtype=torch.float32,device='cuda')[None] for a in arr]
    outputs={ep:{} for ep in episodes}
    for arm in CFG['arms']:
        for seed in CFG['seeds']:
            ck=BASE/'runs'/arm/f'seed_{seed}'/'last.pt';inputs.append(ck)
            net=ContactMotionNet(arm=='with_motion').cuda()
            net.load_state_dict(torch.load(ck,map_location='cuda',weights_only=False)['model']);net.eval()
            with torch.no_grad():
                for ep,x in data.items():
                    logits,dy,near=net(*x)
                    outputs[ep].setdefault(arm,[]).append(logits[0].sigmoid().cpu().numpy())
            del net;torch.cuda.empty_cache()
    summary=[]
    for ep,arms in outputs.items():
        dest=BASE/'evaluation'/ep;dest.mkdir(parents=True,exist_ok=False)
        oldpath=OLD/'evaluation'/ep/'artifacts/frame_predictions.npz';inputs.append(oldpath)
        with np.load(oldpath) as old: legacy=old['probabilities'].copy();sep=old['geometry_separation_evidence'].copy()
        arrays=dict(legacy=legacy,separation=sep)
        row=dict(episode=ep,split='training' if ep in CFG['train_episodes'] else 'held_review',arms={})
        for arm,ps in list(arms.items())+[('legacy',legacy)]:
            a=np.asarray(ps);assert a.shape[0]==3 and np.isfinite(a).all()
            arrays[arm]=a
            row['arms'][arm]=dict(candidate_ranges=runs(a.mean(0)>=.8),
                candidate_frames=int((a.mean(0)>=.8).sum()),
                mean_score=float(a.mean()))
        np.savez_compressed(dest/'predictions.npz',**arrays);summary.append(row)
    save(BASE/'evaluation/prediction_summary.json',dict(time=now(),rows=summary,manual_ranges_read=False))
    save(BASE/'evaluation_input_locked.json',lock(inputs))
    save(BASE/'predictions_frozen.json',dict(time=now(),files=lock(list((BASE/'evaluation').glob('*/predictions.npz')))))
    # Human intervals are loaded only after every checkpoint and prediction has been frozen.
    manual_path=OLD/'review_ranges.json'
    human=json.loads(manual_path.read_text())
    save(BASE/'audits/review_access.json',dict(time=now(),source=str(manual_path),sha256=sha(manual_path),
        use='post-training review only; no selection or training',schema=list(human)))
    save(BASE/'evaluation/manual_review_source.json',human)
    stage('predictions_frozen_ready_for_review_metrics')
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(BASE/'failure_eval.json',dict(time=now(),type=type(e).__name__,message=str(e),
             stack=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(e.__traceback__)]))
        stage('evaluation_failed');raise
