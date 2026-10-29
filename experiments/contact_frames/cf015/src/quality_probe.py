from pathlib import Path
import sys,json,numpy as np,torch,traceback
sys.path.insert(0,str(Path(__file__).parent))
import diagnose_v2 as d
BASE=d.BASE;OUT=BASE/'quality_probe';cfg=json.loads((OUT/'config.json').read_text())
def main():
 inputs=[Path(__file__),BASE/'src/diagnose_v2.py',OUT/'config.json',BASE/'config.json']
 inputs.extend(d.PREV/'runs/with_motion'/f'seed_{s}'/'last.pt' for s in d.CFG['seeds'])
 d.save(OUT/'source_weight_locked.json',d.lock(inputs))
 allrows=[]
 for split in ['train','review']:
  if split=='review':assert (OUT/'train_frozen.json').exists()
  rows=[]
  for ep in d.CFG[split+'_episodes']:
   raw,x=d.load(ep);res=[]
   for seed in d.CFG['seeds']:
    net=d.ContactMotionNet(True).cuda()
    net.load_state_dict(torch.load(d.PREV/'runs/with_motion'/f'seed_{seed}'/'last.pt',map_location='cuda',weights_only=False)['model']);net.eval();ps=[]
    with torch.no_grad():
     for name,idx in cfg['variants'].items():
      xx=list(x);xx[-1]=x[-1].clone()
      if idx:xx[-1][:,:,idx]=0
      ps.append(net(*xx)[0][0].sigmoid().cpu().numpy())
    res.append(ps);del net;torch.cuda.empty_cache()
   res=np.asarray(res)
   with np.load(BASE/'probes'/f'{ep}.npz') as saved:assert np.max(np.abs(res[:,0]-saved['probabilities'][:,0]))<1e-6
   p=OUT/f'{ep}.npz';assert not p.exists();np.savez_compressed(p,probabilities=res)
   rows.append(dict(episode=ep,split=split,mean_abs_changes={v:float(np.abs(res[:,i]-res[:,0]).mean()) for i,v in enumerate(cfg['variants'])}))
   del x,raw
  d.save(OUT/f'{split}_summary.json',dict(time=d.now(),rows=rows,human_ranges_read=False))
  if split=='train':d.save(OUT/'train_frozen.json',dict(time=d.now(),files=d.lock([OUT/'train_summary.json']+[OUT/f'{e}.npz' for e in d.CFG['train_episodes']])))
  allrows+=rows
 d.save(OUT/'predictions_frozen.json',dict(time=d.now(),files=d.lock(list(OUT.glob('*.npz')))))
 h=json.loads((d.PREV.parent/'cf010/review_ranges.json').read_text())
 reviews=[]
 for ep,pos in h['positive'].items():
  with np.load(OUT/f'{ep}.npz') as n:
   groups={'contact':pos}
   if ep in h['negative_explicit_only']:groups['no_contact']=h['negative_explicit_only'][ep]
   r=dict(episode=ep,groups={})
   for name,(a,b) in groups.items():
    r['groups'][name]={v:dict(mean_score=float(n['probabilities'][:,i,a:b+1].mean()),
       high_frames=int((n['probabilities'][:,i,a:b+1].mean(0)>=.8).sum()),
       mean_abs_delta=float(np.abs(n['probabilities'][:,i,a:b+1]-n['probabilities'][:,0,a:b+1]).mean()))
       for i,v in enumerate(cfg['variants'])}
   reviews.append(r)
 d.save(OUT/'review_metrics.json',dict(time=d.now(),rows=reviews,manual_use='review only after complete prediction freeze'))
 print(json.dumps(dict(complete=True,episodes=9,variants=list(cfg['variants']))))
if __name__=='__main__':
 try:main()
 except Exception as e:
  d.save(OUT/'failure.json',dict(time=d.now(),type=type(e).__name__,message=str(e),stack=[dict(file=x.filename,line=x.lineno) for x in traceback.extract_tb(e.__traceback__)]));raise
