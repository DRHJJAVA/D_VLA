from pathlib import Path
import sys,json,datetime,hashlib,traceback
import numpy as np,torch
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/contact_frames/cf002';P=R/'experiments/contact_frames/cf001_frozen_review_002'
sys.path.insert(0,str(B/'src'))
from model import ContactFrameNet
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def runs(m):
 a=np.flatnonzero(m);return [[int(x[0]),int(x[-1])] for x in np.split(a,np.flatnonzero(np.diff(a)>1)+1) if len(x)]
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.12)
 assert json.loads((B/'stage_state.json').read_text())['stage']=='training_complete'
 cases=json.loads((P/'plan.json').read_text())['cases'];save(B/'plan.json',{'cases':cases,'training':False,'purpose':'post_training_fixed_evaluation'})
 locks={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in B.glob('artifacts/seed_*/last.pt')};save(B/'evaluation_weights_locked.json',locks)
 summary=[];comparison=[]
 for ep in cases:
  dst=B/ep;(dst/'artifacts').mkdir(parents=True)
  (dst/'artifacts/dataset.npz').symlink_to(P/ep/'artifacts/dataset.npz')
  (dst/'config.json').write_bytes((P/ep/'config.json').read_bytes())
  d=np.load(dst/'artifacts/dataset.npz');control=np.load(P/ep/'artifacts/frame_predictions.npz');idx=control['context_indices'];n=len(idx);pred=[]
  for seed in [2026,2027,2028]:
   ck=torch.load(B/f'artifacts/seed_{seed}/last.pt',map_location='cpu',weights_only=False)
   m=ContactFrameNet(50).cuda().eval();m.load_state_dict(ck['model'])
   st=((d['states']-ck['state_mean'])/ck['state_std']).astype('float32');out=[]
   with torch.no_grad():
    for a in range(0,n,16):
     ix=idx[a:a+16];z,_=m(torch.tensor(d['images'][ix],device='cuda'),torch.tensor(d['points'][ix],device='cuda'),torch.tensor(st[ix],device='cuda'));out.extend(torch.sigmoid(z).cpu().tolist())
   pred.append(out);del m
  p=np.array(pred,dtype='float32');lo=p.min(0);hi=p.max(0);reliable=d['geometry_reliable']&d['state_valid'];pos=(lo>=.8)&reliable;neg=(hi<=.2)&reliable&((d['distance']-2*d['sigma'])>.006);unknown=~(pos|neg)
  assert np.isfinite(p).all()
  np.savez_compressed(dst/'artifacts/frame_predictions.npz',probabilities=p,mean=p.mean(0),low=lo,high=hi,contact_candidate=pos,observed_separation_candidate=neg,unknown=unknown,context_indices=idx)
  s={'episode':ep,'frames':n,'contact_candidate':int(pos.sum()),'separation_candidate':int(neg.sum()),'unknown':int(unknown.sum()),'contact_ranges':runs(pos),'timestamp':now(),'weights_unchanged':True,'trained_on_this_video':False}
  save(dst/'inference_summary.json',s);summary.append(s)
  evidence_negative=(d['weight']>0)&(d['target']<=.2)&d['state_valid']
  def metrics(arr):
   mean=arr.mean(0);low=arr.min(0);w=d['weight']*d['state_valid'];q=np.clip(mean,1e-7,1-1e-7);y=d['target']
   return {'ensemble_spread_mean':float((arr.max(0)-arr.min(0)).mean()),'raw_unanimous_high':int((low>=.8).sum()),'weak_negative_frames':int(evidence_negative.sum()),'raw_high_on_weak_negative':int(((low>=.8)&evidence_negative).sum()),'weak_weighted_bce':float(np.sum(w*(-y*np.log(q)-(1-y)*np.log(1-q)))/max(w.sum(),1e-8))}
  row={'episode':ep,'old':metrics(control['probabilities']),'new':metrics(p)}
  if ep=='porcelain_mug_demo_9':
   ix=np.arange(56,96);row['review_56_95']={'new_seed_high_counts':(p[:,ix]>=.8).sum(1).tolist(),'old_seed_high_counts':(control['probabilities'][:,ix]>=.8).sum(1).tolist(),'new_unanimous_high':int((lo[ix]>=.8).sum()),'new_final_candidate':int(pos[ix].sum()),'new_separation':int(neg[ix].sum()),'new_unknown':int(unknown[ix].sum()),'geometry_reliable':int(reliable[ix].sum()),'outside_interval':'not_labelled'}
  comparison.append(row)
 save(B/'summary.json',summary);save(B/'comparison.json',comparison)
 assert all(hashlib.sha256((R/k).read_bytes()).hexdigest()==v for k,v in locks.items())
 save(B/'evaluation_state.json',{'stage':'complete','timestamp':now()})
if __name__=='__main__':
 try:main()
 except Exception as e:save(B/'evaluation_failure.json',{'type':type(e).__name__,'message':str(e),'stack':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
