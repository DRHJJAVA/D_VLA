from pathlib import Path
import sys,json,hashlib,time,random
import numpy as np,torch
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose/iter_016';O=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B/'src'))
import train as T
from model import RegionPoseNet
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def check(lock): 
 for p,h in lock.items():assert sha(P/p)==h,(p,'hash mismatch')
def dist(a):
 return dict(n=len(a),quantiles=dict(zip(['min','p05','p25','median','p75','p95','max'],map(float,np.quantile(a,[0,.05,.25,.5,.75,.95,1]))))) if len(a) else dict(n=0,quantiles=None)
def measure(s,d):
 m=T.metrics(s,d);y=d['target'].cpu().numpy().astype(bool);k=d['known'].cpu().numpy();assert not (y&~k).any()
 order=np.argsort(-s,kind='stable');curves=[]
 for b in [1,2,5,10,20]:
  n=max(1,int(np.ceil(len(s)*b/100)));ix=order[:n]
  curves.append(dict(budget_percent=b,selected=n,observed_positive_coverage=float(y[ix].sum()/y.sum()),known_positive_fraction=float(y[ix].mean()),known_negative_fraction=float((k[ix]&~y[ix]).mean()),unknown_fraction=float((~k[ix]).mean()),random_expected_coverage=n/len(s)))
 return dict(metrics=m,curves=curves)
def aggregate(rows):
 def avg(rr):
  keys=['loss_head','loss_training_objective','ap','coverage10','coverage20','coverage30','valid_fraction']
  return dict(n_queries=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in keys},curves=[dict(budget_percent=b,**{k:float(np.mean([r['curves'][i][k] for r in rr])) for k in ['observed_positive_coverage','known_positive_fraction','known_negative_fraction','unknown_fraction','random_expected_coverage']}) for i,b in enumerate([1,2,5,10,20])])
 episodes=[dict(episode=ep,**avg([r for r in rows if r['episode']==ep])) for ep in sorted({r['episode'] for r in rows})]
 assert len({r['event'] for r in rows})==len(episodes)
 return dict(overall=avg(rows),episodes=episodes,offsets=[dict(offset=o,**avg([r for r in rows if r['offset']==o])) for o in [15,10,5,1]],rows=rows)
def main():
 torch.set_num_threads(4);torch.manual_seed(2026);np.random.seed(2026);random.seed(2026);torch.cuda.set_per_process_memory_fraction(.15)
 old=json.loads((B/'completed_locked.json').read_text());check(old)
 inputs=json.loads((O/'input_locked.json').read_text());check(inputs)
 source=json.loads((O/'source_locked.json').read_text());check(source)
 manifest=json.loads((B/'manifest.json').read_text());runs=json.loads((B/'config.json').read_text())['runs']
 hist={}
 for rc in runs:
  run=rc['group']+'_'+rc['variant'];h=json.loads((B/'runs'/run/'history.json').read_text())
  hist[run]=dict(blocks={str(a)+'-'+str(b):float(np.mean([x['train_loss'] for x in h[a-1:b]])) for a,b in [(1,4),(17,20),(25,28),(29,32)]},last8_slope=float(np.polyfit(np.arange(8),[x['train_loss'] for x in h[-8:]],1)[0]),loss_includes_auxiliary=rc['variant']=='pose_conditioned',history=h)
 put(O/'history_analysis.json',hist)
 differences=[]
 for split in ['train','val']:
  if split=='val':
   assert (O/'pre_validation_freeze.json').exists();check(source);check(inputs)
  put(O/'stage_state.json',dict(stage=split,status='running',time=time.time()))
  rows=T.loadrows(dict(rows=[r for r in manifest['rows'] if r['split']==split]),torch.device('cuda'))
  baseline=[]
  for r,d in rows:
   center=-torch.linalg.vector_norm(d['xyz'],dim=1).cpu().numpy()
   local=(d['xyz']-d['hand_t'])@d['hand_R'];gap=d['pose'][-1]*.1
   lo=local.new_tensor([-.008,0,.0854]);hi=local.new_tensor([.008,0,.1014]);lo[1]=-gap/2;hi[1]=gap/2
   geometry=-torch.linalg.vector_norm(torch.maximum(lo-local,torch.zeros_like(local))+torch.maximum(local-hi,torch.zeros_like(local)),dim=1).cpu().numpy()
   rng=np.random.default_rng(2026);randomm=[measure(rng.random(len(center)),d) for _ in range(64)]
   baseline.append(dict(key=r['key'],episode=r['episode'],event=r['event'],offset=r['offset'],center=measure(center,d),current_gripper=measure(geometry,d),random=dict(repeats=64,ap_mean=float(np.mean([v['metrics']['ap'] for v in randomm])),ap_std=float(np.std([v['metrics']['ap'] for v in randomm])),curves=[dict(budget_percent=b,coverage_mean=float(np.mean([v['curves'][i]['observed_positive_coverage'] for v in randomm])),coverage_std=float(np.std([v['curves'][i]['observed_positive_coverage'] for v in randomm])),unknown_mean=float(np.mean([v['curves'][i]['unknown_fraction'] for v in randomm])),known_negative_mean=float(np.mean([v['curves'][i]['known_negative_fraction'] for v in randomm]))) for i,b in enumerate([1,2,5,10,20])])))
  put(O/split/'baselines.json',baseline)
  summary=[]
  for rc in runs:
   run=rc['group']+'_'+rc['variant'];variant=rc['variant']
   for kind in ['best','last']:
    ck=torch.load(B/'runs'/run/(kind+'.pt'),weights_only=False,map_location='cuda');net=RegionPoseNet().cuda();net.load_state_dict(ck['state']);net.eval();out=[]
    with torch.inference_mode():
     for r,d in rows:
      a,b=T.forward(net,d);s=a if variant=='object_only' else b;score=s.sigmoid().cpu().numpy();logit=s.cpu().numpy()
      m=measure(logit,d);y=d['target'].cpu().numpy().astype(bool);known=d['known'].cpu().numpy()
      p=O/split/'predictions'/run/kind/(r['key']+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
      np.savez_compressed(p,logit=logit,score=score,target=y,known=known)
      out.append(dict(key=r['key'],episode=r['episode'],event=r['event'],offset=r['offset'],loss_head=float(T.loss(s,d)),loss_training_objective=float(T.loss(s,d)+(.5*T.loss(a,d) if variant=='pose_conditioned' else 0)),**m['metrics'],curves=m['curves'],distributions={name:dict(logit=dist(logit[mask]),score=dist(score[mask])) for name,mask in [('positive',y),('negative',known&~y),('unknown',~known)]}))
      if split=='val' and kind=='best':
       with np.load(B/'runs'/run/'val_predictions'/(r['key']+'.npz')) as z:delta=float(np.max(np.abs(z['score']-score)))
       differences.append(dict(run=run,key=r['key'],max_abs_score_difference=delta));assert delta<1e-6
    result=dict(run=run,checkpoint=kind,epoch=ck['epoch'],checkpoint_sha256=sha(B/'runs'/run/(kind+'.pt')),**aggregate(out));put(O/split/(run+'_'+kind+'.json'),result);summary.append(result)
    print(split,run,kind,ck['epoch'],result['overall']['ap'],result['overall']['loss_head'],flush=True)
    del net,ck;torch.cuda.empty_cache()
  put(O/split/'summary.json',summary)
  locked={str(p.relative_to(P)):sha(p) for p in (O/split).rglob('*') if p.is_file()};put(O/split/'locked.json',locked)
  if split=='train':
   put(O/'pre_validation_freeze.json',dict(time=time.time(),source=source,train=locked,scope='6 training events; rules frozen before val samples loaded'))
  del rows;torch.cuda.empty_cache()
 check(old);check(source);check(inputs)
 put(O/'audits/completion_audit.json',dict(old_016_files_verified=len(old),inputs_verified=len(inputs),source_verified=len(source),train_queries=24,val_queries=8,train_events=6,val_events=2,inferences=256,best_val_comparisons=differences,max_best_val_difference=max(x['max_abs_score_difference'] for x in differences),no_training=True,no_pose_swap=True))
 put(O/'stage_state.json',dict(stage='complete',status='complete',time=time.time()))
if __name__=='__main__':main()
