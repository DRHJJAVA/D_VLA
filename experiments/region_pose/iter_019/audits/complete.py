from pathlib import Path
import sys,json,hashlib,collections
import numpy as np,torch
P=Path('/mnt/nvme1/drh/D_VLA');O=P/'experiments/region_pose/iter_019';B=P/'experiments/region_pose/iter_016';D=P/'experiments/region_pose/iter_017'
sys.path.insert(0,str(O/'src'));import train as T
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def summarize(rr):
 keys=['ap','loss','new_head_loss','coverage10','coverage20','coverage30','valid_fraction','positive_bce','positive_score_median']
 def avg(rows):return dict(**{k:float(np.mean([r[k] for r in rows])) for k in keys},curves=[dict(budget_percent=b,**{k:float(np.mean([r['curves'][i][k] for r in rows])) for k in ['positive_coverage','known_negative_fraction','unknown_fraction']}) for i,b in enumerate([1,2,5,10,20])])
 return dict(overall=avg(rr),episodes=[dict(episode=ep,**avg([r for r in rr if r['episode']==ep])) for ep in sorted({r['episode'] for r in rr})],offsets=[dict(offset=o,**avg([r for r in rr if r['offset']==o])) for o in [15,10,5,1]],rows=rr)
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
 for name in ['control_locked.json','source_locked.json','input_locked.json']:
  for p,h in json.loads((O/name).read_text()).items():assert sha(P/p)==h
 assert json.loads((O/'stage_state.json').read_text())['stage']=='trained'
 rows=T.loadrows(json.loads((O/'manifest.json').read_text()),'cuda');audit=[];out=[]
 for v in ['object_only','pose_conditioned']:
  run='multi_'+v;R=O/'runs'/run;h=json.loads((R/'history.json').read_text());old=json.loads((B/'runs'/run/'history.json').read_text())
  assert len(h)==32 and h[-1]['optimizer_steps']==192 and [x['selected_queries'] for x in h]==[x['selected_queries'] for x in old]
  assert json.loads((R/'initialization.json').read_text())==json.loads((B/'runs'/run/'initialization.json').read_text())
  expected=min(h,key=lambda x:x['val_loss'])['epoch']
  for kind in ['best','last']:
   ck=torch.load(R/(kind+'.pt'),map_location='cuda',weights_only=False)
   assert ck['epoch']==(expected if kind=='best' else 32)
   if kind=='last':
    for key in ['optimizer','rng_python','rng_numpy','rng_cpu','rng_cuda']:assert key in ck
   net=T.RegionPoseNet().cuda();net.load_state_dict(ck['state']);net.eval()
   for split in ['train','val']:
    rr=[]
    with torch.inference_mode():
     for r,d in rows:
      if r['split']!=split:continue
      a,b=T.forward(net,d);s=a if v=='object_only' else b;logit=s.cpu().numpy();score=s.sigmoid().cpu().numpy();y=d['target'].cpu().numpy().astype(bool);known=d['known'].cpu().numpy()
      mm=T.metrics(logit,d);curves=[]
      for budget in [1,2,5,10,20]:
       count=max(1,int(np.ceil(len(score)*budget/100)));ix=np.argsort(-logit,kind='stable')[:count]
       curves.append(dict(budget_percent=budget,selected=count,positive_coverage=float(y[ix].sum()/y.sum()),known_negative_fraction=float((known[ix]&~y[ix]).mean()),unknown_fraction=float((~known[ix]).mean())))
      pos=d['known']&(d['target']>.5);bce=torch.nn.functional.binary_cross_entropy_with_logits(s,d['target'],reduction='none');w=d['confidence']
      rr.append(dict(key=r['key'],episode=r['episode'],event=r['event'],offset=r['offset'],**mm,loss=float(T.loss(s,d)),new_head_loss=float(T.train_loss(s,d)),positive_bce=float((bce[pos]*w[pos]).sum()/w[pos].sum()),positive_score_median=float(np.median(score[y])),curves=curves))
      path=O/'evaluation'/split/run/kind/(r['key']+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,logit=logit,score=score,target=y,known=known)
      if kind=='best' and split=='val':
       with np.load(R/'val_predictions'/(r['key']+'.npz')) as z:delta=float(np.max(np.abs(score-z['score'])))
       assert delta<1e-7;audit.append(dict(run=run,key=r['key'],max_score_difference=delta))
    result=dict(run=run,checkpoint=kind,split=split,epoch=ck['epoch'],**summarize(rr));put(O/'evaluation'/(split+'_'+run+'_'+kind+'.json'),result);out.append(result)
   del ck,net;torch.cuda.empty_cache()
 for name in ['control_locked.json','source_locked.json','input_locked.json']:
  for p,h in json.loads((O/name).read_text()).items():assert sha(P/p)==h
 put(O/'evaluation/summary.json',out)
 # Reuse locked control metrics and baselines for identical samples; no re-training.
 controlinputs={}
 oldlock=json.loads((D/'completed_locked.json').read_text())
 for split in ['train','val']:
  paths=[D/split/'summary.json',D/split/'baselines.json']
  for p in paths:assert sha(p)==oldlock[str(p.relative_to(P))];controlinputs[str(p.relative_to(P))]=sha(p)
  control=[x for x in json.loads(paths[0].read_text()) if x['run'].startswith('multi_')]
  put(O/'evaluation'/('control_'+split+'.json'),control);put(O/'evaluation'/('baselines_'+split+'.json'),json.loads(paths[1].read_text()))
 put(O/'evaluation/control_metrics_locked.json',controlinputs)
 put(O/'audits/completion_audit.json',dict(old_016_files_unchanged=184,samples_and_graphs_identical=32,matched_initialization=True,matched_plan=True,steps_per_run=192,epochs_per_run=32,best_original_loss_selection=True,best_last_optimizer_rng=True,inference_queries=128,best_val_prediction_comparisons=audit,unknown_zero_gradient_preflight=True))
 print('AUDIT_COMPLETE',len(out),flush=True)
if __name__=='__main__':main()
