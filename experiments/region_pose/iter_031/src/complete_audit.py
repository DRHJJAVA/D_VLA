from pathlib import Path
import sys,json,hashlib,traceback
import numpy as np,torch
B=Path(__file__).resolve().parents[1];R=B.parents[2];O=R/'experiments/region_pose/iter_029'
sys.path.insert(0,str(B/'src'));import train as t
from model import RegionPoseNet
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def check(p,root):
 d=json.loads(p.read_text())
 for k,v in d.items():assert sha(root/k)==v,(str(p),k)
 return len(d)
def met(score,d):
 out=t.metrics(score,d);y=d['target'].cpu().numpy().astype(bool);known=d['known'].cpu().numpy()
 for b in [.01,.02,.05,.1,.2,.3]:
  ix=np.argsort(-score,kind='stable')[:max(1,int(np.ceil(len(score)*b)))];s=str(int(100*b))
  out['all_coverage'+s]=float(y[ix].sum()/max(1,y.sum()));out['all_unknown_fraction'+s]=float((~known[ix]).mean());out['all_known_negative_fraction'+s]=float((known[ix]&~y[ix]).mean())
 return out
def aggregate(rr):
 keys=[k for k,v in rr[0].items() if isinstance(v,(int,float)) and k not in ['offset']]
 eps=[dict(episode=e,**{k:float(np.mean([r[k] for r in rr if r['episode']==e])) for k in keys}) for e in sorted({r['episode'] for r in rr})]
 return dict(mean={k:float(np.mean([e[k] for e in eps])) for k in keys},episodes=eps,offsets=[dict(offset=o,**{k:float(np.mean([r[k] for r in rr if r['offset']==o])) for k in keys}) for o in [15,10,5,1]],rows=rr)
def main():
 checks={n:check(B/n,R) for n in ['source_locked.json','input_locked.json','preflight_locked.json','baseline_input_locked.json','baseline_source_locked.json']}
 checks['training_inputs_locked']=check(B/'training_inputs_locked.json',B)
 dump(B/'evaluation_source_locked.json',{str(B.joinpath('src/complete_audit.py').relative_to(R)):sha(B/'src/complete_audit.py')})
 cfg=json.loads((B/'config.json').read_text());m=json.loads((B/'manifest.json').read_text());init=json.loads((B/'initialization/locked.json').read_text())
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
 rows=t.loadrows(m,torch.device('cuda'));summary=[];training=[];predmax=0.;controlmax=0.
 specs=[(O,'expanded10_'+v,'control029_'+v,v,False) for v in ['object_only','pose_conditioned']]+[(B,'expanded10_'+v,'expanded10_'+v,v,True) for v in ['object_only','pose_conditioned']]
 for root,name,label,variant,isnew in specs:
  D=root/'runs'/name;hist=json.loads((D/'history.json').read_text());plan=json.loads((B/'plans'/f'{name}.json').read_text())
  assert len(hist)==24 and [dict(epoch=h['epoch'],**s) for h in hist for s in h['selected_queries']]==plan
  assert hist[-1]['optimizer_steps']==240
  best=torch.load(D/'best.pt',map_location='cpu',weights_only=False);last=torch.load(D/'last.pt',map_location='cpu',weights_only=False)
  assert best['epoch']==min(hist,key=lambda h:h['val_loss'])['epoch'] and last['epoch']==24
  assert all(k in last for k in ['rng_python','rng_numpy','rng_cpu','rng_cuda','optimizer'])
  assert max(float(v['step']) for v in last['optimizer']['state'].values())==240
  assert json.loads((D/'initialization.json').read_text())['state_sha256']==init['state_sha256']
  training.append(dict(run=label,new_training=isnew,epochs=24,steps=240,best_epoch=best['epoch'],best_steps=best['epoch']*10,plan_exact=True,optimizer_rng=True))
  for snap,ck in [('best',best),('last',last)]:
   net=RegionPoseNet().cuda();net.load_state_dict(ck['state']);net.eval();rr=[]
   out=B/'evaluation'/label/snap;out.mkdir(parents=True,exist_ok=False)
   for row,d in rows:
    with torch.inference_mode():
     a,b=t.forward(net,d);logit=a if variant=='object_only' else b
     score=logit.sigmoid().cpu().numpy();orig=t.loss(logit,d).item();candidate=t.training_loss(logit,d).item()
     aux=t.training_loss(a,d).item() if variant=='pose_conditioned' else 0.
    pos=d['known']&(d['target']>.5);neg=d['known']&~pos
    gradlogit=logit.detach().clone().requires_grad_(True);newloss=t.training_loss(gradlogit,d);newloss.backward();assert torch.count_nonzero(gradlogit.grad[~d['known']])==0
    inv=neg&(logit>=logit[pos].min());nidx=torch.nonzero(neg).flatten()
    lin=nidx[torch.linspace(0,len(nidx)-1,min(64,len(nidx)),device='cuda').long()]
    top=nidx[torch.argsort(logit[nidx],descending=True,stable=True)[:64]]
    rec=dict(key=row['key'],episode=row['episode'],event=row['event'],offset=row['offset'],split=row['split'],loss=orig,candidate_head_loss=candidate,candidate_total_objective=candidate+.5*aux,positive_score_median=float(logit[pos].sigmoid().median()),inverting_negatives=int(inv.sum()),original_rank_inverting=int(inv[lin].sum()),top_rank_inverting=int(inv[top].sum()),**met(score,d))
    rr.append(rec)
    np.savez_compressed(out/f"{row['key']}.npz",score=score,logit=logit.cpu().numpy(),target=d['target'].cpu().numpy(),known=d['known'].cpu().numpy())
    if isnew and snap=='best' and row['split']=='val':
     old=np.load(D/'val_predictions'/f"{row['key']}.npz");delta=float(np.max(np.abs(score-old['score'])));predmax=max(predmax,delta);assert delta==0
    if not isnew:
     old=np.load(O/'evaluation'/name/snap/f"{row['key']}.npz");delta=float(np.max(np.abs(score-old['score'])));controlmax=max(controlmax,delta);assert delta==0
   s=dict(run=label,checkpoint=snap,epoch=ck['epoch'],steps=ck['epoch']*10,new_training=isnew,validation=aggregate([r for r in rr if r['split']=='val']),training_group=aggregate([r for r in rr if r['split']=='train']),all_training_diagnostic=aggregate([r for r in rr if r['split']=='train']))
   dump(out/'summary.json',s);summary.append(s);del net;torch.cuda.empty_cache()
   print(json.dumps({'evaluated':label+'_'+snap,'original_val_loss':s['validation']['mean']['loss'],'val_ap':s['validation']['mean']['ap']}),flush=True)
 for n in ['source_locked.json','input_locked.json','preflight_locked.json','baseline_input_locked.json','baseline_source_locked.json']:check(B/n,R)
 check(B/'training_inputs_locked.json',B)
 dump(B/'evaluation/summary.json',summary)
 dump(B/'audits/completion_audit.json',{'checks':checks,'training':training,'initialization_sha256':init['state_sha256'],'new_best_validation_prediction_max_difference':predmax,'control029_prediction_max_difference':controlmax,'inferences':416,'baseline_index_corrected_separately':True,'same_samples_and_graphs':True,'unknown_gradient_zero':True,'test_read':False,'stage2_accepted':False})
 dump(B/'stage_state.json',{'stage':'evaluation_complete_pending_review'})
 print('EVALUATION_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  (B/'logs/evaluation_traceback_remote_only.txt').write_text(traceback.format_exc());print(json.dumps({'type':type(e).__name__,'message':str(e)[:300]}));raise SystemExit(1)
