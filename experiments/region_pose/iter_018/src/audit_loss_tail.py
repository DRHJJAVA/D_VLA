from pathlib import Path
import sys,json,hashlib,time
import numpy as np,torch
from scipy.special import expit
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose/iter_016';C=P/'experiments/region_pose/iter_017';O=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B/'src'))
import train as T
import supervision as S
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def verify(lock):
 for p,h in lock.items():assert sha(P/p)==h,p
def stats(x):
 return dict(n=len(x),mean=float(np.mean(x)),min=float(np.min(x)),p05=float(np.quantile(x,.05)),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(np.max(x))) if len(x) else dict(n=0)
def subset(ids):
 return ids[torch.linspace(0,len(ids)-1,min(64,len(ids)),device='cuda').long().cpu().numpy()]
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
 inputs=json.loads((O/'input_locked.json').read_text());source=json.loads((O/'source_locked.json').read_text());verify(inputs);verify(source)
 manifest=json.loads((O/'train_manifest.json').read_text())
 references=json.loads((P/'experiments/region_pose/iter_013/targets/train/summary.json').read_text())['rows']
 sampleaudit=[];cache={}
 for r in manifest:
  assert r['split']=='train'
  with np.load(B/'samples'/(r['key']+'.npz')) as z:sample={k:z[k] for k in z.files}
  with np.load(r['target_source']) as z:original={k:z[k] for k in z.files}
  y,k,w,idx=S.targets(r,original);fps=sample['causal_fps_indices']
  for name,v in [('target',y),('known',k),('confidence',w),('reference_source_index',idx)]:assert np.array_equal(sample[name],v[fps]),(r['key'],name)
  refpath=Path(next(x for x in references if x['event']==r['event'])['target_source'])
  with np.load(refpath) as z:ref={k:z[k] for k in z.files}
  ix=sample['reference_source_index'];known=sample['known'].astype(bool)
  assert np.all(ref['known'][ix[known]]) and np.array_equal(ref['target'][ix[known]],sample['target'][known])
  assert np.all(sample['confidence'][known]<=ref['confidence'][ix[known]]+1e-6) and np.all(sample['confidence'][~known]==0)
  sampleaudit.append(dict(key=r['key'],points=len(ix),positive=int(sample['target'].sum()),negative=int((known&~sample['target'].astype(bool)).sum()),unknown=int((~known).sum()),unique_positive_sources=len(np.unique(ix[sample['target'].astype(bool)])),all_source_arrays_equal=True))
  cache[r['key']]=(sample,ref)
 put(O/'audits/source_reconstruction.json',sampleaudit)
 prior=json.loads((C/'train/summary.json').read_text());allrows=[];errors=[]
 for model in prior:
  run=model['run'];kind=model['checkpoint'];rows=[]
  for m in model['rows']:
   r=next(x for x in manifest if x['key']==m['key']);s,ref=cache[r['key']]
   with np.load(C/'train/predictions'/run/kind/(r['key']+'.npz')) as z:logit=z['logit'].astype(np.float64);assert np.array_equal(z['known'],s['known']) and np.array_equal(z['target'],s['target'])
   y=s['target'].astype(bool);known=s['known'].astype(bool);w=s['confidence'].astype(np.float64);ip=np.flatnonzero(y);ineg=np.flatnonzero(known&~y);sp=subset(ip);sn=subset(ineg)
   bce=np.logaddexp(0,logit)-y*logit
   contribution=np.zeros(len(y));contribution[ip]=.5*bce[ip]*w[ip]/w[ip].sum();contribution[ineg]=.5*bce[ineg]*w[ineg]/w[ineg].sum()
   bcegrad=np.zeros(len(y));bcegrad[ip]=.5*(expit(logit[ip])-1)*w[ip]/w[ip].sum();bcegrad[ineg]=.5*expit(logit[ineg])*w[ineg]/w[ineg].sum()
   diff=logit[sn][None,:]-logit[sp][:,None];rank=float(np.logaddexp(0,diff).mean())
   full=logit[ineg][None,:]-logit[ip][:,None];rankfull=float(np.logaddexp(0,full).mean())
   rg=np.zeros(len(y));rg[sp]=-.1*expit(diff).mean(axis=1)/len(sp);rg[sn]=.1*expit(diff).mean(axis=0)/len(sn)
   fg=np.zeros(len(y));fg[ip]=-.1*expit(full).mean(axis=1)/len(ip);fg[ineg]=.1*expit(full).mean(axis=0)/len(ineg)
   recomputed=float(contribution.sum()+.1*rank)
   d={key:torch.tensor(s[key],device='cuda',dtype=torch.bool if key=='known' else torch.float32) for key in ['target','known','confidence']}
   actual=float(T.loss(torch.tensor(logit,device='cuda',dtype=torch.float32),d))
   err=max(abs(recomputed-actual),abs(actual-m['loss_head']));errors.append(err);assert err<2e-5,(r['key'],err)
   above=ineg[logit[ineg]>=logit[ip].min()];abovemed=ineg[logit[ineg]>=np.median(logit[ip])]
   tails=[]
   for percent in [1,5,10]:
    ix=ineg[np.argsort(-logit[ineg],kind='stable')[:max(1,int(np.ceil(len(ineg)*percent/100)))]]
    tails.append(dict(negative_top_percent=percent,count=len(ix),selected_by_rank_subset=int(np.isin(ix,sn).sum()),subset_coverage=float(np.isin(ix,sn).mean()),negative_bce_share=float(contribution[ix].sum()/max(1e-300,contribution[ineg].sum())),negative_confidence_mass_share=float(w[ix].sum()/w[ineg].sum()),bce_gradient_mass=float(bcegrad[ix].sum()),sample_rank_gradient_mass=float(rg[ix].sum()),full_rank_gradient_mass=float(fg[ix].sum()),unique_reference_sources=len(np.unique(s['reference_source_index'][ix]))))
   detail=[]
   for i in ineg[np.argsort(-logit[ineg],kind='stable')[:10]]:
    j=int(s['reference_source_index'][i])
    detail.append(dict(index=int(i),logit=float(logit[i]),score=float(expit(logit[i])),confidence=float(w[i]),in_rank_subset=bool(i in sn),weighted_bce_contribution=float(contribution[i]),rank_gradient=float(rg[i]),query_pixel=s['pixel'][i].tolist(),query_xyz_world=s['xyz_world'][i].tolist(),reference_index=j,reference_pixel=ref['query_pixel'][j].tolist(),reference_xyz=ref['query_xyz'][j].tolist(),reference_confidence=float(ref['confidence'][j]),reference_known=bool(ref['known'][j]),reference_target=bool(ref['target'][j])))
   x=dict(run=run,checkpoint=kind,epoch=model['epoch'],key=r['key'],episode=r['episode'],event=r['event'],offset=r['offset'],ap=m['ap'],coverage10=m['coverage10'],coverage20=m['coverage20'],coverage30=m['coverage30'],valid_fraction=m['valid_fraction'],loss_head=actual,positive_bce_weighted=float(2*contribution[ip].sum()),negative_bce_weighted=float(2*contribution[ineg].sum()),positive_bce_unweighted=float(bce[ip].mean()),negative_bce_unweighted=float(bce[ineg].mean()),sample_rank=rank,full_rank=rankfull,rank_full_minus_sample=rankfull-rank,positive_count=len(ip),negative_count=len(ineg),unknown_count=int((~known).sum()),negative_outranking_any_positive=len(above),negative_outranking_median_positive=len(abovemed),outranking_in_subset=int(np.isin(above,sn).sum()),outranking_subset_coverage=float(np.isin(above,sn).mean()) if len(above) else None,outranking_bce_share=float(contribution[above].sum()/max(1e-300,contribution[ineg].sum())),positive_confidence=stats(w[ip]),negative_confidence=stats(w[ineg]),outranking_confidence=stats(w[above]),tails=tails,top10_known_negatives=detail)
   path=O/'points'/run/kind/(r['key']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(path,logit=logit,target=y,known=known,confidence=w,reference_source_index=s['reference_source_index'],query_pixel=s['pixel'],query_xyz_world=s['xyz_world'],selected_positive_indices=sp,selected_negative_indices=sn,weighted_bce_contribution=contribution,bce_gradient=bcegrad,subset_rank_gradient=rg,full_rank_gradient=fg)
   rows.append(x);allrows.append(x)
  keys=['ap','loss_head','positive_bce_weighted','negative_bce_weighted','sample_rank','full_rank','negative_outranking_any_positive','outranking_in_subset']
  agg=lambda rr:{k:float(np.mean([x[k] for x in rr])) for k in keys}
  out=dict(run=run,checkpoint=kind,overall=agg(rows),episodes=[dict(episode=ep,**agg([x for x in rows if x['episode']==ep])) for ep in sorted({x['episode'] for x in rows})],offsets=[dict(offset=o,**agg([x for x in rows if x['offset']==o])) for o in [15,10,5,1]],rows=rows)
  put(O/'summaries'/(run+'_'+kind+'.json'),out);print(run,kind,out['overall'],flush=True)
 put(O/'summary.json',dict(rows=allrows,scope='training only, 6 events 24 queries, 8 checkpoint identities, no network inference or optimization'))
 verify(inputs);verify(source)
 put(O/'audits/completion_audit.json',dict(samples_reconstructed=24,prediction_diagnostics=192,source_files=len(source),input_files=len(inputs),max_loss_reconstruction_error=max(errors),no_validation_samples_read=True,no_training=True,unknown_supervision_zero=True,all_inputs_unchanged=True))
 put(O/'stage_state.json',dict(stage='complete',time=time.time()))
if __name__=='__main__':main()
