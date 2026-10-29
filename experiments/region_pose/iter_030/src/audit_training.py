from pathlib import Path
import json,hashlib,ast,datetime,traceback,collections,math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit
R=Path('/mnt/nvme1/drh/D_VLA'); P=R/'experiments/region_pose/iter_030'; O=R/'experiments/region_pose/iter_029'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
 cfg=json.loads((O/'config.json').read_text()); rows=[x for x in json.loads((O/'manifest.json').read_text())['rows'] if x['split']=='train']
 assert len(rows)==40 and len(set(x['episode'] for x in rows))==10
 runs=[x['group']+'_'+x['variant'] for x in cfg['runs']]
 inputs=[O/'manifest.json',O/'config.json',O/'src/train.py',O/'audits/completion_audit.json']
 for run in runs:
  inputs += [O/'plans'/f'{run}.json', O/'runs'/run/'history.json']
 for row in rows:
  inputs += [O/'samples'/f"{row['key']}.npz",Path(row['target_source'])]
  for run in runs:
   for snap in ['best','last']:inputs.append(O/'evaluation'/run/snap/f"{row['key']}.npz")
 oldlocks={}
 for it in ['029','027']:oldlocks.update(json.loads((R/f'experiments/region_pose/iter_{it}/completed_locked.json').read_text()))
 locked={}
 for p in inputs:
  rel=str(p.relative_to(R));h=sha(p);assert oldlocks[rel]==h,rel;locked[rel]=h
 dump(P/'input_locked.json',locked)
 dump(P/'source_locked.json',{str(p.relative_to(P)):sha(p) for p in [P/'src/audit_training.py',P/'config.json',P/'设计方案.md']})
 tree=ast.parse((O/'src/train.py').read_text());fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='loss')
 env={'torch':torch,'F':F};exec(compile(ast.Module(body=[fun],type_ignores=[]),'frozen_loss','exec'),env); loss=env['loss']
 trainmeta=json.loads((O/'audits/completion_audit.json').read_text())['training']
 dump(P/'input_slices/training_metadata.json',trainmeta)
 samples={}; recon=[]
 for row in rows:
  key=row['key'];d=dict(np.load(O/'samples'/f'{key}.npz',allow_pickle=False));z=dict(np.load(row['target_source'],allow_pickle=False));ix=d['causal_fps_indices']
  for field in ['target','known','confidence','reference_source_index']:assert np.array_equal(d[field],z[field][ix]),(key,field)
  assert np.array_equal(d['xyz_world'],z['query_xyz'][ix].astype(np.float32))
  assert np.array_equal(d['pixel'],z['query_pixel'][ix])
  assert np.array_equal(ix,z['selected_index'])
  pos=d['known']&d['target'];neg=d['known']&~d['target']
  assert len(np.unique(d['reference_source_index'][pos]))>=3 and len(np.unique(d['reference_source_index'][neg]))>=16
  assert np.all(d['confidence'][~d['known']]==0)
  samples[key]=d;recon.append({'key':key,'positive_sources':int(len(np.unique(d['reference_source_index'][pos]))),'negative_sources':int(len(np.unique(d['reference_source_index'][neg]))),'equal':True})
 dump(P/'audits/source_reconstruction.json',recon)
 history_audit=[]
 for run in runs:
  h=json.loads((O/'runs'/run/'history.json').read_text())
  sliced=[{k:x[k] for k in ['epoch','train_loss','selected_queries','optimizer_steps']} for x in h]
  dump(P/'input_slices'/f'{run}_history_train_only.json',sliced)
  plan=json.loads((O/'plans'/f'{run}.json').read_text())
  actual=[dict(epoch=x['epoch'],**s) for x in sliced for s in x['selected_queries']]
  assert actual==plan and len(plan)==240
  counts=collections.Counter((x['event'],x['offset']) for x in plan)
  n=6 if run.startswith('control') else 10;expected=10 if n==6 else 6
  assert len(counts)==4*n and set(counts.values())=={expected}
  for x in sliced:assert x['optimizer_steps']==x['epoch']*n
  cycles=[{'epoch_start':i+1,'epoch_end':i+4,'steps_end':(i+4)*n,'mean_total_objective':float(np.mean([x['train_loss'] for x in sliced[i:i+4]]))} for i in range(0,len(sliced),4)]
  history_audit.append({'run':run,'steps':240,'events':n,'exposure_per_offset':expected,'plan_exact':True,'counts':[{'event':e,'offset':o,'count':c} for (e,o),c in sorted(counts.items())],'cycles':cycles,'per_event_historical_loss_available':False,'first24epochs_steps':24*n,'first24epochs_exposure_per_event':24,'first24epochs_mean_total_objective':float(np.mean([x['train_loss'] for x in sliced[:24]]))})
 dump(P/'audits/exposure_history.json',history_audit)
 def metrics(score,d):
  known=d['known'];y=d['target'];pos=known&y;neg=known&~y;npos=int(pos.sum())
  order=np.argsort(-score[known],kind='stable');yy=y[known][order]
  ap=float(np.sum((np.cumsum(yy)/(np.arange(len(yy))+1))*yy)/npos)
  knowncov={str(b):float(yy[:max(1,int(math.ceil(b*len(yy))))].sum()/npos) for b in [.1,.2,.3]}
  order=np.argsort(-score,kind='stable');budget={}
  for b in [.01,.02,.05,.1,.2,.3]:
   sel=order[:max(1,int(math.ceil(b*len(score))))]
   budget[str(b)]={'coverage':float(pos[sel].sum()/npos),'unknown':float((~known[sel]).mean()),'negative':float(neg[sel].mean())}
  return {'ap':ap,'known_coverage':knowncov,'all_budget':budget}
 def aggregate(rr):
  return {'n_queries':len(rr),'events':len(set(x['episode'] for x in rr)),'mean':{k:float(np.mean([x[k] for x in rr])) for k in ['ap','head_loss','positive_bce','negative_bce','rank','positive_bce_fraction','inverting_negatives','rank_inverting_negatives','negative_top1_bce_fraction','negative_top1_confidence_fraction']},'known_coverage':{b:float(np.mean([x['known_coverage'][b] for x in rr])) for b in ['0.1','0.2','0.3']},'all_budget':{b:{k:float(np.mean([x['all_budget'][b][k] for x in rr])) for k in ['coverage','unknown','negative']} for b in ['0.01','0.02','0.05','0.1','0.2','0.3']}}
 checks={'max_loss_difference':0.,'max_gradient_difference':0.,'max_rank_gradient_sum':0.,'unknown_gradient_max':0.}
 allrows=[]
 for run in runs:
  for snap in ['best','last']:
   rr=[]
   outdir=P/'points'/run/snap;outdir.mkdir(parents=True,exist_ok=False)
   for row in rows:
    key=row['key'];d=samples[key];p=dict(np.load(O/'evaluation'/run/snap/f'{key}.npz',allow_pickle=False))
    for f in ['known','target']:assert np.array_equal(p[f],d[f])
    l=p['logit'].astype(np.float64);known=d['known'];target=d['target'];pos=known&target;neg=known&~target;w=d['confidence'].astype(np.float64)
    logit=torch.tensor(p['logit'],device='cuda',requires_grad=True)
    td={k:torch.tensor(d[k],device='cuda') for k in ['known','target','confidence']};td['target']=td['target'].float()
    original=loss(logit,td);original.backward()
    ip=torch.nonzero(td['known']&(td['target']>.5)).flatten();ineg=torch.nonzero(td['known']&~(td['target']>.5)).flatten()
    ip=ip[torch.linspace(0,len(ip)-1,min(64,len(ip)),device='cuda').long()].cpu().numpy()
    ni=ineg[torch.linspace(0,len(ineg)-1,min(64,len(ineg)),device='cuda').long()].cpu().numpy()
    weight=np.zeros(len(l));weight[pos]=.5*w[pos]/max(w[pos].sum(),1e-8);weight[neg]=.5*w[neg]/max(w[neg].sum(),1e-8)
    b=np.logaddexp(0,l)-target*l;bc=weight*b;bg=weight*(expit(l)-target)
    diff=l[ni][None,:]-l[ip][:,None];mat=np.logaddexp(0,diff);rank=float(mat.mean());g=.1*expit(diff)/diff.size
    rg=np.zeros(len(l));rg[ni]=g.sum(axis=0);rg[ip]=-g.sum(axis=1)
    rc=np.zeros(len(l));rc[ni]=.05*mat.mean(axis=0)/len(ni);rc[ip]=.05*mat.mean(axis=1)/len(ip)
    total=float(bc.sum()+.1*rank); grad=bg+rg
    ld=abs(total-original.item());gd=float(np.max(np.abs(grad-logit.grad.cpu().numpy())))
    assert ld<2e-6 and gd<2e-6,(key,ld,gd)
    assert np.all(grad[~known]==0) and np.all(logit.grad.cpu().numpy()[~known]==0)
    assert abs(rc.sum()-.1*rank)<1e-10
    checks['max_loss_difference']=max(checks['max_loss_difference'],ld);checks['max_gradient_difference']=max(checks['max_gradient_difference'],gd);checks['max_rank_gradient_sum']=max(checks['max_rank_gradient_sum'],float(abs(rg.sum())))
    inv=neg&(l>=l[pos].min());tail=np.flatnonzero(neg)[np.argsort(-l[neg])[:max(1,int(math.ceil(.01*neg.sum())))]]
    fullrank=float(np.logaddexp(0,l[neg][None,:]-l[pos][:,None]).mean())
    score=p['score'];q=lambda a:[float(x) for x in np.quantile(a,[0,.1,.5,.9,1])]
    r={'run':run,'snapshot':snap,'episode':row['episode'],'event':row['event'],'key':key,'offset':int(row['offset']),'trained_event':bool(run.startswith('expanded') or row['episode'] in cfg['control_episodes']),'head_loss':total,'positive_bce':float(2*bc[pos].sum()),'negative_bce':float(2*bc[neg].sum()),'rank':rank,'rank_full_pair_diagnostic':fullrank,'positive_bce_fraction':float(bc[pos].sum()/max(total,1e-30)),'inverting_negatives':int(inv.sum()),'rank_inverting_negatives':int(inv[ni].sum()),'negative_top1_count':len(tail),'negative_top1_bce_fraction':float(bc[tail].sum()/max(bc[neg].sum(),1e-30)),'negative_top1_confidence_fraction':float(w[tail].sum()/w[neg].sum()),'positive_score_quantiles':q(score[pos]),'negative_score_quantiles':q(score[neg]),'positive_count':int(pos.sum()),'negative_count':int(neg.sum()),**metrics(score,d)}
    np.savez_compressed(outdir/f'{key}.npz',logit=p['logit'],score=score,target=target,known=known,confidence=w,pixel=d['pixel'],xyz_world=d['xyz_world'],reference_source_index=d['reference_source_index'],bce_contribution=bc,rank_contribution=rc,bce_gradient=bg,rank_gradient=rg,total_gradient=grad,rank_positive_index=ip,rank_negative_index=ni,inverting_negative=inv)
    rr.append(r);allrows.append(r)
   trained=[x for x in rr if x['trained_event']]
   summary={'run':run,'snapshot':snap,'trained_only':aggregate(trained),'all10_diagnostic':aggregate(rr),'episodes':{e:aggregate([x for x in rr if x['episode']==e]) for e in sorted(set(x['episode'] for x in rr))},'offsets_trained':{str(o):aggregate([x for x in trained if x['offset']==o]) for o in [15,10,5,1]},'queries':rr}
   dump(P/'summaries'/f'{run}_{snap}.json',summary)
   print(json.dumps({'completed':run+'_'+snap,'train_ap':summary['trained_only']['mean']['ap'],'train_loss':summary['trained_only']['mean']['head_loss']}),flush=True)
 dump(P/'summary.json',{'scope':'training_only_frozen_predictions','queries':40,'events':10,'snapshots':8,'point_files':320,'numerical_checks':checks,'source_reconstruction_queries':40,'no_training':True,'no_validation_analysis':True,'no_test_access':True,'stage2_accepted':False,'history_event_losses_unavailable':True})
 dump(P/'audits/numerical_reconstruction.json',checks)
 dump(P/'audits/access_boundary.json',{'read_inputs':list(locked),'excluded':'validation samples/predictions/images and tests; history only training fields analyzed','manifest_metadata_only':True,'completion_metadata_only':True,'no_network_forward':True})
 assert all(sha(R/k)==v for k,v in locked.items())
 dump(P/'audits/input_integrity.json',{'files':len(locked),'unchanged':True,'prior_selected_hashes_verified':True})
 dump(P/'stage_state.json',{'stage':'analysis_complete','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
if __name__=='__main__':
 try:main()
 except Exception as e:
  (P/'logs/error_traceback_remote_only.txt').write_text(traceback.format_exc())
  dump(P/'failure.json',{'type':type(e).__name__,'message':str(e)[:400]})
  print(json.dumps({'error_type':type(e).__name__,'error_message':str(e)[:400]}),flush=True)
  raise SystemExit(1)
