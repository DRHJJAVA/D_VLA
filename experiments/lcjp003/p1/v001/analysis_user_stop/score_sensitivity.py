import sys,pathlib,json,traceback,hashlib,datetime
P=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/p1/v001');A=P/'analysis_user_stop';A.mkdir(exist_ok=True)
sys.path.insert(0,str(P));sys.path.insert(0,str(P/'source'))
try:
 import torch,numpy as np
 from common import Dataset,batch,cpu_setup
 from model import LCJP003,ModelConfig
 locked=json.loads((P/'source_lock.json').read_text())
 assert all(hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in locked.items())
 cpu_setup();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ds=Dataset(cached=True);allpairs=json.loads((P/'evaluation_lock.json').read_text())['pairs']
 grouped={}
 for eid,t in allpairs:grouped.setdefault(eid,[]).append([eid,t])
 wanted=[0,1,2,3,45,46];pairs=[]
 for demo in wanted:
  seq=next(v for k,v in grouped.items() if k.endswith('__demo_'+str(demo)))
  pairs.extend([seq[2],seq[5]])
 m=LCJP003(ModelConfig()).cuda().eval();s=torch.load(P/'step3000.pt',map_location='cpu',weights_only=False);m.load_state_dict(s['model']);del s
 rows=[]
 for i,(eid,t) in enumerate(pairs):
  x,target,valid=batch(ds,[[eid,t]])
  noise=torch.randn(1,8,16,7,generator=torch.Generator(device='cpu').manual_seed(20261026+i)).cuda()
  with torch.inference_mode():base=m(x,noise,return_diagnostics=True)
  def err(out):
   d=(out['actions']-target[:,None]).abs()
   mask=valid[:,None,:,None].expand_as(d)
   return float(d[mask].mean().item())
  def arr(v):return v.detach().cpu().numpy()
  pv=arr(base['point_valid'][0]);scores=arr(base['scores'][0])[pv];alpha=arr(base['alpha'][0])[pv]
  ids=arr(x['object_id'][0,-1])[pv];stds=[]
  for oid in np.unique(ids):
   mask=ids==oid
   if mask.sum()>1:stds.append({'object_id':int(oid),'points':int(mask.sum()),'score_mean':float(scores[mask].mean()),'score_std':float(scores[mask].std()),'score_mass':float(scores[mask].sum())})
  row={'episode':eid,'t':t,'split':'dev' if eid.endswith(('__demo_45','__demo_46')) else 'train','baseline_mae':err(base),'score_sum':float(scores.sum()),'score_quantiles':np.quantile(scores,[0,.1,.5,.9,1]).tolist(),'alpha_quantiles':np.quantile(alpha,[0,.1,.5,.9,1]).tolist(),'beta':arr(base['beta'][0]).tolist(),'within_object':stds,'interventions':{}}
  for mode in ['uniform','zero_relation','zero_state','zero_text']:
   xx=dict(x)
   if mode=='zero_relation':xx['relation']=torch.zeros_like(x['relation'])
   if mode=='zero_state':xx['state']=torch.zeros_like(x['state'])
   if mode=='zero_text':xx['text_features']=torch.zeros_like(x['text_features'])
   with torch.inference_mode():out=m(xx,noise,return_diagnostics=True,uniform=(mode=='uniform'))
   row['interventions'][mode]={'mae':err(out),'action_abs_change_mean':float((out['actions']-base['actions']).abs().mean().item()),'action_abs_change_max':float((out['actions']-base['actions']).abs().max().item()),'score_total_variation':float((out['scores']-base['scores']).abs().sum().item()/2),'beta_total_variation':float((out['beta']-base['beta']).abs().sum().item()/2),'alpha_abs_change_mean':float((out['alpha']-base['alpha']).abs().mean().item())}
  rows.append(row)
  (A/'score_sensitivity_progress.json').write_text(json.dumps({'done':i+1,'total':len(pairs)}))
 summary={}
 for split in ['train','dev','all']:
  rr=rows if split=='all' else [r for r in rows if r['split']==split]
  baseerr=float(np.mean([r['baseline_mae'] for r in rr]));ss={'samples':len(rr),'baseline_mae':baseerr,'interventions':{}}
  for mode in rows[0]['interventions']:
   vals=[r['interventions'][mode] for r in rr];s={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
   s['mae_relative_change_pct']=100*(s['mae']/baseerr-1);ss['interventions'][mode]=s
  summary[split]=ss
 result={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'checkpoint_sha256':hashlib.sha256((P/'step3000.pt').read_bytes()).hexdigest(),'source_hashes_unchanged':all(hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in locked.items()),'scope':'12 fixed windows from train0-3/dev45-46, G8 fixed noise; no optimizer; diagnostic only, not official80-window metric','interpretation':'zero input interventions are out-of-distribution sensitivity tests; uniform is model-native intervention, not retraining; does not prove physically valid top-vs-side preference or score causality alone','summary':summary,'rows':rows}
 (A/'score_sensitivity.json').write_text(json.dumps(result,indent=2))
except Exception as e:
 (A/'score_sensitivity_failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e)[:300],'frames':[{'file':t.filename,'line':t.lineno} for t in traceback.extract_tb(e.__traceback__)]}))
