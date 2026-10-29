from pathlib import Path
import sys,json,hashlib,datetime,traceback,importlib.util
import numpy as np,torch
from torch.nn import functional as F
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/contact_frames/cf003';P=R/'experiments/contact_frames/cf001_frozen_review_002'
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def module(arm):
 spec=importlib.util.spec_from_file_location('model_'+arm,B/f'src/model_{arm}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.ContactFrameNet
def tensorhash(state):
 h=hashlib.sha256()
 for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def context(d):
 n=len(d['train']);block=d['block'];return np.array([np.maximum(np.arange(t-8,t+1),max(int(np.flatnonzero(block==block[t])[0]),1 if t>0 else 0)) for t in range(n)])
def predict(m,d,st,ix):
 pred=[]
 with torch.no_grad():
  for i in range(0,len(ix),16):
   a=ix[i:i+16];z,_=m(torch.as_tensor(d['images'][a],device='cuda'),torch.as_tensor(d['points'][a],device='cuda'),torch.as_tensor(st[a],device='cuda'))
   pred.extend(torch.sigmoid(z).cpu().tolist())
 return np.array(pred,dtype='float32')
def runs(m):
 a=np.flatnonzero(m);return [[int(x[0]),int(x[-1])] for x in np.split(a,np.flatnonzero(np.diff(a)>1)+1) if len(x)]
def weak_metrics(pred,d,mask=None):
 if mask is None:mask=d['state_valid']
 mean=pred.mean(0) if pred.ndim==2 else pred;q=np.clip(mean.astype('float64'),1e-7,1-1e-7);w=d['weight']*mask;y=d['target']
 neg=(d['weight']>0)&(y<=.2)&mask
 low=pred.min(0) if pred.ndim==2 else pred
 return {'weighted_bce_per_all_frames':float(np.sum(w*(-y*np.log(q)-(1-y)*np.log(1-q)))/max(int(mask.sum()),1)),'weighted_bce_known_weight':float(np.sum(w*(-y*np.log(q)-(1-y)*np.log(1-q)))/max(w.sum(),1e-8)),'weak_negative_frames':int(neg.sum()),'high_on_weak_negative':int(((low>=.8)&neg).sum()),'raw_high':int(((low>=.8)&mask).sum())}
def main():
 cfg=json.loads((B/'config.json').read_text());torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.12)
 data={ep:dict(np.load(B/'data'/ep/'artifacts/dataset.npz')) for ep in cfg['train_episodes']}
 inputlocks={str((B/'data'/ep/'artifacts/dataset.npz').resolve().relative_to(R)):hashlib.sha256((B/'data'/ep/'artifacts/dataset.npz').read_bytes()).hexdigest() for ep in data}
 sourcepaths=[B/'config.json',*list((B/'src').glob('*.py'))]
 inputlocks.update({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcepaths})
 save(B/'training_inputs_locked.json',inputlocks)
 masks={ep:d['train']&d['state_valid'] for ep,d in data.items()}
 cat=np.concatenate([d['states'][masks[ep]] for ep,d in data.items()])
 mean=cat.mean(0);std=np.maximum(cat.std(0),.01);save(B/'normalization.json',{'mean':mean.tolist(),'std':std.tolist(),'source':'four training splits only; std floor .01'})
 zstates={ep:((d['states']-mean)/std).astype('float32') for ep,d in data.items()};indices={ep:context(d) for ep,d in data.items()}
 coverage={ep:{'frames':len(d['target']),'train_frames':int(masks[ep].sum()),'train_anchor_frames':int(((d['weight']>0)&masks[ep]).sum()),'train_weak_positive_ge08':int(((d['target']>=.8)&(d['weight']>0)&masks[ep]).sum()),'train_weak_negative_le02':int(((d['target']<=.2)&(d['weight']>0)&masks[ep]).sum())} for ep,d in data.items()};save(B/'training_coverage.json',coverage)
 test=torch.tensor([.3,-.2],requires_grad=True);w=torch.tensor([1.,0.]);((F.binary_cross_entropy_with_logits(test,torch.tensor([1.,.5]),reduction='none')*w).mean()).backward();assert test.grad[1]==0
 init={};trainresults=[]
 for seed in cfg['seeds']:
  rng=np.random.default_rng(seed);plan=[]
  for step in range(cfg['steps']):
   ep=cfg['train_episodes'][step%4];pool=np.flatnonzero(masks[ep]);a=rng.choice(pool,size=min(16,len(pool)),replace=False);plan.append({'episode':ep,'frames':a.tolist()})
  save(B/f'plans/seed_{seed}.json',plan)
  for arm in cfg['arms']:
   torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);m=module(arm)(50).cuda();initial=tensorhash(m.state_dict())
   if seed in init:assert init[seed]==initial
   else:init[seed]=initial
   out=B/f'runs/{arm}/seed_{seed}';out.mkdir(parents=True)
   torch.save({'model':m.state_dict(),'seed':seed,'initial_tensor_hash':initial},out/'initial.pt')
   opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);history=[]
   for step,row in enumerate(plan,1):
    ep=row['episode'];d=data[ep];a=np.array(row['frames']);ix=indices[ep][a]
    m.train();opt.zero_grad(set_to_none=True)
    logits,dy=m(torch.as_tensor(d['images'][ix],device='cuda'),torch.as_tensor(d['points'][ix],device='cuda'),torch.as_tensor(zstates[ep][ix],device='cuda'))
    y=torch.as_tensor(d['target'][a],device='cuda');w=torch.as_tensor(d['weight'][a],device='cuda');dv=torch.as_tensor(d['dynvalid'][a],device='cuda')
    bce=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).mean();aux=(F.smooth_l1_loss(dy,torch.as_tensor(d['dy'][a],device='cuda'),reduction='none').mean(-1)*dv).mean();loss=bce+.2*aux
    assert torch.isfinite(loss)
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
    history.append({'step':step,'episode':ep,'bce':float(bce),'aux':float(aux),'loss':float(loss)})
    if step%100==0:save(B/'stage_state.json',{'stage':'training','arm':arm,'seed':seed,'step':step,'time':now()})
   save(out/'history.json',history)
   torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'steps':600,'state_mean':mean,'state_std':std,'config':cfg,'initial_tensor_hash':initial,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},out/'last.pt')
   m.eval();metrics={}
   for ep,d in data.items():
    pred=predict(m,d,zstates[ep],indices[ep]);np.savez_compressed(out/f'{ep}_prediction.npz',probability=pred)
    metrics[ep]={k:weak_metrics(pred,d,d[k]&d['state_valid']) for k in ['train','audit']}
   trainresults.append({'arm':arm,'seed':seed,'episodes':metrics})
   save(B/'training_summary.json',trainresults)
   del m,opt
 save(B/'audits/paired_training.json',{'initial_tensor_hashes':init,'plans_shared_between_arms':True,'unknown_bce_gradient_zero':True,'steps_per_run':600,'runs':6})
 weights={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in B.glob('runs/*/seed_*/last.pt')};save(B/'evaluation_weights_locked.json',weights)
 allcomparisons=[]
 for arm in cfg['arms']:
  summaries=[];edir=B/f'evaluation_{arm}';edir.mkdir()
  save(edir/'plan.json',{'cases':cfg['fixed_review_episodes']})
  for ep in cfg['fixed_review_episodes']:
   c=edir/ep;(c/'artifacts').mkdir(parents=True);(c/'artifacts/dataset.npz').symlink_to(P/ep/'artifacts/dataset.npz');(c/'config.json').write_bytes((P/ep/'config.json').read_bytes())
   d=dict(np.load(c/'artifacts/dataset.npz'));control=np.load(P/ep/'artifacts/frame_predictions.npz');ix=control['context_indices'];st=((d['states']-mean)/std).astype('float32');ps=[]
   for seed in cfg['seeds']:
    ck=torch.load(B/f'runs/{arm}/seed_{seed}/last.pt',map_location='cpu',weights_only=False);m=module(arm)(50).cuda().eval();m.load_state_dict(ck['model']);ps.append(predict(m,d,st,ix));del m
   p=np.array(ps);lo=p.min(0);hi=p.max(0);valid=d['geometry_reliable']&d['state_valid'];pos=(lo>=.8)&valid;neg=(hi<=.2)&valid&((d['distance']-2*d['sigma'])>.006);unk=~(pos|neg)
   np.savez_compressed(c/'artifacts/frame_predictions.npz',probabilities=p,mean=p.mean(0),low=lo,high=hi,contact_candidate=pos,observed_separation_candidate=neg,unknown=unk,context_indices=ix)
   s={'episode':ep,'arm':arm,'frames':len(lo),'contact_candidate':int(pos.sum()),'separation_candidate':int(neg.sum()),'unknown':int(unk.sum()),'contact_ranges':runs(pos),'trained_on_this_video':False};save(c/'inference_summary.json',s);summaries.append(s)
   row={'arm':arm,'episode':ep,'old':weak_metrics(control['probabilities'],d),'new':weak_metrics(p,d),'old_spread_mean':float((control['high']-control['low']).mean()),'new_spread_mean':float((hi-lo).mean())}
   if ep=='porcelain_mug_demo_9':
    a=np.arange(56,96);row['review_56_95']={'seed_high_counts':(p[:,a]>=.8).sum(1).tolist(),'unanimous_high':int((lo[a]>=.8).sum()),'candidate':int(pos[a].sum()),'separation':int(neg[a].sum()),'unknown':int(unk[a].sum()),'geometry_reliable':int(valid[a].sum())}
   allcomparisons.append(row)
  save(edir/'summary.json',summaries)
 save(B/'comparison.json',allcomparisons)
 assert all(hashlib.sha256((R/k).read_bytes()).hexdigest()==v for k,v in inputlocks.items())
 assert all(hashlib.sha256((R/k).read_bytes()).hexdigest()==v for k,v in weights.items())
 save(B/'stage_state.json',{'stage':'evaluation_complete','time':now()})
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(B/'training_failure.json',{'type':type(e).__name__,'message':str(e),'stack':[{'file':x.filename,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(e.__traceback__)]});raise SystemExit(1)
