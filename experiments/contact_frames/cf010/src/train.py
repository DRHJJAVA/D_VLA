from pathlib import Path
import json,hashlib,random,importlib.util,traceback,os
import numpy as np,torch
from datetime import datetime
from zoneinfo import ZoneInfo
B=Path('/mnt/nvme1/drh/D_VLA');BASE=B/'experiments/contact_frames';R=BASE/'cf010';OLD=BASE/'cf008'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
 cfg=json.loads((R/'config.json').read_text());norm=json.loads((BASE/'cf003/normalization.json').read_text())
 mu=np.asarray(norm['mean'],dtype='float32');sd=np.asarray(norm['std'],dtype='float32')
 assert np.all(sd>0)
 modelmod=module(R/'src/model.py','cf10');control=module(OLD/'src/model.py','cf8')
 locks={str(p):sha(p) for p in [R/'config.json',R/'src/model.py',R/'src/train.py',BASE/'cf003/normalization.json',OLD/'src/model.py']}
 data={};tensors={}
 for ep in cfg['train_episodes']:
  p=OLD/'data'/ep/'artifacts/dataset.npz';locks[str(p)]=sha(p)
  d=dict(np.load(p));data[ep]=d
  tensors[ep]={k:torch.as_tensor(v.astype('float32'),device='cuda') for k,v in d.items() if k in ['images','points','states','target','weight','near_target','near_weight','dy','dynvalid']}
  tensors[ep]['states']=torch.as_tensor((d['states']-mu)/sd,device='cuda')
 for seed in cfg['seeds']:
  for p in [OLD/'runs/evidence'/('seed_'+str(seed))/'initial.pt',R/'plans'/('seed_'+str(seed)+'.json')]:locks[str(p)]=sha(p)
 save(R/'input_locked.json',locks)
 save(R/'source_locked.json',{str(p):sha(p) for p in (R/'src').glob('*.py')})
 pre=[]
 for seed in cfg['seeds']:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  initial=torch.load(OLD/'runs/evidence'/('seed_'+str(seed))/'initial.pt',map_location='cpu',weights_only=False)
  model=modelmod.ContactFrameNet(50).cuda();model.load_state_dict(initial['model']);model.eval()
  legacy=control.ContactFrameNet(50).cuda();legacy.load_state_dict(initial['model']);legacy.eval()
  dt=tensors[cfg['train_episodes'][0]]
  args=[dt[k][None] for k in ['images','points','states']]
  with torch.no_grad():
   full=model(*args);past=model(*[x[:,:60] for x in args]);orig=legacy(*args)
   lastdiff=max(float((a[:,-1]-b).abs().max()) for a,b in zip(full,orig))
   prefixdiff=max(float((a[:,:60]-b).abs().max()) for a,b in zip(full,past))
  assert lastdiff<2e-5 and prefixdiff<2e-5,(lastdiff,prefixdiff)
  pre.append({'seed':seed,'full_last_vs_legacy_max_diff':lastdiff,'prefix_vs_full_max_diff':prefixdiff,'parameters':sum(p.numel() for p in model.parameters())})
  del legacy;torch.cuda.empty_cache()
  dest=R/'runs/full_history'/('seed_'+str(seed));dest.mkdir(parents=True)
  torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'seed':seed},dest/'initial.pt')
  opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
  plan=json.loads((R/'plans'/('seed_'+str(seed)+'.json')).read_text());assert len(plan)==600
  hist=[];model.train()
  for step,item in enumerate(plan,1):
   dt=tensors[item['episode']];ids=torch.as_tensor(item['frames'],device='cuda')
   e,aux,near=model(*[dt[k][None] for k in ['images','points','states']]);e=e[0,ids];aux=aux[0,ids];near=near[0,ids]
   mainloss=(torch.nn.functional.binary_cross_entropy_with_logits(e,dt['target'][ids],reduction='none')*dt['weight'][ids]).mean()
   nearloss=(torch.nn.functional.binary_cross_entropy_with_logits(near,dt['near_target'][ids],reduction='none')*dt['near_weight'][ids]).mean()
   dyn=(torch.nn.functional.smooth_l1_loss(aux,dt['dy'][ids],reduction='none').mean(-1)*dt['dynvalid'][ids]).mean()
   loss=mainloss+.5*nearloss+.2*dyn
   assert torch.isfinite(loss)
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
   hist.append({'step':step,'episode':item['episode'],'loss':float(loss.detach()),'main':float(mainloss.detach()),'near':float(nearloss.detach()),'dyn':float(dyn.detach())})
   if step%100==0:save(R/'stage_state.json',{'stage':'training','seed':seed,'step':step,'time':now()})
  torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'seed':seed,'steps':600,'state_mean':mu,'state_std':sd,'config':cfg,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},dest/'last.pt')
  save(dest/'history.json',hist)
 save(R/'audits/preflight.json',{'passed':True,'rows':pre})
 assert all(sha(p)==v for p,v in locks.items())
 weights={str(p):sha(p) for p in R.glob('runs/*/seed_*/last.pt')}
 save(R/'evaluation_weights_locked.json',weights);save(R/'pre_review_freeze.json',{'time':now(),'weights':weights,'human_ranges_used':False})
 rows=[]
 for ep in cfg['train_episodes']+cfg['fixed_review_episodes']:
  if ep in cfg['train_episodes']:src=OLD/'data'/ep/'artifacts/dataset.npz'
  elif ep in ['tomato_sauce_demo_3','alphabet_soup_demo_1']:src=BASE/'cf005_frozen_review_003'/ep/'artifacts/dataset.npz'
  else:src=BASE/'cf005/evaluation_all_train_frames'/ep/'artifacts/dataset.npz'
  d=dict(np.load(src));dt={k:torch.as_tensor(d[k],device='cuda') for k in ['images','points']}
  dt['states']=torch.as_tensor((d['states']-mu)/sd,device='cuda')
  ps=[];ns=[];prefixchecks=[]
  for seed in cfg['seeds']:
   m=modelmod.ContactFrameNet(50).cuda();m.load_state_dict(torch.load(R/'runs/full_history'/('seed_'+str(seed))/'last.pt',map_location='cuda',weights_only=False)['model']);m.eval()
   with torch.no_grad():
    e,a,n=m(*[dt[k][None] for k in ['images','points','states']])
    prefix=m(*[dt[k][None,:60] for k in ['images','points','states']])
    diff=max(float((v[:,:60]-w).abs().max()) for v,w in zip([e,a,n],prefix));assert diff<2e-5
    ps.append(torch.sigmoid(e[0]).cpu().numpy());ns.append(torch.sigmoid(n[0]).cpu().numpy());prefixchecks.append(diff)
   del m
  p=np.stack(ps);near=np.stack(ns)
  reliable=d['geometry_reliable']&d['state_valid']
  separation=reliable&(d['distance']-2*d['sigma']>.006)
  predicted=(p.min(0)>=.8)&~separation
  pos=predicted&reliable
  neg=(near.max(0)<=.2)&separation
  out=R/'evaluation'/ep/'artifacts';out.mkdir(parents=True)
  (out/'dataset.npz').symlink_to(src.resolve())
  np.savez_compressed(out/'frame_predictions.npz',probabilities=p,proximity_probabilities=near,mean=p.mean(0),low=p.min(0),high=p.max(0),contact_candidate=pos,observed_separation_candidate=neg,unknown=~(pos|neg),predicted_contact=predicted,predicted_unknown=~(predicted|neg),geometry_reliable=reliable,geometry_separation_evidence=separation)
  rows.append({'episode':ep,'frames':len(pos),'strict_candidate':int(pos.sum()),'predicted_contact':int(predicted.sum()),'separation':int(neg.sum()),'strict_unknown':int((~(pos|neg)).sum()),'prefix_checks':prefixchecks})
 assert all(sha(p)==v for p,v in locks.items()) and all(sha(p)==v for p,v in weights.items())
 save(R/'summary.json',{'rows':rows,'time':now(),'selection':'last600','human_ranges_training':False})
 save(R/'stage_state.json',{'stage':'training_evaluation_complete','time':now()})
 print(json.dumps({'complete':True,'rows':rows}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  (R/'failure.log').write_text(traceback.format_exc())
  save(R/'failure.json',{'type':type(e).__name__,'time':now(),'frames':[{'file':f.filename,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(e.__traceback__)]})
  save(R/'stage_state.json',{'stage':'failed','type':type(e).__name__,'time':now()})
  raise SystemExit(1)
