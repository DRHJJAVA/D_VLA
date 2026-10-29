from pathlib import Path
import sys,json,hashlib,traceback,random,os
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path('/mnt/nvme1/drh/D_VLA');BASE=ROOT/'experiments/contact_frames/cf015';PREV=BASE.parent/'cf014'
CFG=json.loads((BASE/'config.json').read_text())
sys.path.insert(0,str(PREV/'src'))
from model14 import ContactMotionNet
from common14 import dataset
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.22)
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
def lock(ps):return {str(p):sha(p) for p in ps}
def stage(s,**kw):save(BASE/'stage_state.json',dict(stage=s,time=now(),**kw))
def quant(a):
 a=np.asarray(a);a=a[np.isfinite(a)]
 return None if not len(a) else dict(n=len(a),min=float(a.min()),q25=float(np.quantile(a,.25)),median=float(np.median(a)),q75=float(np.quantile(a,.75)),max=float(a.max()),mean=float(a.mean()))
def corr(a,b):
 return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-9 and np.std(b)>1e-9 else None
def load(ep):
 with np.load(dataset(ep)) as n: d={k:n[k].copy() for k in n.files}
 with np.load(PREV/'features'/f'{ep}.npz') as n: d['motion_raw']=n['features'].copy();d['motion_valid']=n['valid'].copy()
 mn=json.loads((PREV/'motion_normalization.json').read_text())
 sn=json.loads((BASE.parent/'cf003/normalization.json').read_text())
 mo=(d['motion_raw']-mn['mean'])/mn['std'];mo[~d['motion_valid']]=0
 arr=[d['images'],d['points'],(d['states']-sn['mean'])/sn['std'],mo]
 x=[torch.tensor(a,dtype=torch.float32,device='cuda')[None] for a in arr]
 return d,x
def supervision(ep,d):
 with np.load(PREV/'features'/f'{ep}_supervision.npz') as n: t={k:n[k].copy() for k in n.files}
 original=d['near_target']*d['interaction_support']
 targetdiff=float(np.max(np.abs(d['target']-original)))
 # Explicit separated entries are expected zero by the original preparation.
 counts=[]
 for seed in CFG['seeds']:
  plan=json.loads((BASE.parent/'cf010/plans'/f'seed_{seed}.json').read_text())
  c=np.zeros(len(d['target']),np.int64)
  for item in plan:
   if item['episode']==ep:np.add.at(c,item['frames'],1)
  counts.append(c)
 counts=np.asarray(counts)
 pos=(t['target']==1)&(t['weight']>0);neg=(t['target']==0)&(t['weight']>0);unk=t['weight']==0
 out=dict(episode=ep,original_target_product_max_difference=targetdiff,groups={})
 for name,mask in [('positive',pos),('separated_negative',neg),('unknown',unk)]:
  out['groups'][name]=dict(frames=int(mask.sum()),near=quant(d['near_target'][mask]),support=quant(d['interaction_support'][mask]),
    old_soft_target=quant(d['target'][mask]),new_weight=quant(t['weight'][mask]),distance_m=quant(d['distance'][mask]),
    sampled_occurrences_per_seed=[int(c[mask].sum()) for c in counts],
    effective_main_mass_per_seed=[float((c[mask]*t['weight'][mask]).sum()) for c in counts],
    effective_aux_near_mass_per_seed=[float((c[mask]*d['near_weight'][mask]*.5).sum()) for c in counts])
 out['positive_soft_target_bands']=[]
 for l,h in [(0,.01),(.01,.1),(.1,.5),(.5,.8),(.8,1.000001)]:
  mask=pos&(original>=l)&(original<h)
  out['positive_soft_target_bands'].append(dict(low=l,high=h,frames=int(mask.sum()),
    mass=float(t['weight'][mask].sum()),exposure_mass=[float((c[mask]*t['weight'][mask]).sum()) for c in counts]))
 out['near_bands']=[]
 for l,h in [(0,.2),(.2,.5),(.5,.8),(.8,.95),(.95,1.000001)]:
  mask=(d['near_target']>=l)&(d['near_target']<h)
  out['near_bands'].append(dict(low=l,high=h,positive=int((pos&mask).sum()),negative=int((neg&mask).sum()),unknown=int((unk&mask).sum())))
 sp=BASE/'train'/f'{ep}_supervision.npz'
 arrays=dict(**t,old_target=d['target'],near=d['near_target'],support=d['interaction_support'],exposure=counts)
 if sp.exists():
  with np.load(sp) as prev:assert all(np.array_equal(prev[k],v,equal_nan=True) for k,v in arrays.items())
 else:np.savez_compressed(sp,**arrays)
 return out,t,counts
def gradient_check(net,x,d,t,count):
 net.train()
 out=net(*x);mainlog,dy,nearlog=out
 w=torch.tensor(t['weight'],device='cuda');y=torch.tensor(t['target'],device='cuda')
 exposure=torch.tensor(count/count.sum(),dtype=torch.float32,device='cuda')
 near_y=torch.tensor(d['near_target'],device='cuda');near_w=torch.tensor(d['near_weight'],device='cuda')
 loss_main=(F.binary_cross_entropy_with_logits(mainlog[0],y,reduction='none')*w*exposure).sum()
 loss_near=(F.binary_cross_entropy_with_logits(nearlog[0],near_y,reduction='none')*near_w*exposure).sum()*.5
 loss_dyn=(F.smooth_l1_loss(dy[0],torch.tensor(d['dy'],device='cuda'),reduction='none').mean(-1)*torch.tensor(d['dynvalid'],device='cuda')*exposure).sum()*.2
 pars=list(net.base.fuse.parameters())+list(net.base.temporal.parameters())
 grads=[]
 for loss in [loss_main,loss_near,loss_dyn]:
  g=torch.autograd.grad(loss,pars,retain_graph=True,allow_unused=True)
  grads.append(torch.cat([torch.zeros_like(p).flatten() if v is None else v.flatten() for p,v in zip(pars,g)]))
 def cosine(a,b):return float(F.cosine_similarity(a[None],b[None],eps=1e-12)[0])
 net.eval()
 return dict(main_loss=float(loss_main.detach()),near_weighted_loss=float(loss_near.detach()),dynamics_weighted_loss=float(loss_dyn.detach()),
   shared_gradient_norms=[float(g.norm()) for g in grads],cos_main_near=cosine(grads[0],grads[1]),cos_main_dyn=cosine(grads[0],grads[2]),
   interpretation='frozen last weights, plan-weighted gradient; not historical training gradient proof')
def phase(split):
 eps=CFG[split+'_episodes'];inputs=[BASE/'config.json',Path(__file__),PREV/'src/model14.py',PREV/'src/common14.py',PREV/'motion_normalization.json',BASE.parent/'cf003/normalization.json']
 for ep in eps:
  inputs.extend([dataset(ep),PREV/'features'/f'{ep}.npz',PREV/'evaluation'/ep/'predictions.npz'])
  if split=='train':inputs.append(PREV/'features'/f'{ep}_supervision.npz')
 for seed in CFG['seeds']:
  inputs.append(BASE.parent/'cf010/plans'/f'seed_{seed}.json')
  for arm in ['without_motion','with_motion']:inputs.append(PREV/'runs'/arm/f'seed_{seed}'/'last.pt')
 input_lock=lock(inputs);save(BASE/f'{split}/input_locked.json',input_lock)
 rows=[];superrows=[];grads=[];all_diffs=[]
 for ep in eps:
  stage('diagnosing_'+split,episode=ep)
  d,x=load(ep);t=None;counts=None
  if split=='train':
   sr,t,counts=supervision(ep,d);superrows.append(sr)
  a_probs=[];a_logits=[];a_near=[];controls=[]
  with np.load(PREV/'evaluation'/ep/'predictions.npz') as saved:
   for si,seed in enumerate(CFG['seeds']):
    net=ContactMotionNet(True).cuda()
    net.load_state_dict(torch.load(PREV/'runs/with_motion'/f'seed_{seed}'/'last.pt',map_location='cuda',weights_only=False)['model']);net.eval()
    prs=[];logs=[];nrs=[]
    with torch.no_grad():
     for variant in CFG['variants']:
      xx=list(x);mo=x[-1].clone()
      if variant=='branch_zero':mo.zero_()
      elif variant in CFG['groups']:mo[:,:,CFG['groups'][variant]]=0
      elif variant=='half_episode_shift':mo=torch.roll(mo,shifts=mo.shape[1]//2,dims=1)
      xx[-1]=mo
      log,dy,nlog=net(*xx)
      prs.append(log[0].sigmoid().cpu().numpy());logs.append(log[0].cpu().numpy());nrs.append(nlog[0].sigmoid().cpu().numpy())
    a_probs.append(prs);a_logits.append(logs);a_near.append(nrs)
    diff=float(np.max(np.abs(np.asarray(prs)[0]-saved['with_motion'][si])));assert diff<1e-6;all_diffs.append(diff)
    if split=='train':grads.append(dict(episode=ep,arm='with_motion',seed=seed,**gradient_check(net,x,d,t,counts[si])))
    del net
    net=ContactMotionNet(False).cuda()
    net.load_state_dict(torch.load(PREV/'runs/without_motion'/f'seed_{seed}'/'last.pt',map_location='cuda',weights_only=False)['model']);net.eval()
    with torch.no_grad():
     lp,_,ln=net(*x);cp=lp[0].sigmoid().cpu().numpy();cn=ln[0].sigmoid().cpu().numpy()
    diff=float(np.max(np.abs(cp-saved['without_motion'][si])));assert diff<1e-6;all_diffs.append(diff);controls.append(cp)
    if split=='train':grads.append(dict(episode=ep,arm='without_motion',seed=seed,**gradient_check(net,x,d,t,counts[si])))
    del net;torch.cuda.empty_cache()
  a_probs=np.asarray(a_probs);a_logits=np.asarray(a_logits);a_near=np.asarray(a_near)
  out=BASE/'probes'/f'{ep}.npz'
  assert not out.exists()
  np.savez_compressed(out,probabilities=a_probs,logits=a_logits,proximity=a_near,without_motion=np.asarray(controls),
    motion_raw=d['motion_raw'],motion_valid=d['motion_valid'],distance=d['distance'],quality=d['quality'],near_automatic=d.get('near_target',d['target']))
  row=dict(episode=ep,frames=len(d['target']),motion_valid=int(d['motion_valid'].sum()),
    main_near_probability_correlation=[corr(a_probs[i,0],a_near[i,0]) for i in range(3)],variants={})
  for vi,v in enumerate(CFG['variants']):
   row['variants'][v]=dict(mean_abs_probability_change=float(np.abs(a_probs[:,vi]-a_probs[:,0]).mean()),
    mean_abs_logit_change=float(np.abs(a_logits[:,vi]-a_logits[:,0]).mean()),
    high_frames=int((a_probs[:,vi].mean(0)>=.8).sum()))
  rows.append(row);del d,x;torch.cuda.empty_cache()
 summary=dict(time=now(),split=split,rows=rows,supervision=superrows,gradients=grads,original_prediction_max_difference=max(all_diffs),human_ranges_read=False)
 assert all(sha(p)==h for p,h in input_lock.items())
 save(BASE/split/'summary.json',summary)
 save(BASE/split/'output_locked.json',lock([BASE/'probes'/f'{e}.npz' for e in eps]+[BASE/split/'summary.json']))
 return summary
def main():
 stage('starting')
 phase('train')
 save(BASE/'pre_review_freeze.json',dict(time=now(),source=lock([Path(__file__),BASE/'config.json']),train=lock([BASE/'train/summary.json',BASE/'train/output_locked.json'])))
 phase('review')
 stage('frozen_diagnostics_complete')
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(BASE/'failure.json',dict(time=now(),type=type(e).__name__,message=str(e),stack=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(e.__traceback__)]));stage('diagnosis_failed');raise
