import sys,pathlib,json,time,hashlib,traceback,random,datetime
import numpy as np,torch
RUN=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r2/v2/v001')
OUT=RUN/'acceptance'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(RUN/'src'))
from common import build_expert
from network import R2Front,install_lora
from objective import to_device,infer
from train import fixed_windows,check_sources
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)
def point_resample(x,valid,index,seed):
 y=x.clone();v=valid.clone();rng=torch.Generator(device='cpu').manual_seed(seed)
 for patch in range(x.shape[1]):
  q=torch.where(valid[index,patch])[0].cpu()
  if q.numel()<2:continue
  choice=q[torch.randint(q.numel(),(x.shape[2],),generator=rng)].to(x.device)
  y[index,patch]=x[index,patch,choice]
  v[index,patch]=True
 return y,v
def rank_corr(a,b):
 if len(a)<4:return None
 aa=np.argsort(np.argsort(a));bb=np.argsort(np.argsort(b))
 return float(np.corrcoef(aa,bb)[0,1])
def bootstrap_episode(rows,field,seed):
 groups={}
 for a in rows:groups.setdefault(a['episode'],[]).append(a[field])
 z=np.array([np.mean(x) for x in groups.values()],dtype=float)
 rng=np.random.default_rng(seed);boots=np.array([np.mean(rng.choice(z,size=len(z),replace=True)) for _ in range(5000)])
 return {'episodes':len(z),'mean_delta':float(z.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist(),'positive_lower':bool(np.quantile(boots,.025)>0)}
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 check_sources()
 c=torch.load(RUN/'checkpoints/last.pt',map_location='cpu',weights_only=False)
 assert c['completed_updates']==3000
 assert digest(RUN/'checkpoints/last.pt')==json.loads((RUN/'complete.json').read_text())['checkpoint_sha256']
 front=R2Front().cuda().eval();front.load_state_dict(c['front'],strict=True)
 expert=build_expert('cuda');install_lora(expert);res=expert.load_state_dict(c['lora'],strict=False);assert not res.unexpected_keys
 expert.eval();rows=[];start=time.time()
 target={ (v['episode'],v['frame']):v for v in json.loads((RUN/'eval3000.json').read_text())['records'] if v['split']=='dev'}
 for k,(episode,t,ns) in enumerate(x for x in fixed_windows() if x[0]['split']=='dev'):
  bundle=torch.load(RUN/'cache'/episode['id']/(str(t)+'.pt'),map_location='cpu',weights_only=False);d=to_device(bundle)
  noise=torch.randn(1,16,6,generator=torch.Generator().manual_seed(ns)).cuda()
  with torch.inference_mode():
   motion,grip,normal=infer(front,expert,d,noise=noise)
   normal_error=float((motion-d['target']).abs().mean())
   assert abs(normal_error-target[(episode['id'],t)]['normalized_motion_mae'])<1e-6
   selection=normal['pi'].detach().cpu().numpy();top=int(np.argmax(selection[:-1]))
   mask=d['valid'].any(-1).unsqueeze(-1)
   original_raw=[None]
   def intervention(mode):
    def hook(module,inputs,raw):
     original_raw[0]=raw.detach()
     out=raw.clone()
     if mode=='uniform':
      weight=mask.to(out.dtype)
      mean=(out*weight).sum(1,keepdim=True)/weight.sum(1,keepdim=True).clamp_min(1)
      return torch.where(mask,mean.expand_as(out),out)
     gen=random.Random(ns+971)
     for i in range(out.shape[0]):
      ids=torch.where(mask[i,:,0])[0].tolist()
      perm=ids[:];gen.shuffle(perm)
      if ids:out[i,ids]=raw[i,perm]
     return out
    return hook
   variants={}
   for mode in ['uniform','shuffled']:
    h=front.score.register_forward_hook(intervention(mode))
    try:m,g,o=infer(front,expert,d,noise=noise)
    finally:h.remove()
    variants[mode]={'error':float((m-d['target']).abs().mean()),'motion_delta':float((m-motion).abs().mean()),'grip_flips':int((g.argmax(-1)!=grip.argmax(-1)).sum())}
   check_m,check_g,_=infer(front,expert,d,noise=noise)
   assert torch.equal(check_m,motion) and torch.equal(check_g,grip)
   correlations=[]
   scores=normal['scores'][top].detach().cpu().numpy()
   active=normal['patch_valid'][top].detach().cpu().numpy().astype(bool)
   for j in [0,1]:
    xx,vv=point_resample(d['x'],d['valid'],top,ns+50000+j)
    variant_d=dict(d);variant_d['x']=xx;variant_d['valid']=vv
    altered=front.encode(*[variant_d[z] for z in ['x','valid','h','words','priors']])
    cc=rank_corr(scores[active],altered['scores'][top].detach().cpu().numpy()[active])
    if cc is not None:correlations.append(cc)
  row={'episode':episode['id'],'frame':t,'noise_seed':ns,'candidate_count':len(selection)-1,'normal_error':normal_error,'null_prob':float(selection[-1]),'top_id':bundle['candidate_ids'][top],'top_prob':float(selection[top]),'normalized_entropy':float(-(selection*np.log(selection+1e-12)).sum()/np.log(len(selection))),'top_valid_patches':int(active.sum()),'top_score_std':float(scores[active].std()) if active.any() else 0,'resampling_rank_correlations':correlations}
  for name,x in variants.items():
   row[name+'_error']=x['error'];row[name+'_loss_delta']=x['error']-normal_error;row[name+'_motion_delta']=x['motion_delta'];row[name+'_grip_flips']=x['grip_flips']
  rows.append(row)
  if k%5==4:save('progress.json',{'time':datetime.datetime.now().isoformat(),'completed':k+1,'total':40,'seconds':time.time()-start})
 summary={'time':datetime.datetime.now().isoformat(),'checkpoint_sha256':digest(RUN/'checkpoints/last.pt'),'source_lock_sha256':digest(RUN/'source_lock.json'),'config_sha256':digest(RUN/'config.json'),'cases':len(rows),'episodes':len(set(x['episode'] for x in rows)),'baseline_mae':float(np.mean([x['normal_error'] for x in rows])),'top_probability':{'mean':float(np.mean([x['top_prob'] for x in rows])),'median':float(np.median([x['top_prob'] for x in rows]))},'null_probability_mean':float(np.mean([x['null_prob'] for x in rows])),'normalized_selection_entropy_mean':float(np.mean([x['normalized_entropy'] for x in rows])),'top_score_std_median':float(np.median([x['top_score_std'] for x in rows])),'resampling':{},'ablation':{},'elapsed_seconds':time.time()-start}
 cor=[c for x in rows for c in x['resampling_rank_correlations']]
 summary['resampling']={'cases':len(cor),'median_rank_correlation':float(np.median(cor)) if cor else None,'p10':float(np.quantile(cor,.1)) if cor else None,'passed_median_at_least_0_8':bool(cor and np.median(cor)>=.8),'method':'within-patch bootstrap from existing valid current RGB-D points; original candidate and patch indices fixed'}
 for mode in ['uniform','shuffled']:
  abl=np.mean([x[mode+'_error'] for x in rows]);base=summary['baseline_mae'];test=bootstrap_episode(rows,mode+'_loss_delta',20261028+len(mode))
  summary['ablation'][mode]={'mae':float(abl),'normal_improvement_percent':float((abl-base)/abl*100),'mean_action_delta':float(np.mean([x[mode+'_motion_delta'] for x in rows])),'grip_flips':sum(x[mode+'_grip_flips'] for x in rows),'episode_bootstrap':test,'passed_5pct_and_ci':bool((abl-base)/abl>=.05 and test['positive_lower'])}
 summary['role_accuracy']={'status':'requires independent object identity annotation','not_inferred_from_model_priors':True}
 summary['score_contribution_proven']=bool(all(v['passed_5pct_and_ci'] for v in summary['ablation'].values()))
 save('metrics.json',summary);save('per_window.json',rows)
if __name__=='__main__':
 try:main()
 except Exception as e:
  save('failure.json',{'time':datetime.datetime.now().isoformat(),'type':type(e).__name__,'message_summary':str(e)[:120],'frames':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)[-4:]]})
