from pathlib import Path
import sys,json,hashlib,random,collections,traceback
import numpy as np
import torch
import torch.nn.functional as F
B=Path(__file__).resolve().parents[1];R=B.parents[2];O=R/'experiments/region_pose/iter_029'
sys.path.insert(0,str(B/'src'))
import train as trainer
from model import RegionPoseNet
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def main():
 for fn in ['source_locked.json','input_locked.json']:
  for k,v in json.loads((B/fn).read_text()).items():assert sha(R/k)==v,k
 cfg=json.loads((B/'config.json').read_text());m=json.loads((B/'manifest.json').read_text());rows=m['rows']
 tr={r['episode'] for r in rows if r['split']=='train'};va={r['episode'] for r in rows if r['split']=='val'}
 assert len(tr)==10 and len(va)==3 and tr.isdisjoint(va) and len(rows)==52
 checks=[]
 for r in rows:
  d=np.load(B/'samples'/f"{r['key']}.npz");known=d['known'];pos=known&d['target'];neg=known&~d['target']
  assert r['query_frame']<r['closed_command_start_frame']
  assert len(np.unique(d['reference_source_index'][pos]))>=3 and len(np.unique(d['reference_source_index'][neg]))>=16
  assert np.all(d['confidence'][~known]==0) and np.isfinite(d['dino_feat']).all()
  assert sha(B/'samples'/f"{r['key']}.npz")==sha(O/'samples'/f"{r['key']}.npz")
  checks.append({'key':r['key'],'positive_sources':int(len(np.unique(d['reference_source_index'][pos]))),'negative_sources':int(len(np.unique(d['reference_source_index'][neg]))),'same_frozen_sample':True})
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
 random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
 net=RegionPoseNet().cuda()
 statehash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
 il=json.loads((B/'initialization/locked.json').read_text());assert statehash==il['state_sha256'];assert sha(B/'initialization/initial.pt')==il['checkpoint_sha256']
 del net;torch.cuda.empty_cache()
 for run in cfg['runs']:
  name=run['group']+'_'+run['variant'];plan=json.loads((B/'plans'/f'{name}.json').read_text());events=sorted({r['event'] for r in rows if r['split']=='train'});rng=random.Random(cfg['seed']);expected=[]
  for ep in range(1,25):
   order=events.copy();rng.shuffle(order)
   for ev in order:
    off=[15,10,5,1][(ep-1+events.index(ev))%4];r=next(r for r in rows if r['event']==ev and r['offset']==off)
    expected.append({'epoch':ep,'event':ev,'offset':off,'key':r['key']})
  assert plan==expected and len(plan)==240
 diagnostics=[];max_bce_rank_error=0.
 for run in cfg['runs']:
  name=run['group']+'_'+run['variant']
  for snap in ['best','last']:
   out=B/'preflight'/name/snap;out.mkdir(parents=True,exist_ok=False)
   for r in [r for r in rows if r['split']=='train']:
    key=r['key'];a=np.load(B/'samples'/f'{key}.npz');p=np.load(O/'evaluation'/name/snap/f'{key}.npz')
    d={k:torch.as_tensor(a[k],device='cuda',dtype=torch.float32) for k in ['target','confidence']};d['known']=torch.as_tensor(a['known'],device='cuda')
    old=torch.tensor(p['logit'],device='cuda',requires_grad=True);new=old.detach().clone().requires_grad_(True)
    lo=trainer.loss(old,d);ln=trainer.training_loss(new,d);lo.backward();ln.backward()
    assert torch.isfinite(ln) and torch.isfinite(new.grad).all()
    assert torch.count_nonzero(new.grad[~d['known']])==0 and torch.count_nonzero(old.grad[~d['known']])==0
    pos=d['known']&(d['target']>.5);neg=d['known']&~pos
    ix=torch.nonzero(neg).flatten();selected=ix[torch.argsort(new[ix].detach(),descending=True,stable=True)[:min(64,len(ix))]]
    expected_np=np.flatnonzero(a['known']&~a['target']);expected_np=expected_np[np.argsort(-p['logit'][expected_np],kind='stable')[:64]]
    assert np.array_equal(selected.cpu().numpy(),expected_np)
    assert torch.equal(selected,ix[torch.argsort(new[ix].detach(),descending=True,stable=True)[:min(64,len(ix))]])
    assert torch.all(neg[selected])
    ip=torch.nonzero(pos).flatten();ip=ip[torch.linspace(0,len(ip)-1,min(64,len(ip)),device='cuda').long()]
    b=F.binary_cross_entropy_with_logits(new,d['target'],reduction='none');w=d['confidence']
    bce=.5*((b[pos]*w[pos]).sum()/w[pos].sum().clamp_min(1e-8)+(b[neg]*w[neg]).sum()/w[neg].sum().clamp_min(1e-8))
    rank=F.softplus(new[selected][None,:]-new[ip][:,None]).mean()
    er=abs(float((ln-bce-.1*rank).detach()));max_bce_rank_error=max(max_bce_rank_error,er);assert er<2e-6
    inv=neg&(new>=new[pos].min())
    rec={'run':name,'snapshot':snap,'key':key,'episode':r['episode'],'offset':r['offset'],'original_loss':lo.item(),'candidate_loss':ln.item(),'positive_gradient_l1_original':old.grad[pos].abs().sum().item(),'positive_gradient_l1_candidate':new.grad[pos].abs().sum().item(),'negative_gradient_l1_original':old.grad[neg].abs().sum().item(),'negative_gradient_l1_candidate':new.grad[neg].abs().sum().item(),'inverting_negatives':int(inv.sum()),'selected_inverting_negatives':int(inv[selected].sum()),'unknown_gradient_zero':True}
    diagnostics.append(rec)
    np.savez_compressed(out/f'{key}.npz',original_gradient=old.grad.cpu().numpy(),candidate_gradient=new.grad.cpu().numpy(),negative_selected=selected.cpu().numpy(),logit=p['logit'],known=a['known'],target=a['target'])
 # deterministic ties; very high unknown never selected or given gradient
 a=torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,1e6,1e6],device='cuda',requires_grad=True)
 d={'known':torch.tensor([True]*8+[False]*2,device='cuda'),'target':torch.tensor([1.,1.]+[0.]*8,device='cuda'),'confidence':torch.tensor([1.]*8+[0.]*2,device='cuda')}
 trainer.training_loss(a,d).backward();assert torch.count_nonzero(a.grad[8:])==0
 idx=torch.arange(2,8,device='cuda');assert torch.equal(idx,idx[torch.argsort(a[idx].detach(),descending=True,stable=True)])
 dump(B/'audits/preflight_loss_gradient.json',{'observations':diagnostics,'max_bce_rank_reconstruction_difference':max_bce_rank_error,'tie_and_extreme_unknown_check':True,'unknown_gradient_zero':True,'selection_deterministic':True,'no_model_inference_for_static_probe':True})
 dump(B/'audits/preflight.json',{'samples':checks,'train_events':10,'val_events':3,'train_queries':40,'val_queries':12,'same_original_initialization':statehash,'plans_exact':True,'updates_per_run':240,'static_train_snapshots':len(diagnostics),'passed':True})
 for fn in ['source_locked.json','input_locked.json']:
  for k,v in json.loads((B/fn).read_text()).items():assert sha(R/k)==v,k
 dump(B/'preflight_locked.json',{str(p.relative_to(R)):sha(p) for p in list((B/'preflight').rglob('*.npz'))+[B/'audits/preflight.json',B/'audits/preflight_loss_gradient.json']})
 print(json.dumps({'preflight_passed':True,'static_snapshots':len(diagnostics),'init':statehash}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  (B/'logs/preflight_traceback_remote_only.txt').write_text(traceback.format_exc());print(json.dumps({'error_type':type(e).__name__,'message':str(e)[:250]}));raise SystemExit(1)
