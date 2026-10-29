from common import *
import model as modelmod
from model import LCJP003,ModelConfig
from drifting import conditional_drifting_loss
from geometry_cache import GeometryCache
import math,shutil,contextlib,fcntl
def sync():torch.cuda.synchronize()
def snap(model,opt,step,sampler):
 return {'step':step,'model':model.state_dict(),'optimizer':opt.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'sampler':sampler.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'config_sha256':sha(P/'config.json'),'source_lock_sha256':sha(P/'source_lock.json')}
def save(model,opt,step,sampler,milestone=False):
 t=time.monotonic();tmp=P/'last.tmp.pt';torch.save(snap(model,opt,step,sampler),tmp);tmp.replace(P/'last.pt')
 if milestone:
  out=P/('step%04d.pt'%step)
  if not out.exists():os.link(P/'last.pt',out)
 return time.monotonic()-t
def contextstats(c):
 a=c['alpha'].detach().float();b=c['beta'].detach().float()
 av=a[c['point_valid']];bv=b[c['part_valid']]
 return {'alpha_mean':float(av.mean()),'alpha_std':float(av.std()),'alpha_max':float(av.max()),'beta_std':float(bv.std()),'beta_max':float(bv.max()),'beta_entropy':float(-(b*b.clamp_min(1e-30).log()).sum(-1).mean())}
def work(model,ds,pairs,noise,micro,backward=True):
 tm={'data_h2d_s':0.,'encode_s':0.,'action_forward_loss_s':0.,'backward_s':0.};lossval=0.;summary=None
 for start in range(0,len(pairs),micro):
  take=pairs[start:start+micro];scale=len(take)/len(pairs)
  t=time.monotonic();b,y,m=batch(ds,take);sync();tm['data_h2d_s']+=time.monotonic()-t
  t=time.monotonic();c=model.encode(b);sync();tm['encode_s']+=time.monotonic()-t
  t=time.monotonic();pred=model.decode(c,noise[start:start+len(take)])
  loss,_=conditional_drifting_loss(pred,y,m);loss=loss*scale;sync();tm['action_forward_loss_s']+=time.monotonic()-t
  assert torch.isfinite(loss)
  if backward:
   t=time.monotonic();loss.backward();sync();tm['backward_s']+=time.monotonic()-t
  lossval+=float(loss.detach());summary=contextstats(c)
 return lossval,tm,summary
def gradients(model):
 return {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
def compare_grads(old,model):
 maxerr=0.;sqerr=0.;sqref=0.
 for n,p in model.named_parameters():
  assert p.grad is not None and torch.isfinite(p.grad).all()
  v=p.grad.detach().cpu();a=old[n];delta=(v-a).double()
  maxerr=max(maxerr,float(delta.abs().max()));sqerr+=float((delta*delta).sum());sqref+=float(a.double().square().sum())
 return {'max_abs':maxerr,'relative_l2':math.sqrt(sqerr/max(sqref,1e-30))}
def evaluate(model,ds,step,pairs):
 start=time.monotonic();rows=[];model.eval()
 std=torch.tensor(ds.action_std,device='cuda')
 with torch.no_grad():
  for i,(eid,frame) in enumerate(pairs):
   state('evaluate',step=step,index=i,total=len(pairs))
   b,y,m=batch(ds,[(eid,frame)])
   rng=torch.Generator(device='cuda').manual_seed(CFG['seed']+10000+i)
   noise=torch.randn(1,8,16,7,device='cuda',generator=rng)
   c=model.encode(b);pred=model.decode(c,noise)
   err=(pred-y[:,None]).abs();mask=m[:,None,:,None].expand_as(err)
   axis=(err*mask).sum((0,1,2))/mask.sum((0,1,2)).clamp_min(1)
   raw=axis*std
   stats=contextstats(c)
   rows.append({'episode':eid,'frame':frame,'normalized_mae':float(axis.mean()),'normalized_axis_mae':axis.tolist(),'raw_axis_mae':raw.tolist(),'translation_mae':float(raw[:3].mean()),'rotation_mae':float(raw[3:6].mean()),'gripper_mae':float(raw[6]),**stats})
   out=P/'predictions'/('step%04d'%step);out.mkdir(parents=True,exist_ok=True)
   np.savez(out/('%03d.npz'%i),prediction=pred.cpu().numpy(),target=y.cpu().numpy(),valid=m.cpu().numpy(),alpha=c['alpha'].cpu().numpy(),beta=c['beta'].cpu().numpy(),group_id=b['group_id'].cpu().numpy(),frame=np.array(frame),episode=np.array(eid))
 model.train()
 groups={}
 for split,ids in [('train',CFG['train']),('dev',CFG['dev'])]:
  rr=[x for x in rows if x['episode'] in ids]
  groups[split]={k:float(np.mean([x[k] for x in rr])) for k in ['normalized_mae','translation_mae','rotation_mae','gripper_mae','alpha_std','beta_std','beta_entropy']}
 if step>0:
  base=json.loads((P/'eval0000.json').read_text())
  for split in groups:groups[split]['mae_reduction_vs_init']=1-groups[split]['normalized_mae']/base['groups'][split]['normalized_mae']
 result={'step':step,'time':now(),'seconds':time.monotonic()-start,'groups':groups,'rows':rows,'noise':'fixed per evaluation condition, G8','closed_loop_evaluated':False}
 write('eval%04d.json'%step,result)
 if step==250:
  doc=ROOT/'docs'/'LCJP003_状态条件三维部位评分与DriftingVLA设计方案.md'
  with doc.open('a',encoding='utf-8') as f:
   f.write('\n\n### '+now()+' P1 第250次更新阶段结果\n\n'+json.dumps(groups,ensure_ascii=False,indent=2)+'\n\n固定噪声和同一80个评估窗口；完整逐样本结果在p1/v001/eval0250.json。此处为动作拟合与评分数值阶段检查，闭环成功率待独立执行。训练继续，未读取留出。\n')
  write('milestone250.json',result)
 return result
def main():
 cpu_setup();seed(CFG['seed']);sampler=np.random.default_rng(CFG['seed'])
 lock=(P/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 inputs=json.loads((P/'inputs.json').read_text());assert inputs['status']=='complete'
 assert inputs['config_sha256']==sha(P/'config.json')
 for n,h in CFG['p0_source'].items():assert sha(P/'source'/n)==h
 for n,h in inputs['frames'].items():assert sha(P/n)==h
 ds=Dataset(cached=True);assert ds.normalization==inputs['normalization']
 evalpairs=[(eid,int(f)) for eid in CFG['train']+CFG['dev'] for f in np.linspace(0,len(ds.actions[eid])-1,8,dtype=int)]
 write('evaluation_lock.json',{'pairs':evalpairs,'noise_seed_base':CFG['seed']+10000,'closed_loop':{'episodes':[CFG['train'][i%8] for i in range(20)],'noise_seeds':[CFG['seed']+i for i in range(20)]}})
 model=LCJP003(ModelConfig()).cuda();model.train()
 opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 geo=GeometryCache(modelmod);startstep=0
 if (P/'last.pt').exists():
  cp=torch.load(P/'last.pt',map_location='cpu',weights_only=False)
  assert cp['config_sha256']==sha(P/'config.json') and cp['source_lock_sha256']==sha(P/'source_lock.json')
  model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer'])
  random.setstate(cp['python_rng']);np.random.set_state(cp['numpy_rng']);sampler.bit_generator.state=cp['sampler'];torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state_all(cp['cuda_rng'])
  startstep=cp['step'];del cp;geo.apply()
  micro=json.loads((P/'engineering.json').read_text())['selected_microbatch']
 else:
  pairs=[(CFG['train'][0],len(ds.actions[CFG['train'][0]])//2)]
  noise=torch.randn(1,8,16,7,device='cuda')
  state('geometry_equivalence_baseline')
  loss0,t0,_=work(model,ds,pairs,noise,1);base=gradients(model);opt.zero_grad(set_to_none=True)
  geo.apply()
  state('geometry_cache_warmup')
  with torch.no_grad():work(model,ds,pairs,noise,1,False)
  state('geometry_equivalence_cached')
  loss1,t1,_=work(model,ds,pairs,noise,1);cmp=compare_grads(base,model)
  assert loss0==loss1 and cmp['max_abs']==0
  del base;opt.zero_grad(set_to_none=True)
  # Same eight conditions and exact noise; no optimizer update during engineering measurement.
  pairs=[(eid,len(ds.actions[eid])//2) for eid in CFG['train']]
  noise=torch.randn(8,8,16,7,device='cuda');bench=[];ref=None;ref_loss=None
  state('batch_benchmark_geometry_warmup')
  with torch.no_grad():work(model,ds,pairs,noise,2,False)
  for mb in CFG['microbatch_candidates']:
   state('batch_benchmark',microbatch=mb)
   try:
    opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();loss,tm,_=work(model,ds,pairs,noise,mb)
    measured_seconds=time.monotonic()-start
    delta={'max_abs':0.,'relative_l2':0.}
    if ref is None:ref=gradients(model);ref_loss=loss
    else:
     delta=compare_grads(ref,model)
     assert delta['relative_l2']<1e-4 and abs(loss-ref_loss)<1e-5
    bench.append({'microbatch':mb,'seconds':measured_seconds,'timing':tm,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'loss':loss,'gradient_comparison':delta})
   except torch.OutOfMemoryError:
    opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();bench.append({'microbatch':mb,'oom':True})
  del ref;opt.zero_grad(set_to_none=True)
  assert any('seconds' in b for b in bench)
  micro=min([b for b in bench if 'seconds' in b],key=lambda b:b['seconds'])['microbatch']
  write('engineering.json',{'time':now(),'geometry_equivalence':cmp,'geometry_loss_equal':loss0==loss1,'uncached':t0,'cached':t1,'batch_benchmarks':bench,'selected_microbatch':micro,'effective_batch':8,'precision':'FP32','updates_during_benchmark':0})
  # Restore fresh initialization and RNG, not benchmark activations or P0 weights.
  del model,opt;torch.cuda.empty_cache();seed(CFG['seed']);sampler=np.random.default_rng(CFG['seed'])
  model=LCJP003(ModelConfig()).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
  save(model,opt,0,sampler,True)
 if not (P/'eval0000.json').exists():evaluate(model,ds,0,evalpairs)
 elapsed_rows=[];train_start=time.monotonic()
 for step in range(startstep+1,CFG['steps']+1):
  state('training',step=step,completed_updates=step-1,total=3000,microbatch=micro)
  pairs=[]
  for j in range(8):
   eid=CFG['train'][int(sampler.integers(8))];pairs.append((eid,int(sampler.integers(len(ds.actions[eid])))))
  opt.zero_grad(set_to_none=True);noise=torch.randn(8,8,16,7,device='cuda')
  start=time.monotonic();loss,tm,stats=work(model,ds,pairs,noise,micro)
  t=time.monotonic()
  norms=torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters() if p.grad is not None])
  assert torch.isfinite(norms).all()
  grad={}
  for key in ['point_score','group_score']:
   ps=[p.grad for n,p in model.named_parameters() if n.startswith(key+'.') and p.grad is not None]
   grad[key]=float(torch.stack([g.square().sum() for g in ps]).sum().sqrt())
  opt.step();sync();tm['optimizer_check_s']=time.monotonic()-t
  duration=time.monotonic()-start
  row={'step':step,'time':now(),'loss':loss,**tm,**stats,'score_gradients':grad,'seconds':duration,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'geometry_cache_hits':geo.hits,'geometry_cache_misses':geo.misses,'samples':pairs}
  with (P/'runtime_timings.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  elapsed_rows.append(row)
  if step%10==0 or step in [20,50] or step==3000:save(model,opt,step,sampler,step in [20,50] or step%250==0)
  if step in [20,50]:
   stable=elapsed_rows[-min(len(elapsed_rows),30 if step==50 else 15):];avg=float(np.mean([x['seconds'] for x in stable]));baseline=json.loads((P/'eval0000.json').read_text())['seconds']
   eta={'step':step,'time':now(),'mean_update_s':avg,'p50_update_s':float(np.median([x['seconds'] for x in stable])),'phase_mean_s':{k:float(np.mean([x[k] for x in stable])) for k in tm},'eta_to_250_s':max(0,250-step)*avg+baseline,'eta_remaining_training_and_evaluation_s':(3000-step)*avg+math.ceil((3000-step)/250)*baseline,'closed_loop_time_included':False,'checkpoint_preserved':True,'shared_gpu_load_variable':True}
   write('eta%04d.json'%step,eta)
  if step%250==0:evaluate(model,ds,step,evalpairs)
  state('training',step=step,completed_updates=step,total=3000,last_update_s=duration)
 state('training_complete_pending_closed_loop',completed_updates=3000)
if __name__=='__main__':
 try:main()
 except Exception as e:failure(e);raise SystemExit(1)
