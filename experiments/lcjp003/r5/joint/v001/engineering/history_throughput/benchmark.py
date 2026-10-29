import sys,pathlib,json,time,traceback,shutil,hashlib,os
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r5/joint/v001')
E=B/'engineering/history_throughput'
sys.path.insert(0,str(B/'src'));sys.path.insert(0,str(E))
import torch,numpy as np,copy
import train,io_runtime as io
from optimized_history import install
def save(name,data):
 p=E/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
def snap(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:snap(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [snap(v) for v in x]
 return x
def compare(a,b,atol=1e-6,rtol=1e-4):
 stats={'max_abs':0.,'mismatches':0,'tensors':0,'nonfinite':0}
 def walk(a,b):
  if torch.is_tensor(a):
   stats['tensors']+=1
   if not torch.is_tensor(b) or a.shape!=b.shape:stats['mismatches']+=1;return
   if a.dtype in (torch.bool,torch.int64,torch.int32,torch.uint8):
    if not torch.equal(a,b):stats['mismatches']+=1
   else:
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():stats['nonfinite']+=1
    if a.numel():stats['max_abs']=max(stats['max_abs'],float((a-b).abs().max()))
    if not torch.allclose(a,b,atol=atol,rtol=rtol):stats['mismatches']+=1
  elif isinstance(a,dict):
   for k in a:
    if k not in b:stats['mismatches']+=1
    else:walk(a[k],b[k])
  elif isinstance(a,list):
   if len(a)!=len(b):stats['mismatches']+=1
   else:
    for x,y in zip(a,b):walk(x,y)
 walk(a,b);stats['pass']=stats['mismatches']==0 and stats['nonfinite']==0;return stats
def main():
 io.setup()
 fixed=B/'checkpoints/step0500.pt'
 assert fixed.exists()
 shutil.copy2(fixed,E/'reference0500.pt')
 ck=torch.load(E/'reference0500.pt',map_location='cpu',weights_only=False)
 model=io.build();model.train()
 schedule=json.loads((B/'sampling_schedule.json').read_text())
 records=schedule[ck['step']+1]
 reference=None;reports=[]
 for label,batch in [('warmup',None),('baseline',None),('reuse4',4),('reuse8',8),('reuse16',16),('baseline_repeat',None),('reuse16_repeat',16),('baseline_final',None)]:
  save('state.json',{'status':'benchmark','mode':label,'reference_step':ck['step'],'pid':os.getpid(),'time':time.time()})
  model.load_state_dict(ck['model']);opt=io.optimizer(model);opt.load_state_dict(copy.deepcopy(ck['optimizer']));io.restore_rng(ck['rng'])
  undo=install(train,batch) if batch else None
  model._engineering_history_cache=None;model._engineering_computed=0;model._engineering_reused=0
  opt.zero_grad(set_to_none=True)
  data=[];outputs=[];losses=[];phases={'data':0.,'previous':0.,'objective':0.,'backward':0.}
  for rec in records[:2] if label=='warmup' else records:
   t=time.perf_counter();d=io.move(io.load_record(rec));torch.cuda.synchronize();phases['data']+=time.perf_counter()-t
   t=time.perf_counter();prev=train.previous_encoding(model,rec);torch.cuda.synchronize();phases['previous']+=time.perf_counter()-t
   noise=torch.randn(8,16,6,device='cuda')
   t=time.perf_counter();loss,metrics,e,pred=train.objective(model,d,noise,prev);torch.cuda.synchronize();phases['objective']+=time.perf_counter()-t
   t=time.perf_counter();(loss/8).backward();torch.cuda.synchronize();phases['backward']+=time.perf_counter()-t
   losses.append(float(loss.detach()))
   if label!='warmup':
    outputs.append(snap({'p':e['points'],'group':e['group_index'],'neighbor_valid':e['neighbor_valid'],'history_summaries':e.get('history_summaries'),'score':e['score'],'prediction':pred,'prefix':e['prefix']}))
   del loss,e,pred,prev,d
  grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  t=time.perf_counter();opt.step();torch.cuda.synchronize();phases['optimizer']=time.perf_counter()-t
  params=snap(model.state_dict());ost=snap(opt.state_dict())
  if undo:undo()
  report={'mode':label,'seconds':sum(phases.values()),'phases':phases,'losses':losses,'computed_history':model._engineering_computed,'reused_history':model._engineering_reused}
  if label=='baseline':reference={'outputs':outputs,'grads':grads,'params':params,'optimizer':ost,'losses':torch.tensor(losses),'rng':snap(io.rng_state())}
  elif label not in ('warmup',):
   report['comparison']={'outputs':compare(reference['outputs'],outputs,1e-6,1e-5),'gradients':compare(reference['grads'],grads),'parameters_after_adamw':compare(reference['params'],params,1e-7,1e-5),'optimizer_after_adamw':compare(reference['optimizer'],ost),'loss':compare(reference['losses'],torch.tensor(losses),1e-6,1e-5),'rng':compare(reference['rng'],snap(io.rng_state()),0,0)}
   report['qualified']=all(v['pass'] for v in report['comparison'].values())
  reports.append(report);save('results.json',{'reference_step':ck['step'],'reports':reports,'done':False})
 save('results.json',{'reference_step':ck['step'],'reports':reports,'done':True})
 save('state.json',{'status':'complete','time':time.time()})
try:main()
except Exception as e:
 save('failure.json',{'type':type(e).__name__,'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]})
