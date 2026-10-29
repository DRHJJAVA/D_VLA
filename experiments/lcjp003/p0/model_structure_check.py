
import pathlib,sys,json,time,hashlib,traceback
root=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003')
sys.path.insert(0,str(root/'src'))
try:
 import torch
 from model import LCJP003,ModelConfig,NoValidObservationError
 torch.set_num_threads(4)
 torch.manual_seed(3003)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 device=torch.device('cuda')
 cfg=ModelConfig(semantic_dim=768,state_dim=18,action_dim=7)
 m=LCJP003(cfg).to(device).eval()
 B,T,N=1,16,64
 obj=torch.cat([torch.zeros(32),torch.ones(32)]).long().view(1,1,N).expand(B,T,-1).to(device)
 valid=torch.ones(B,T,N,dtype=torch.bool,device=device)
 valid[:,:,-1]=False
 xyz=torch.randn(B,T,N,3,device=device)*0.02
 xyz[:,:,32:,0]+=0.3
 norm=xyz.clone();norm[:,:,32:,0]-=0.3
 state=torch.randn(B,T,18,device=device)
 semantic=torch.randn(B,T,N,768,device=device)
 text=torch.randn(B,6,960,device=device)
 relation=torch.randn(B,T,N,16,device=device)*0.1
 relation[...,14:16]=1
 gid=torch.arange(N,device=device).view(1,N)//16
 history=torch.ones(B,T,dtype=torch.bool,device=device);history[:,:2]=False
 batch={'semantic':semantic,'xyz_base':xyz,'xyz_norm':norm,'relation':relation,
  'object_id':obj,'point_valid':valid,'state':state,'history_valid':history,
  'text_features':text,'text_valid':torch.ones(B,6,dtype=torch.bool,device=device),'group_id':gid}
 noise=torch.randn(B,2,16,7,device=device)
 start=time.monotonic()
 ctx=m.encode(batch);pred=m.decode(ctx,noise)
 loss=(pred-torch.randn_like(pred)).square().mean()
 loss.backward()
 grads={}
 for name,module in m.parameter_groups().items():
  g=[p.grad for p in module.parameters() if p.grad is not None]
  grads[name]={'finite':all(bool(torch.isfinite(x).all()) for x in g),'norm':float(sum((x.float().square().sum() for x in g),torch.zeros((),device=device)).sqrt()),'tensors_with_grad':len(g)}
 assert all(v['finite'] and v['norm']>0 for v in grads.values())
 alpha_grad=float(ctx['alpha'].grad.norm());beta_grad=float(ctx['beta'].grad.norm())
 assert alpha_grad>0 and beta_grad>0
 assert bool(torch.isfinite(pred).all())
 assert torch.allclose(ctx['scores'].sum(-1),torch.ones(B,device=device),atol=1e-6)
 assert float(ctx['scores'][~ctx['point_valid']].abs().max())==0
 probes={name:next(module.parameters()).detach().reshape(-1)[:4096].clone() for name,module in m.parameter_groups().items()}
 opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=0)
 opt.step()
 updates={name:float((next(module.parameters()).detach().reshape(-1)[:4096]-probes[name]).norm()) for name,module in m.parameter_groups().items()}
 assert all(v>0 for v in updates.values())
 opt.zero_grad(set_to_none=True)
 with torch.no_grad():
  reference=m(batch,noise,return_diagnostics=False)
  mutated={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
  for key in ['semantic','xyz_base','xyz_norm','relation']:
   mutated[key][:,:,-1]=float('nan')
   mutated[key][:,:2]=float('nan')
  mutated['state'][:,:2]=float('nan')
  invalid=m(mutated,noise,return_diagnostics=False)
  invalid_diff=float((reference['actions']-invalid['actions']).abs().max())
  assert invalid_diff==0
  by_g=torch.stack([m.decode(reference,noise[:,g]) for g in range(2)],1)
  repeat_diff=float((reference['actions']-by_g).abs().max())
  assert repeat_diff<1e-5
  empty={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
  empty['point_valid'][:,-1]=False
  empty_failed=False
  try:m.encode(empty)
  except NoValidObservationError:empty_failed=True
  assert empty_failed
 schema=m.schema()
 schema.update({'status':'synthetic_structure_check_passed','formal_p0_result':False,'pretrained_backbone_used':False,
  'self_test_input':{k:list(v.shape) for k,v in batch.items()},
  'context_tensors':{k:list(v.shape) for k,v in ctx.items() if isinstance(v,torch.Tensor)},
  'action_shape':list(pred.shape),'loss':float(loss.detach()),'gradient_norms':grads,'alpha_chain_grad_norm':alpha_grad,
  'beta_chain_grad_norm':beta_grad,'sampled_parameter_update_norms':updates,'invalid_input_invariance_maxdiff':invalid_diff,
  'multi_noise_vs_separate_maxdiff':repeat_diff,'empty_current_input_rejected':empty_failed,
  'elapsed_seconds':time.monotonic()-start,
  'source_sha256':hashlib.sha256((root/'src/model.py').read_bytes()).hexdigest(),
  'notes':['Random tensors validate structure and autograd only.','Formal P0 must use actual frozen features and real training observations.']})
 (root/'p0').mkdir(parents=True,exist_ok=True)
 (root/'p0/model_schema.json').write_text(json.dumps(schema,indent=2))
 print(json.dumps(schema))
except Exception as e:
 safe={'status':'model_structure_check_failed','exception_type':type(e).__name__,
       'frames':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]}
 (root/'p0').mkdir(parents=True,exist_ok=True)
 (root/'p0/model_failure.json').write_text(json.dumps(safe,indent=2))
 print(json.dumps(safe))
 raise SystemExit(1)
