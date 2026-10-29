import os,sys,json,time,hashlib,traceback,random
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
R=Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003')
sys.path.insert(0,str(R/'src'))
from model import LCJP003,ModelConfig
from drifting import conditional_drifting_loss
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def write(name,d):
 p=R/'p0'/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));os.replace(t,p)
def progress(stage,**kwargs):write('network_check_state.json',dict(status='running',stage=stage,pid=os.getpid(),timestamp_bjt=now(),**kwargs))
KEYS=['semantic','xyz_base','xyz_norm','relation','object_id','point_valid','state','history_valid','group_id']
def getbatch(samples):
 data=[]
 for s in samples:
  p=R.parent.parent/s['path']
  fp=R/'p0/features'/(p.stem+'.npz')
  with np.load(p) as z: d={k:z[k].copy() for k in z.files}
  with np.load(fp) as z:
   for k in ['semantic','text_features','text_valid']:d[k]=z[k].copy()
   if 'sample_sha256' in z:assert str(z['sample_sha256'].item())==sha(p)
  data.append(d)
 batch={k:torch.from_numpy(np.stack([d[k] for d in data])).to('cuda') for k in KEYS}
 length=max(d['text_features'].shape[0] for d in data)
 text=np.zeros((len(data),length,960),np.float32);mask=np.zeros((len(data),length),bool)
 for i,d in enumerate(data):
  L=d['text_features'].shape[0];text[i,:L]=d['text_features'];mask[i,:L]=d['text_valid']
 batch['text_features']=torch.from_numpy(text).cuda();batch['text_valid']=torch.from_numpy(mask).cuda()
 actions=torch.from_numpy(np.stack([d['actions'] for d in data])).cuda()
 valid=torch.from_numpy(np.stack([d['action_valid'] for d in data])).cuda()
 return batch,actions,valid
def fin(x):
 if isinstance(x,torch.Tensor):return bool(torch.isfinite(x).all())
 if isinstance(x,dict):return all(fin(v) for v in x.values())
 if isinstance(x,(list,tuple)):return all(fin(v) for v in x)
 return True
def compare_context(c1,c2):
 return {k:float((c1[k]-c2[k]).abs().max()) for k in ['parts','alpha','beta','scores']}
def main():
 torch.set_num_threads(4)
 torch.manual_seed(20261026);np.random.seed(20261026);random.seed(20261026)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(20261026)
 torch.backends.cuda.matmul.allow_tf32=False
 start=time.monotonic()
 audit=json.loads((R/'p0/data_audit.json').read_text())
 fa=json.loads((R/'p0/features_audit.json').read_text())
 assert fa.get('status')=='completed', 'features_not_complete'
 samples=audit['samples']
 manifest={'created_bjt':now(),'experiment':'LCJP003 P0 real-data engineering check','data':{s['path']:sha(R.parent.parent/s['path']) for s in samples},'features':{p.name:sha(p) for p in (R/'p0/features').glob('*.npz')},'source':{p.name:sha(p) for p in (R/'src').glob('*.py')},'frozen_features_audit_sha256':sha(R/'p0/features_audit.json'),'seed':20261026,'dtype':'float32','G':8,'optimizer':{'type':'AdamW','lr':1e-4,'weight_decay':.01,'updates':1},'scope':'No training efficacy or success claim; only P0 one update'}
 write('network_input_lock.json',manifest)
 progress('initializing_network')
 model=LCJP003(ModelConfig()).cuda().eval()
 torch.cuda.reset_peak_memory_stats()
 rows=[]
 for i,s in enumerate(samples):
  progress('real_window_forward',index=i,total=len(samples),frame=s['current_frame'])
  b,y,m=getbatch([s])
  with torch.no_grad():
   context=model.encode(b)
   noise=torch.randn(1,8,16,7,device='cuda')
   pred=model.decode(context,noise)
   loss,diag=conditional_drifting_loss(pred,y,m)
   assert fin(context) and fin(pred) and fin(loss)
   assert torch.count_nonzero(context['scores'][~context['point_valid']])==0
   row={'episode_id':s['episode_id'],'frame':s['current_frame'],'history_valid':int(b['history_valid'].sum()),'actions_valid':int(m.sum()),'output_shape':list(pred.shape),'loss':float(loss),'finite':True,'parts':int(context['part_valid'].sum()),'alpha_group_sum_max_error':max(abs(float(context['alpha'][0][context['group_map'][0]==k].sum())-1.) for k in range(context['parts'].shape[1]) if bool(context['part_valid'][0,k])),'beta_sum':float(context['beta'].sum()),'drifting':diag}
   rows.append(row)
  del b,y,m,context,pred,loss
  write('forward_checks.json',{'status':'running' if i+1<len(samples) else 'completed','rows':rows})
 # Invalid history and point values: same masks, change only masked values.
 progress('invalid_value_invariance')
 s=next(s for s in samples if s['case']=='early_boundary')
 b,_,_=getbatch([s])
 # Set explicit few current points invalid in both test copies.
 b['point_valid'][:,:,-8:]=False
 b['object_id'][:,:,-8:]=-1
 b['group_id'][:,-8:]=-1
 bad={k:v.clone() for k,v in b.items()}
 for k in ['semantic','xyz_base','xyz_norm','relation']:
  bad[k][~b['point_valid']]=float('nan')
 bad['state'][~b['history_valid']]=float('nan')
 with torch.no_grad():
  c1=model.encode(b);c2=model.encode(bad)
  dif=compare_context(c1,c2)
  noise=torch.randn(1,8,16,7,device='cuda')
  dif['actions']=float((model.decode(c1,noise)-model.decode(c2,noise)).abs().max())
 assert max(dif.values())==0.
 del b,bad,c1,c2
 progress('real_data_action_loss_backward')
 chosen=[s for s in samples if s['case']=='middle']
 assert len(chosen)==2
 b,y,m=getbatch(chosen)
 model.train()
 groups=model.parameter_groups()
 before={name:[p.detach().cpu().clone() for p in mod.parameters()] for name,mod in groups.items()}
 opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
 opt.zero_grad(set_to_none=True)
 context=model.encode(b)
 context['alpha'].retain_grad();context['beta'].retain_grad()
 noise=torch.randn(2,8,16,7,device='cuda')
 pred=model.decode(context,noise)
 loss,diag=conditional_drifting_loss(pred,y,m)
 assert fin(loss) and fin(pred)
 loss.backward()
 grads={}
 for name,mod in groups.items():
  params=list(mod.parameters());active=[p.grad for p in params if p.grad is not None]
  norm=float(torch.sqrt(sum(g.detach().double().square().sum() for g in active)))
  grads[name]={'finite':all(fin(g) for g in active),'gradient_l2':norm,'parameters_with_gradient':sum(p.numel() for p in params if p.grad is not None),'nonzero_gradient_values':sum(int(torch.count_nonzero(g)) for g in active)}
  assert grads[name]['finite'] and norm>0, 'module_gradient_gate'
 chain={}
 for k in ['alpha','beta']:
  g=context[k].grad
  chain[k]={'finite':g is not None and fin(g),'gradient_l2':None if g is None else float(g.norm()),'nonzero_gradient_values':0 if g is None else int(torch.count_nonzero(g))}
  assert chain[k]['finite'] and chain[k]['gradient_l2']>0,'score_chain_gradient_gate'
 progress('one_optimizer_update')
 opt.step()
 assert all(fin(p) for p in model.parameters()), 'nonfinite_parameter_after_update'
 for name,mod in groups.items():
  diffs=[(p.detach().cpu()-q) for p,q in zip(mod.parameters(),before[name])]
  grads[name]['parameter_delta_max']=max(float(d.abs().max()) for d in diffs)
  grads[name]['changed_values']=sum(int(torch.count_nonzero(d)) for d in diffs)
  assert grads[name]['parameter_delta_max']>0,'parameter_update_gate'
 # Save separate P0-only disposable checkpoint, not a trained policy.
 cp=R/'p0/one_update_engineering_only.pt'
 torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'updates':1,'scope':'P0 only; must not enter formal training','seed':20261026,'source':manifest['source']},cp)
 result={'status':'completed','timestamp_bjt':now(),'scope':manifest['scope'],'windows_checked':len(rows),'full_windows':6,'boundary_windows':4,'training_demo_count':2,'G':8,'model_schema':model.schema() if hasattr(model,'schema') else {'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)},'invalid_nan_invariance':dif,'optimizer_updates':1,'one_update_loss':float(loss.detach()),'modules':grads,'score_chain':chain,'drifting':diag,'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'seconds':time.monotonic()-start,'checkpoint_sha256':sha(cp),'checkpoint_scope':'P0 engineering only','input_lock_sha256':sha(R/'p0/network_input_lock.json')}
 write('network_check_result.json',result)
 write('network_check_state.json',{'status':'completed','timestamp_bjt':now(),'result':'network_check_result.json','pid':os.getpid()})
 print(json.dumps({'status':'completed','windows':len(rows),'updates':1}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  f={'status':'failed','timestamp_bjt':now(),'exception_type':type(e).__name__,'frames':[{'file':x.filename,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(e.__traceback__)]}
  write('network_check_failure.json',f);write('network_check_state.json',f);print(json.dumps(f));sys.exit(1)
