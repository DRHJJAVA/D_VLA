import sys,pathlib,random,importlib.util,gc
import torch,numpy as np
from data import ROOT,RUN,R2,CFG
def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def setup():
 torch.set_num_threads(4);torch.set_num_interop_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False
def build(device='cuda'):
 spec=importlib.util.spec_from_file_location('r4_official_factory',R2/'src/common.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 official=module.build_expert('cpu');expert=official.lm_expert
 del official;gc.collect()
 from network import R4Network
 return R4Network(expert).float().to(device)
def move(x,device='cuda'):
 if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
 return x.to(device,non_blocking=True) if torch.is_tensor(x) else x
def rng_state():
 return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state_all(s['cuda'])
def optimizer(model):
 c=CFG['optimizer'];return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr'],weight_decay=c['weight_decay'],betas=tuple(c['betas']),eps=c['eps'])
def checkpoint(model,opt,step):
 data={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'rng':rng_state(),'config':CFG,
 'source_lock_sha256':__import__('hashlib').sha256((RUN/'source_lock.json').read_bytes()).hexdigest()}
 p=RUN/'checkpoints/last.pt';tmp=p.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(p)
 if step in [0,20,50] or step%250==0:
  import shutil;shutil.copy2(p,RUN/('checkpoints/step%04d.pt'%step))
