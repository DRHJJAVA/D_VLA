import os,json,sys,ast,subprocess,traceback
from pathlib import Path
from common14 import *
import torch,numpy as np,random
from train14 import load_data,forward,losses
from model14 import ContactMotionNet
CONTROL=ROOT/'experiments/contact_frames/cf014'
def main():
 stage('preflight')
 verify(json.loads((CONTROL/'completed_locked.json').read_text()))
 data,norm=load_data()
 assert (BASE/'src/train14.py').read_bytes()==(CONTROL/'src/train14.py').read_bytes()
 assert ast.dump(ast.parse((BASE/'src/model14.py').read_text()).body[-1].body[-1]) 
 target_checks={}
 for ep in CFG['train_episodes']:
  a=np.load(BASE/'features'/f'{ep}_supervision.npz');b=np.load(CONTROL/'features'/f'{ep}_supervision.npz')
  for k in b.files:assert np.array_equal(a[k],b[k]),(ep,k)
  target_checks[ep]=list(a.files)
 init_checks={}
 for seed in CFG['seeds']:
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
  net=ContactMotionNet(True).cuda()
  init=torch.load(OLD/'runs/full_history'/f'seed_{seed}'/'initial.pt',map_location='cuda',weights_only=False)
  net.base.load_state_dict(init['model'])
  ref=torch.load(CONTROL/'runs/with_motion'/f'seed_{seed}'/'initial.pt',map_location='cuda',weights_only=False)['model']
  assert all(torch.equal(v,ref[k]) for k,v in net.state_dict().items())
  init_checks[str(seed)]=True
 net.load_state_dict(torch.load(CONTROL/'runs/with_motion/seed_2026/last.pt',map_location='cuda',weights_only=False)['model']);net.eval()
 d=data[CFG['train_episodes'][0]];dd=dict(d);dd['motion']=d['motion'].clone();dd['motion'][...,21:23]=137.0
 with torch.no_grad():
  a=forward(net,d);b=forward(net,dd)
 delta=max(float((x-y).abs().max()) for x,y in zip(a,b));assert delta==0
 params=sum(p.numel() for p in net.parameters());assert params==231671
 save(BASE/'audits/single_factor_preflight.json',dict(time=now(),initial_exact=init_checks,supervision_exact=target_checks,masked_input_invariance=delta,parameters=params,training_source_byte_equal=True,control_lock_verified=True,manual_ranges_read=False))
 save(BASE/'source_locked.json',lock(list((BASE/'src').glob('*.py'))+[BASE/'config.json']))
 del net,data,dd,d;torch.cuda.empty_cache()
 stage('training')
 p=subprocess.run([sys.executable,'-u',str(BASE/'src/train14.py')],cwd=BASE,stdout=open(BASE/'train.log','w'),stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError('training_child_exit_'+str(p.returncode))
 stage('trained_pending_evaluation')
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(BASE/'failure_runner.json',dict(time=now(),type=type(e).__name__,message=str(e),stack=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(e.__traceback__)]))
  stage('failed')
  raise SystemExit(1)
