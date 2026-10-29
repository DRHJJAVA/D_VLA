import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='2'
import pathlib,sys,time,json,traceback
import numpy as np,torch
N=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r3/joint/v001')
sys.path.insert(0,str(N/'src'))
from geometry import policy_input,OLD
from objective import targets,loss
from network import JointNetwork
def main():
 torch.set_num_threads(4);torch.manual_seed(20261028);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 e=json.loads((N/'episodes.json').read_text())[0];d,y,o,m=policy_input(e,13);y=targets(e,13,d,y,o)
 assert set(d)=={'x','xyz','normal','valid','box','anchors','knn','interp','interp_weight','dual_visible','h','words','fingers'}
 assert torch.allclose(y['role'].sum(),torch.tensor(1.))
 assert torch.isfinite(y['parts']).all() and float(y['role_weight'])>0
 sys.path.insert(0,str(OLD/'src'));import common
 official=common.build_expert('cpu');ex=official.lm_expert;del official
 net=JointNetwork(ex).cuda().train();d={k:v.cuda() for k,v in d.items()};y={k:v.cuda() if torch.is_tensor(v) else v for k,v in y.items()}
 value,metrics=loss(net,d,y,torch.randn(8,16,6,device='cuda'));value.backward()
 gradients={}
 for name in ['role','destination','score','gate','aux']:
  gradients[name]=float(sum(p.grad.square().sum() for p in getattr(net,name).parameters() if p.grad is not None).sqrt())
  assert np.isfinite(gradients[name]) and gradients[name]>0,name
 assert torch.isfinite(value)
 out={'passed':True,'real_training_episode':e['id'],'frame':13,'loss':float(value.detach()),'metrics':metrics,'gradients':gradients,'formal_updates':0,'future_inputs_absent':True}
 (N/'checks/joint_objective.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':
 try:main()
 except Exception as e:
  d={'type':type(e).__name__,'summary':str(e)[:300],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]}
  (N/'checks/joint_failure.json').write_text(json.dumps(d));print(json.dumps(d));sys.exit(1)
