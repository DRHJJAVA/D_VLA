import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='2'
import pathlib,sys,torch,numpy as np,random,json,copy,traceback,time
N=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r3/joint/v001');sys.path.insert(0,str(N/'src'))
from network import JointNetwork
from geometry import policy_input,OLD
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(20261028);np.random.seed(20261028);torch.manual_seed(20261028)
    ep=json.loads((N/'episodes.json').read_text())[0];d,y,_,_=policy_input(ep,67);d={k:v.cuda() for k,v in d.items()}
    sys.path.insert(0,str(OLD/'src'));import common
    official=common.build_expert('cpu');expert=official.lm_expert;del official
    m=JointNetwork(expert).cuda().train();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    def step():
        opt.zero_grad(set_to_none=True);e=m.encode(d);z=torch.randn(8,16,6,device='cuda')
        a,g=m.generate(e,z,object_index=0)
        l=(a-y['target'].cuda()).square().mean()+e['roles'].square().mean()+e['aux'].square().mean()
        l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        return float(l.detach())
    step()
    snapshot={k:v.detach().clone() for k,v in m.state_dict().items()}
    optimizer=copy.deepcopy(opt.state_dict());cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all();py=random.getstate();npstate=np.random.get_state()
    la=step();after={k:v.detach().clone() for k,v in m.state_dict().items() if ('a.weight' in k or 'b.weight' in k or not k.startswith('expert'))}
    m.load_state_dict(snapshot);opt.load_state_dict(optimizer);torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda);random.setstate(py);np.random.set_state(npstate)
    lb=step();diff=max(float((m.state_dict()[k]-v).abs().max()) for k,v in after.items())
    assert diff==0 and la==lb
    out={'passed':True,'parameter_max_difference':diff,'loss_difference':abs(la-lb),'optimizer_and_all_rng_restored':True,'engineering_optimizer_exercises':3,'formal_updates':0}
    (N/'checks/resume_equivalence.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':
    try:main()
    except Exception as e:
        d={'type':type(e).__name__,'summary':str(e)[:300],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]}
        (N/'checks/resume_failure.json').write_text(json.dumps(d));print(json.dumps(d))
