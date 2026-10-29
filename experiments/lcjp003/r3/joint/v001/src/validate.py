import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('JAX_PLATFORMS','cpu')
import sys,pathlib,json,time,traceback,hashlib
import numpy as np,torch
R=pathlib.Path('/mnt/nvme1/drh/D_VLA')
RUN=R/'experiments/lcjp003/r3/joint/v001'
sys.path.insert(0,str(RUN/'src'))
from geometry import policy_input,OLD
from network import JointNetwork
from drifting import drift_loss

def save(n,d):(RUN/'checks'/n).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def main():
    torch.set_num_threads(4);torch.manual_seed(20261028);np.random.seed(20261028)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    e=json.loads((RUN/'episodes.json').read_text())[0]
    ts=sorted(int(p.stem) for p in (OLD/'cache'/e['id']).glob('*.pt'))
    t=ts[len(ts)//2]
    t0=time.time();d,y,objects,meta=policy_input(e,t)
    assert all(d['valid'].sum(1)>0)
    assert 'target' not in d and 'future' not in d and 'grip' not in d
    save('input_schema.json',{'episode':e['id'],'frame':t,'shapes':{k:list(v.shape) for k,v in d.items()},'seconds':time.time()-t0,'objects':len(objects),'candidate_ids':[o['ids'] for o in objects],'crossview_pairs':meta['crossview_pairs'],'robot_points_removed':sum(o['removed_robot'] for o in objects)})
    sys.path.insert(0,str(OLD/'src'))
    import common
    official=common.build_expert('cpu')
    expert=official.lm_expert
    del official
    model=JointNetwork(expert).cuda().train()
    d={k:v.cuda() for k,v in d.items()};target=y['target'].cuda();grip=y['grip'].cuda()
    z=torch.randn(8,16,6,device='cuda')
    enc=model.encode(d)
    a,g=model.generate(enc,z,object_index=0)
    assert a.shape==(8,16,6) and g.shape==(8,16,2)
    action_loss=drift_loss(a.flatten(1)[None],target.flatten()[None,None]).mean()+torch.nn.functional.cross_entropy(g.flatten(0,1),grip.expand(8,-1).flatten())
    action_grads=torch.autograd.grad(action_loss,tuple(model.score.parameters()),retain_graph=True,allow_unused=True)
    score_action_norm=float(sum(v.square().sum() for v in action_grads if v is not None).sqrt())
    assert score_action_norm>0 and np.isfinite(score_action_norm)
    q=torch.ones_like(enc['roles'])/enc['roles'].numel()
    loss=action_loss-(q*enc['roles'].log_softmax(-1)).sum()+enc['aux'].square().mean()
    loss.backward()
    grad={}
    for name in ['point','object_text','role','score','gate','part_proj','motion','grip','aux']:
        ps=[p.grad for p in getattr(model,name).parameters() if p.grad is not None]
        grad[name]=float(sum(p.square().sum() for p in ps).sqrt()) if ps else 0.
        assert np.isfinite(grad[name]) and grad[name]>0,name
    assert all(p.grad is None for n,p in model.expert.named_parameters() if not p.requires_grad)
    model.eval()
    with torch.no_grad():
        aa,_=model.generate(model.encode(d),z[:1],object_index=0)
        ab,_=model.generate(model.encode(d),z[:1],object_index=0)
        eu=model.encode(d,'uniform');au,_=model.generate(eu,z[:1],object_index=0)
        base0,_=model.generate(model.encode(d),z[:1],object_index=0,lambda_override=0)
        base1,_=model.generate(eu,z[:1],object_index=0,lambda_override=0)
    repeat=float((aa-ab).abs().max());base_diff=float((base0-base1).abs().max());delta=float((aa-au).abs().mean())
    assert repeat<=1e-6 and base_diff<=1e-6
    assert delta>0
    save('network_interface.json',{'passed':True,'formal_updates':0,'synthetic_supervision_for_interface_test_only':True,'real_input_episode':e['id'],'frame':t,'action_shape':list(a.shape),'score_action_gradient_norm':score_action_norm,'module_gradient_norms':grad,'same_input_repeat_max':repeat,'lambda_zero_score_intervention_max':base_diff,'untrained_score_intervention_action_mae':delta,'not_a_score_utility_claim':True,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.time()-t0})
    print(json.dumps({'network_interface_passed':True,'score_action_gradient_norm':score_action_norm,'seconds':time.time()-t0}))
if __name__=='__main__':
    try:main()
    except Exception as e:
        frames=[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]
        save('validation_failure.json',{'type':type(e).__name__,'message':str(e)[:300],'frames':frames})
        print(json.dumps({'failed':type(e).__name__,'frames':frames}));sys.exit(1)
