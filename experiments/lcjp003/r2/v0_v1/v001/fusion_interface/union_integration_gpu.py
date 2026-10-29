import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import pathlib,json,time,traceback,ast,hashlib,datetime
import numpy as np,torch
from fusion_module import FusionFront,LEGACY,OLD,HERE,parameter_digest,expert_forward,action_loss
def save(name,data):
    (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
def norm(x):
    return float(x.norm().detach().cpu())
def main():
    start=time.perf_counter()
    torch.set_num_threads(4);torch.manual_seed(20261027);np.random.seed(20261027)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    ns=LEGACY
    tree=ast.parse((OLD/'v0_corrected_check.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    setup=ast.Module(body=[n for n in fn.body if 136<=n.lineno<=142],type_ignores=[])
    exec(compile(ast.fix_missing_locations(setup),'<official_expert_setup>','exec'),ns)
    backbone=ns['backbone']
    assert all(not p.requires_grad for p in backbone.parameters())
    assert len(backbone.state_dict())==490
    official_before=parameter_digest(backbone)
    data=torch.load(HERE/'union_integration_inputs.pt',map_location='cpu',weights_only=False)
    x,valid,h,words=[data[k].to('cuda') for k in ['x','valid','h','words']]
    target=data['target'].to('cuda');grip=data['grip'].to('cuda')
    b=len(x)
    if target.ndim==2:target=target[None]
    if grip.ndim==1:grip=grip[None]
    target=target.expand(b,-1,-1);grip=grip.expand(b,-1)
    torch.manual_seed(20261027);front=FusionFront().to('cuda').eval()
    before=parameter_digest(front)
    gen=torch.Generator(device='cpu').manual_seed(20261027)
    noise=torch.randn(b,16,6,generator=gen).to('cuda')
    time_scalar=torch.full((b,1,1),.35,device='cuda')
    # Warm actual FP32 expert path with the same shape, without update.
    with torch.no_grad():
        warm=front.encode(x,valid,h,words)
        expert_forward(front,backbone,warm,noise,time_scalar,target)
    torch.cuda.synchronize()
    fstart=time.perf_counter()
    mixed=front.encode(x,valid,h,words)
    part=front.encode(x,valid,h,words,part_only=True)
    for key in ['scores','roles','lambda']:mixed[key].retain_grad()
    part['scores'].retain_grad()
    motion,gg=expert_forward(front,backbone,mixed,noise,time_scalar,target)
    pmotion,pgg=expert_forward(front,backbone,part,noise,time_scalar,target)
    mix_loss=(action_loss(motion,gg,noise,target,grip)*mixed['role_prob']).sum()
    part_loss=(action_loss(pmotion,pgg,noise,target,grip)*mixed['role_prob'].detach()).sum()
    gm=torch.autograd.grad(mix_loss,[mixed['scores'],mixed['roles'],mixed['lambda'],front.gate[-1].weight],retain_graph=True)
    gp=torch.autograd.grad(part_loss,part['scores'],retain_graph=True)[0]
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gm)
    assert torch.isfinite(gp).all() and gp.abs().sum()>0
    beta=.5 # Test fixture, not selected training hyperparameter.
    total=mix_loss+beta*part_loss
    total.backward()
    torch.cuda.synchronize()
    forward_backward_seconds=time.perf_counter()-fstart
    finite=all(torch.isfinite(p.grad).all() for p in front.parameters() if p.grad is not None)
    assert finite and front.shared_queries.grad is not None and front.shared_queries.grad.norm()>0
    assert all(p.grad is None for p in backbone.parameters())
    grad_metrics={'mixed_score_norm':norm(gm[0]),'mixed_role_norm':norm(gm[1]),
       'mixed_lambda_norm':norm(gm[2]),'mixed_gate_last_weight_norm':norm(gm[3]),
       'independent_part_score_norm':norm(gp),'shared_queries_norm':norm(front.shared_queries.grad)}
    # Part-only path must be independent of lambda, including its gradient.
    front.zero_grad(set_to_none=True)
    low=front.encode(x,valid,h,words,lambda_override=.0001,part_only=True)
    high=front.encode(x,valid,h,words,lambda_override=.8,part_only=True)
    low_m,low_g=expert_forward(front,backbone,low,noise,time_scalar,target)
    high_m,high_g=expert_forward(front,backbone,high,noise,time_scalar,target)
    low_loss=action_loss(low_m,low_g,noise,target,grip).mean()
    high_loss=action_loss(high_m,high_g,noise,target,grip).mean()
    low_grad=torch.autograd.grad(low_loss,low['scores'])[0]
    high_grad=torch.autograd.grad(high_loss,high['scores'])[0]
    part_output_diff=max((low_m-high_m).abs().max().item(),(low_g-high_g).abs().max().item())
    part_gradient_diff=(low_grad-high_grad).abs().max().item()
    assert part_output_diff==0 and part_gradient_diff==0 and low_grad.norm()>0
    # Endpoints feed the SAME decoder, and common context remains unchanged.
    with torch.no_grad():
        zero=front.encode(x,valid,h,words,lambda_override=0)
        one=front.encode(x,valid,h,words,lambda_override=1)
        z_m,z_g=expert_forward(front,backbone,zero,noise,time_scalar,target)
        o_m,o_g=expert_forward(front,backbone,one,noise,time_scalar,target)
        endpoint_action_change=(z_m-o_m).abs().max().item()
        assert torch.isfinite(z_m).all() and torch.isfinite(o_m).all() and endpoint_action_change>0
        none=front.encode(x,torch.zeros_like(valid),h,words)
        n_m,n_g=expert_forward(front,backbone,none,noise,time_scalar,target)
        assert none['lambda'].abs().max()==0
        assert torch.isfinite(n_m).all() and torch.isfinite(n_g).all()
    after=parameter_digest(front);official_after=parameter_digest(backbone)
    assert before==after and official_before==official_after
    report={'status':'fusion_expert_interface_passed','formal_training_started':False,'optimizer_updates':0,
       'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),
       'module_sha256':hashlib.sha256((HERE/'fusion_module.py').read_bytes()).hexdigest(),
       'check_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
       'inputs_sha256':hashlib.sha256((HERE/'union_integration_inputs.pt').read_bytes()).hexdigest(),
       'candidate_count':b,'prefix_shape':list(mixed['prefix'].shape),'motion_shape':list(motion.shape),
       'gripper_shape':list(gg.shape),'fp32':True,'tf32':False,'shared_backbone_object':True,
       'official_loaded_tensor_count':len(backbone.state_dict()),'official_backbone_frozen':True,
       'mixed_action_loss':float(mix_loss.detach().cpu()),'part_action_loss':float(part_loss.detach().cpu()),
       'gradients':grad_metrics,'all_computed_gradients_finite':bool(finite),
       'small_lambda_part_gradient_norm':norm(low_grad),'independent_part_output_difference_across_lambda':part_output_diff,
       'independent_part_gradient_difference_across_lambda':part_gradient_diff,
       'endpoint_action_max_difference':endpoint_action_change,
       'no_part_expert_output_finite':True,'no_part_effective_lambda':0,
       'matching_target_noise_and_flow_time':True,'part_loss_not_scaled_by_lambda':True,
       'parameter_hash_before':before,'parameter_hash_after':after,
       'official_parameter_hash_before':official_before,'official_parameter_hash_after':official_after,
       'parameters_changed':False,'warmed_two_path_forward_backward_seconds':forward_backward_seconds,
       'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
       'total_seconds':time.perf_counter()-start,
       'test_fixtures_only':{'independent_part_loss_coefficient':beta,'flow_time':.35,'V2_hyperparameters_selected':False},
       'scope':'Interface and gradient connectivity on real observations and official frozen expert. No optimizer, trained semantics, learned scores, closed loop, or V2 outcome claimed.'}
    save('union_integration_gpu_result.json',report);save('union_integration_gpu_complete.json',{'status':report['status'],'optimizer_updates':0})
try:main()
except BaseException as e:
    save('union_integration_gpu_failure.json',{'type':type(e).__name__,'message':str(e)[:300],
       'locations':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]})
