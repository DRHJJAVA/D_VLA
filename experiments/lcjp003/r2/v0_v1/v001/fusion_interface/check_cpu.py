import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import pathlib,json,time,traceback,ast,io,hashlib
import numpy as np,torch
from fusion_module import FusionFront,LEGACY,OLD,HERE,causal_history,parameter_digest
def save(name,data):
    (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
def main():
    start=time.perf_counter()
    torch.set_num_threads(4);torch.manual_seed(20261027);np.random.seed(20261027)
    ns=LEGACY
    tree=ast.parse((OLD/'v0_corrected_check.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    statements=[n for n in fn.body if 130<=n.lineno<=147]
    class CPU(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='cuda':return ast.copy_location(ast.Constant('cpu'),n)
            return n
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Attribute) and n.func.attr=='cuda':
                n.func.attr='to';n.args=[ast.Constant('cpu')]
            return n
    setup=CPU().visit(ast.Module(body=statements,type_ignores=[]))
    exec(compile(ast.fix_missing_locations(setup),'<inherited_readonly_setup>','exec'),ns)
    x,valid,h,words,target,grip=[ns[k] for k in ['x','valid','h','words','target','grip']]
    assert all(t.device.type=='cpu' for t in [x,valid,h,words,target,grip])
    # Official backbone was loaded exactly as V0 but no CPU expert training.
    assert all(not p.requires_grad for p in ns['backbone'].parameters())
    loaded_seconds=time.perf_counter()-start
    torch.manual_seed(20261027)
    front=FusionFront().eval()
    before=parameter_digest(front)
    hooks=[];pointers={}
    for label,module in [('base',front.base_attention),('part',front.part_attention)]:
        hooks.append(module.register_forward_pre_hook(lambda m,inputs,label=label:pointers.__setitem__(label,inputs[0].data_ptr())))
    with torch.no_grad():
        a=front.encode(x,valid,h,words)
        zero=front.encode(x,valid,h,words,lambda_override=0)
        one=front.encode(x,valid,h,words,lambda_override=1)
        middle=front.encode(x,valid,h,words,lambda_override=.37)
        end0=(zero['mixed']-zero['base']).abs().max().item()
        end1=(one['mixed']-one['part']).abs().max().item()
        algebra=(middle['mixed']-(.63*middle['base']+.37*middle['part'])).abs().max().item()
        assert end0==0 and end1==0 and algebra==0
        assert pointers['base']==pointers['part']
        shared=(zero['common']-one['common']).abs().max().item()
        vv=valid.clone();vv[:,:,-8:]=False
        xx=x.clone();xx[~vv]=1e6
        pad_a=front.encode(x,vv,h,words)
        pad_b=front.encode(xx,vv,h,words)
        pad_max=max((pad_a[k]-pad_b[k]).abs().max().item() for k in ['prefix','lambda','scores'])
        assert pad_max==0
        null=front.encode(x,torch.zeros_like(valid),h,words)
        assert all(torch.isfinite(null[k]).all() for k in ['prefix','lambda','scores','base','part'])
        assert null['lambda'].abs().max()==0 and null['part'].abs().max()==0
        visual=x.clone();visual[...,:384]+=torch.randn_like(visual[...,:384])*.1
        visual_zero=front.encode(visual,valid,h,words,lambda_override=0)
        base_visual_difference=(zero['prefix']-visual_zero['prefix']).abs().max().item()
        assert base_visual_difference==0
        hh=h.clone();hh[-1]+=0.1
        gate_changed=front.encode(x,valid,hh,words)
        gate_change=(gate_changed['lambda']-a['lambda']).abs().max().item()
        assert gate_change>0
        sequence=torch.cat([torch.zeros_like(h[:5]),h,torch.zeros_like(h[:4])])
        future=sequence.clone();future[21:]=1e6
        h1=causal_history(sequence,20);h2=causal_history(future,20)
        assert torch.equal(h1,h) and torch.equal(h2,h)
        future_a=front.encode(x,valid,h1,words);future_b=front.encode(x,valid,h2,words)
        future_max=(future_a['prefix']-future_b['prefix']).abs().max().item()
        assert future_max==0
        front.warmup_updates.fill_(37)
        state_before=front.encode(x,valid,h,words)
        memory=io.BytesIO();torch.save(front.state_dict(),memory);memory.seek(0)
        other=FusionFront().eval();other.load_state_dict(torch.load(memory,map_location='cpu',weights_only=True),strict=True)
        state_after=other.encode(x,valid,h,words)
        restore=max((state_before[k]-state_after[k]).abs().max().item() for k in ['prefix','lambda'])
        assert restore==0 and int(other.warmup_updates)==37
        assert front.cap().item()==other.cap().item()
        front.warmup_updates.zero_()
    for hook in hooks:hook.remove()
    after=parameter_digest(front);assert before==after
    data={k:ns[k].detach().cpu() for k in ['x','valid','h','words','target','grip']}
    data.update({'history_indices':ns['idx'],'candidate_ids':ns['ids'],'episode':ns['e']['id'],'frame':ns['t']})
    torch.save(data,HERE/'real_interface_inputs.pt')
    report={'status':'cpu_interface_passed','formal_training_started':False,'optimizer_updates':0,
        'source':str(OLD/'v0_corrected_check.py'),'source_sha256':hashlib.sha256((OLD/'v0_corrected_check.py').read_bytes()).hexdigest(),
        'module_sha256':hashlib.sha256((HERE/'fusion_module.py').read_bytes()).hexdigest(),
        'real_input':{'episode':data['episode'],'frame':data['frame'],'candidate_count':len(x),'shape':list(x.shape),'history':[int(i) for i in data['history_indices']]},
        'shapes':{k:list(a[k].shape) for k in ['prefix','prefix_valid','base','part','mixed','common','gate_input','lambda']},
        'endpoint_lambda0_max_difference':end0,'endpoint_lambda1_max_difference':end1,'blend_formula_max_difference':algebra,
        'shared_query_parameter':True,'same_query_tensor_used_by_both_attention_calls':pointers['base']==pointers['part'],
        'common_context_unchanged_by_lambda_max':shared,'masked_padding_perturbation_max':pad_max,
        'all_part_missing_finite':True,'all_part_missing_effective_lambda':0.0,
        'base_branch_visual_perturbation_max':base_visual_difference,'input_conditioned_gate_change':gate_change,
        'future_history_perturbation_max':future_max,'state_restoration_max':restore,
        'checkpoint_restored_warmup_updates':37,'checkpoint_restored_cap':other.cap().item(),
        'probe_only_warmup_config':{'initial_cap':.2,'length':100,'not_V2_hyperparameters':True},
        'lambda_range':[a['lambda'].min().item(),a['lambda'].max().item()],
        'parameter_hash_before':before,'parameter_hash_after':after,'parameters_changed':False,
        'new_front_parameters':sum(p.numel() for p in front.parameters()),
        'loaded_setup_seconds':loaded_seconds,'total_cpu_seconds':time.perf_counter()-start,
        'gpu_used':False,'gpu_expert_gradient_check':'pending'}
    save('cpu_result.json',report)
    save('cpu_complete.json',{'status':report['status'],'optimizer_updates':0})
try:main()
except BaseException as e:
    save('cpu_failure.json',{'type':type(e).__name__,'message':str(e)[:300],
       'locations':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]})
