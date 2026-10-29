import pathlib,sys,json,traceback,torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from r5_modules import InstanceFront,R5Parts,HardTopParts,surface_score
from supervision import aggregate_response,restrict_future,require_supervision
torch.set_num_threads(4);torch.manual_seed(20261029)
checks={}
try:
 top=HardTopParts()
 s=torch.linspace(.01,.99,64)[None].requires_grad_();v=torch.ones_like(s,dtype=torch.bool)
 ids,valid,mat=top(s,v);checks['top_indices_exact']=torch.equal(ids,torch.arange(63,55,-1)[None])
 checks['forward_hard_exact']=torch.equal(mat,torch.nn.functional.one_hot(ids,64).float())
 (mat*torch.arange(64)[None,None]).sum().backward();checks['sort_gradient_finite_nonzero']=bool(torch.isfinite(s.grad).all() and s.grad.abs().sum()>0)
 _,vv,mm=top(s,torch.zeros_like(v));checks['empty_selection_zero']=bool(not vv.any() and mm.abs().sum()==0)
 front=InstanceFront().eval()
 with torch.no_grad():f=front(torch.randn(1,2,32,32,384),torch.randn(1,13,960),torch.randn(1,2,32,32,3))
 checks['front_shapes']=(list(f['masks'].shape)==[1,2,16,256,256] and list(f['role'].shape)==[1,2,17])
 net=R5Parts()
 data=dict(point_input=torch.randn(1,2048,268),group_index=torch.randint(0,2048,(1,64,32)),neighbor_valid=torch.ones(1,64,32,dtype=torch.bool),relative_points=torch.randn(1,64,32,3),geometry=torch.randn(1,64,16),history=torch.randn(1,16,42),history_valid=torch.ones(1,16,dtype=torch.bool),fingers=torch.randn(1,64,6),state=torch.randn(1,27),text=torch.randn(1,13,256),coarse=torch.randn(1,272),destination=torch.randn(1,256))
 out=net(**data);loss=(out['prefix']*torch.randn_like(out['prefix'])).sum()+out['prefix_bias'].sum();loss.backward()
 checks['prefix_shape']=list(out['prefix'].shape)==[1,12,256]
 checks['score_action_gradient']=all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.score.parameters()) and bool(net.score[-1].weight.grad.abs().sum()>0)
 checks['geometry_preserved']=torch.equal(out['part_geometry'],data['geometry'][:,out['part_indices'][0]])
 data['neighbor_valid'].zero_();out=net(**data)
 checks['empty_parts_finite']=bool(torch.isfinite(out['prefix']).all() and not out['part_valid'].any())
 response=torch.tensor([.9,.1]);confidence=torch.ones(2);assignment=torch.ones(2,1)
 avg,mass,known=aggregate_response(response,confidence,assignment,torch.tensor([0,1]))
 checks['mean_not_max']=abs(avg.item()-.5)<1e-6
 indices=torch.arange(6);segments=torch.tensor([0,0,0,1,1,1])
 checks['no_cross_attempt']=torch.equal(restrict_future(0,indices,segments),torch.tensor([False,True,True,False,False,False]))
 try:require_supervision({'teacher_version':'r4_future_max'})
 except AssertionError:checks['old_labels_rejected']=True
 else:checks['old_labels_rejected']=False
 weights=torch.rand(1,2048,64);ps,pv=surface_score(torch.ones(1,64)*.3,weights)
 checks['same_score_projection']=bool(torch.allclose(ps,torch.full_like(ps,.3),atol=1e-6))
 report={'stage':'module_contract_only','checks':{k:bool(v) for k,v in checks.items()},'passed':all(checks.values()),'training_started':False,'not_covered':['real_input_geometry','new_label_materialization','full_expert_gradient','training_inference_adapter','resume']}
 (ROOT/'checks/module_contract.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));sys.exit(0 if report['passed'] else 1)
except Exception as e:
 print(json.dumps({'error_type':type(e).__name__,'locations':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]}));sys.exit(1)
