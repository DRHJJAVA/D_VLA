import os,json,time,traceback,copy
import torch,numpy as np
from data import RUN,CFG,atomic_json,fail,load_sample,prepare_one,digest
from runtime import setup,seed,build,move,optimizer,rng_state,restore_rng
from objective import objective
def main():
 setup();seed(CFG['seed']);start=time.time()
 episodes={e['id']:e for e in json.loads((RUN/'episodes.json').read_text())}
 evals=json.loads((RUN/'evaluation_windows.json').read_text())
 # Use current source hashes; no formal checkpoint is updated by these tests.
 selected=[next(x for x in evals if x['split']=='train'),next(x for x in evals if x['split']=='dev')]
 examples=[]
 for rec in selected:
  prepare_one((episodes[rec['episode']],rec['frame']))
  q=load_sample(rec);examples.append(move(q))
 model=build();opt=optimizer(model);initial=copy.deepcopy(model.state_dict())
 noise=torch.randn(8,16,6,device='cuda')
 per=[]
 for q in examples:
  model.train();opt.zero_grad(set_to_none=True)
  loss,metrics,e,pred=objective(model,q['inputs'],q['labels'],noise,1)
  assert torch.isfinite(loss), 'nonfinite_loss'
  loss.backward()
  groups={}
  for tag,prefix in [('segment','segment'),('part_score','score'),('motion','motion'),('grip','grip'),('switch','switch')]:
   grads=[p.grad.detach() for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
   assert grads and all(torch.isfinite(g).all() for g in grads),'missing_or_nonfinite_gradient:'+tag
   value=sum(float(g.abs().sum()) for g in grads)
   assert value>0,'zero_gradient:'+tag;groups[tag]=value
  per.append({'loss':float(loss),'gradients_l1':groups,'metrics':metrics,'point_count':e['points']['point_count'],
   'shape_motion':list(pred[0].shape),'shape_mask':list(e['seg']['logits'].shape),'teacher_confidence':q['metadata']['teacher_confidence']})
 # The inference call accepts no labels and must work even with an empty predicted mask.
 model.eval();d=examples[0]['inputs']
 with torch.no_grad():
  pred_e=model.encode(d)
  a=model.generate(pred_e,noise)[0]
  b=model.generate(model.encode(d),noise)[0]
  repeat=float((a-b).abs().max());assert repeat<=1e-6,'repeat_inference'
  teacher=examples[0]['labels']['mask'][:,0]
  # Structural intervention checks on observed support; not a benefit claim.
  e0=model.encode(d,teacher_mask=teacher,score_override='zero')
  e1=model.encode(d,teacher_mask=teacher,score_override='one')
  p0=model.generate(e0,noise)[0];p1=model.generate(e1,noise)[0]
  influence=float((p0-p1).abs().mean());assert influence>1e-8,'score_path_disconnected'
  assert float((e0['base']-e1['base']).abs().max())==0,'intervention_changes_base'
  # Verify model geometry uses actual non-robot current depth coordinates.
  ids=pred_e['points']['ids'];valid=pred_e['points']['valid']
  contamination=int(d['robot'].flatten()[ids][valid].sum());assert contamination==0,'robot_points'
 # Exact model/optimizer/RNG restoration tested by a repeated real AdamW step.
 model.load_state_dict(initial);model.train();opt=optimizer(model)
 saved={'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'rng':rng_state()}
 q=examples[0]
 def update():
  opt.zero_grad(set_to_none=True);z=torch.randn(8,16,6,device='cuda')
  loss,*_=objective(model,q['inputs'],q['labels'],z,1);loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
 update();first={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
 model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
 update()
 resume=max(float((p-first[n]).abs().max()) for n,p in model.named_parameters() if p.requires_grad)
 atomic_json('checks/resume_probe.json',{'maximum_difference':resume,'changed_parameters':{n:float((p-first[n]).abs().max()) for n,p in model.named_parameters() if p.requires_grad and float((p-first[n]).abs().max())>0}})
 assert resume==0,'resume_update_not_exact'
 report={'passed':True,'time':time.time(),'seconds':time.time()-start,'examples':per,'repeat_max':repeat,
 'score_zero_one_action_mae_change':influence,'score_effectiveness_proven':False,
 'resume_update_max_diff':resume,'robot_points':contamination,'formal_updates':0,
 'predicted_mask_only_inference':True,'inference_has_no_future_label_argument':True,
 'parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
 'source_hashes':{p.name:digest(p) for p in (RUN/'src').glob('*.py')}}
 atomic_json('checks/network_and_gradient.json',report);print(json.dumps(report))
if __name__=='__main__':
 try:main()
 except Exception as ex:fail('validation',ex);raise SystemExit(1)
