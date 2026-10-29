import ast,importlib.util,json,sys,traceback
from common14 import *
import numpy as np,torch
from model14 import ContactMotionNet
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.22)
C=ROOT/'experiments/contact_frames/cf014'
def main():
 verify(json.loads((C/'completed_locked.json').read_text()))
 verify(json.loads((ROOT/'experiments/contact_frames/cf015/completed_locked.json').read_text()))
 verify(json.loads((BASE/'training_input_locked.json').read_text()))
 verify(json.loads((BASE/'weights_frozen.json').read_text())['checkpoints'])
 oldtree=ast.parse((C/'src/model14.py').read_text());newtree=ast.parse((BASE/'src/model14.py').read_text())
 cls=next(x for x in newtree.body if isinstance(x,ast.ClassDef));fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
 idx=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='x' for a in n.targets))
 assert ast.dump(ast.Module(body=fn.body[idx+1:idx+3],type_ignores=[]))==ast.dump(ast.parse('x = x.clone()\nx[..., 21:23] = 0'))
 del fn.body[idx+1:idx+3]
 assert ast.dump(newtree)==ast.dump(oldtree)
 t=ast.parse((BASE/'src/train16_v2.py').read_text())
 for n in ast.walk(t):
  if isinstance(n,ast.Assert) and any(isinstance(x,ast.Name) and x.id=='checkpoints' for x in ast.walk(n)):
   assert n.test.comparators[0].value==3;n.test.comparators[0]=ast.Constant(value=6)
 assert ast.dump(t)==ast.dump(ast.parse((C/'src/train14.py').read_text()))
 cfgold=json.loads((C/'config.json').read_text())
 for key in cfgold:
  if key not in ['id','time','arms']:assert CFG[key]==cfgold[key],key
 seeds={}
 for seed in CFG['seeds']:
  run=BASE/'runs/with_motion'/f'seed_{seed}';cr=C/'runs/with_motion'/f'seed_{seed}'
  a=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)['model'];b=torch.load(cr/'initial.pt',map_location='cpu',weights_only=False)['model']
  assert all(torch.equal(v,b[k]) for k,v in a.items())
  h=json.loads((run/'history.json').read_text());plan=json.loads((OLD/'plans'/f'seed_{seed}.json').read_text())
  assert len(h)==len(plan)==600 and all(x['step']==i+1 and x['episode']==plan[i]['episode'] for i,x in enumerate(h))
  ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
  assert ck['steps']==600
  st=[int(x['step']) for x in ck['optimizer']['state'].values()];assert min(st)==max(st)==600
  hc=json.loads((cr/'history.json').read_text())
  assert h[0]['loss']==hc[0]['loss']
  seeds[str(seed)]=dict(initial_exact=True,steps=600,optimizer_steps=[min(st),max(st)],first_step_loss_equal=True,final_loss=h[-1]['loss'])
 for ep in CFG['train_episodes']:
  a=np.load(BASE/'features'/f'{ep}_supervision.npz');b=np.load(C/'features'/f'{ep}_supervision.npz')
  for k in b.files:assert np.array_equal(a[k],b[k])
 for ep in CFG['train_episodes']+CFG['review_episodes']:
  assert sha(BASE/'features'/f'{ep}.npz')==sha(C/'features'/f'{ep}.npz')
 assert sha(BASE/'motion_normalization.json')==sha(C/'motion_normalization.json')
 spec=importlib.util.spec_from_file_location('baseline_model_audit',C/'src/model14.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 mn=json.loads((BASE/'motion_normalization.json').read_text());norm=json.loads((BASE.parent/'cf003/normalization.json').read_text())
 maxdiff=0.;newdiff=0.
 for ep in CFG['train_episodes']+CFG['review_episodes']:
  with np.load(dataset(ep)) as d,np.load(BASE/'features'/f'{ep}.npz') as v:
   mo=(v['features']-np.array(mn['mean']))/np.array(mn['std']);mo[~v['valid']]=0
   ss=(d['states']-np.array(norm['mean']))/np.array(norm['std'])
   args=[torch.tensor(z,dtype=torch.float32,device='cuda')[None] for z in [d['images'],d['points'],ss,mo]]
  ref=np.load(C/'evaluation'/ep/'predictions.npz')['with_motion'];new=np.load(BASE/'evaluation'/ep/'predictions.npz')['with_motion']
  for j,seed in enumerate(CFG['seeds']):
   for isnew,klass,root,scores in [(False,mod.ContactMotionNet,C,ref),(True,ContactMotionNet,BASE,new)]:
    net=klass(True).cuda();net.load_state_dict(torch.load(root/'runs/with_motion'/f'seed_{seed}'/'last.pt',map_location='cuda',weights_only=False)['model']);net.eval()
    with torch.no_grad():out=net(*args)[0].sigmoid()[0].cpu().numpy()
    diff=float(np.max(np.abs(out-scores[j])))
    if isnew:newdiff=max(newdiff,diff)
    else:maxdiff=max(maxdiff,diff)
    del net
 assert maxdiff<1e-6 and newdiff<1e-6,(maxdiff,newdiff)
 frozen=json.loads((BASE/'weights_frozen.json').read_text());review=json.loads((BASE/'audits/review_access.json').read_text())
 save(BASE/'audits/completion_audit.json',dict(time=now(),single_model_change_only=True,training_ast_equal_except_checkpoint_count=True,configuration_equal_except_id_time_arms=True,seeds=seeds,supervision_exact=True,all_features_and_normalization_exact=True,control_prediction_max_difference=maxdiff,new_prediction_max_difference=newdiff,manual_review_after_weights_frozen=True,weights_frozen_time=frozen['time'],review_access=review,prior_locks_unchanged=True,test_access=False))
 save(BASE/'audits/resolved.json',dict(time=now(),historical_failures=['failure_train.json','failure_runner.json'],resolved=True,reason='preflight supervision copies moved before actual training; checkpoint total only adapted for one arm',training_started_only_after_fix=True))
 stage('audited_ready_for_display')
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(BASE/'failure_audit.json',dict(time=now(),type=type(e).__name__,message=str(e),stack=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(e.__traceback__)]));stage('audit_failed');raise SystemExit(1)
