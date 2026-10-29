from adapter import *
import fcntl
def main():
 OUT.mkdir(exist_ok=True)
 lock=open(OUT/'probe.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save('state.json',{'status':'validating_live_adapter','time':stamp()})
 model=load_model();front=Front();renderer=render_module()
 episodes=json.loads((RUN/'episodes.json').read_text());e=next(e for e in episodes if 'alphabet_soup' in e['id'] and e['episode']=='demo_0')
 work=OUT/'validation';work.mkdir(exist_ok=True)
 render=renderer.LiveRender(e['id'],work,capacity=32)
 save('probe_schema.json',{'render_attributes':{k:str(type(v)) for k,v in render.__dict__.items()},'front_ready':True})
 path=work/'cache/rgbd'/ (e['id']+'.h5')
 words=torch.load(ROOT/'experiments/lcjp003/r2/v2/v001/cache'/ (e['task']+'_words.pt'),map_location='cpu',weights_only=False)
 if isinstance(words,dict):save('word_schema.json',{'keys':list(words)});raise RuntimeError('word_payload_not_tensor')
 with h5py.File(path,'r') as f:d=front.inputs(f,0,words,[])
 sample=torch.load(RUN/'cache'/e['id']/'0.pt',map_location='cpu',weights_only=False)
 comparison={}
 for k,v in d.items():
  ref=sample['inputs'][k];comparison[k]={'shape':list(v.shape),'reference_shape':list(ref.shape),'max_abs':float((v.float()-ref.float()).abs().max()) if v.shape==ref.shape else None}
 save('input_comparison.json',comparison)
 assert all(x['max_abs']==0 for x in comparison.values()),'live_initial_inputs_differ'
 for k in range(5):
  g=torch.Generator().manual_seed(20261028+k);a,_,_=predict(model,d,g)
  g=torch.Generator().manual_seed(20261028+k);b,_,_=predict(model,sample['inputs'],g)
  assert np.array_equal(a,b),'paired_actions_differ'
 acts=[];times=[];plan=None
 g=torch.Generator().manual_seed(20261028)
 for step in range(20):
  start=time.perf_counter()
  with h5py.File(path,'r') as f:di=front.inputs(f,step,words,acts)
  a,plan,enc=predict(model,di,g,plan)
  if step==0:
   from executor import draw
   image=draw(render.out,step,enc,0,step);assert image.shape==(300,512,3)
   save('render_schema.json',{'mask':list(enc['seg']['logits'].shape),'uv':list(enc['points']['uv'].shape),'points':list(enc['points']['xyz'].shape),'scores':list(enc['point_score'].shape)})
  executed=np.clip(a,-1,1);render.step(executed,step+1);acts.append(executed);times.append(time.perf_counter()-start)
 render.close()
 save('ready.json',{'passed':True,'time':stamp(),'all_initial_inputs_exact':True,'paired_actions_exact':5,'predict_step_observe_cycles':20,'step_seconds':times,'sam_online':False,'training_changed':False})
 save('state.json',{'status':'ready','time':stamp()})
if __name__=='__main__':
 try:main()
 except Exception as e:fail('preflight',e);raise SystemExit(1)
