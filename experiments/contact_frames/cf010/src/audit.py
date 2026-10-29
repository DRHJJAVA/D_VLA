from pathlib import Path
import json,hashlib,numpy as np,torch,importlib.util
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf010');OLD=R.parent/'cf008'
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((R/'config.json').read_text());checks={}
for name in ['input_locked.json','source_locked.json','evaluation_source_locked.json','evaluation_weights_locked.json']:
 lock=json.loads((R/name).read_text());assert all(sha(p)==v for p,v in lock.items());checks[name]=len(lock)
for name in ['cf008','cf009']:
 lock=json.loads((R.parent/name/'completed_locked.json').read_text());assert all(sha(p)==v for p,v in lock.items());checks[name]=len(lock)
rows=[]
for seed in cfg['seeds']:
 dest=R/'runs/full_history'/('seed_'+str(seed))
 init=torch.load(dest/'initial.pt',map_location='cpu',weights_only=False)['model']
 orig=torch.load(OLD/'runs/evidence'/('seed_'+str(seed))/'initial.pt',map_location='cpu',weights_only=False)['model']
 assert init.keys()==orig.keys() and all(torch.equal(v,orig[k]) for k,v in init.items())
 c=torch.load(dest/'last.pt',map_location='cpu',weights_only=False)
 h=json.loads((dest/'history.json').read_text());assert len(h)==600 and c['steps']==600
 assert all(np.isfinite([x['loss'],x['main'],x['near'],x['dyn']]).all() for x in h)
 assert max(int(v['step']) for v in c['optimizer']['state'].values())==600
 assert (R/'plans'/('seed_'+str(seed)+'.json')).read_bytes()==(OLD/'plans'/('seed_'+str(seed)+'.json')).read_bytes()
 rows.append({'seed':seed,'steps':600,'all_initial_parameters_equal':True,'plan_equal':True,'weight_hash':sha(dest/'last.pt')})
# independently invoke original last-only model on each selected prefix
s=importlib.util.spec_from_file_location('original_last_only',OLD/'src/model.py');mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
mx=0.;count=0
for ep in cfg['fixed_review_episodes']+['tomato_sauce_demo_7']:
 a=R/'evaluation'/ep/'artifacts';d=np.load(a/'dataset.npz');p=np.load(a/'frame_predictions.npz')
 for j,seed in enumerate(cfg['seeds']):
  ck=torch.load(R/'runs/full_history'/('seed_'+str(seed))/'last.pt',map_location='cuda',weights_only=False)
  m=mod.ContactFrameNet(50).cuda().eval();m.load_state_dict(ck['model'])
  args=[torch.tensor(d[k],device='cuda')[None] for k in ['images','points']]
  args.append(torch.tensor((d['states']-ck['state_mean'])/ck['state_std'],device='cuda')[None])
  for t in sorted(set([0,49,52,65,80,98,len(d['target'])-1])):
   if t>=len(d['target']):continue
   with torch.no_grad():e,aux,n=m(*[v[:,:t+1] for v in args])
   diff=max(abs(float(torch.sigmoid(e)[0])-float(p['probabilities'][j,t])),abs(float(torch.sigmoid(n)[0])-float(p['proximity_probabilities'][j,t])))
   mx=max(mx,diff);assert diff<2e-5
   count+=1
  del m
 reliable=d['geometry_reliable']&d['state_valid'];sep=reliable&(d['distance']-2*d['sigma']>.006)
 pred=(p['probabilities'].min(0)>=.8)&~sep
 assert np.array_equal(pred,p['predicted_contact']) and np.array_equal(pred&reliable,p['contact_candidate'])
causal=json.loads((R/'audits/temporal_causality.json').read_text())
assert causal['passed'] and all(z['future_perturbation_prefix_max_diff']==0 for z in causal['rows'])
out={'passed':True,'training':rows,'locks':checks,'independent_last_only_prefix_inferences':count,'max_probability_diff':mx,'future_perturbation_max_diff':0,'human_ranges_training':False,'prediction_vs_evidence_split':True}
(R/'audits/completion.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'passed':True,'count':count,'max_diff':mx}))
