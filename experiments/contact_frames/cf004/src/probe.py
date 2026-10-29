from pathlib import Path
import numpy as np,torch,json,importlib.util,hashlib,datetime
torch.set_num_threads(4)
ROOT=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames')
B=ROOT/'cf004'
spec=importlib.util.spec_from_file_location('model',ROOT/'cf003/src/model_no_world.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
ep='white_yellow_mug_demo_5'
d=np.load(ROOT/'cf003/evaluation_no_world'/ep/'artifacts/dataset.npz')
p=np.load(ROOT/'cf003/evaluation_no_world'/ep/'artifacts/frame_predictions.npz')
frames=np.array([44,46,48,50,52,60,85])
ix=p['context_indices'][frames]
records=[]
for seed in [2026,2027,2028]:
 c=torch.load(ROOT/f'cf003/runs/no_world/seed_{seed}/last.pt',map_location='cpu',weights_only=False)
 m=mod.ContactFrameNet(50).cuda().eval();m.load_state_dict(c['model'])
 st=(d['states']-np.asarray(c['state_mean']))/np.asarray(c['state_std'])
 I=torch.tensor(d['images'][ix],device='cuda'); P=torch.tensor(d['points'][ix],device='cuda'); S=torch.tensor(st[ix],device='cuda',dtype=torch.float32)
 groups={'original':None,'action':[18,25,43,50],'fingers':[9,11,34,36],'joints':[11,18,36,43],'pose_delta':[25,34],'all_state':[0,50],'image':None,'points':None}
 row={'seed':seed,'frames':frames.tolist(),'probes':{}}
 with torch.no_grad():
  for g,rg in groups.items():
   ii=I.clone();pp=P.clone();ss=S.clone()
   if rg:
    for a,z in zip(rg[::2],rg[1::2]):ss[...,a:z]=0
   if g=='image':ii.zero_()
   if g=='points':pp.zero_()
   pr=m(ii,pp,ss)[0].sigmoid().cpu().numpy().reshape(-1)
   row['probes'][g]=pr.tolist()
 records.append(row)
stats=[]
for ep in json.loads((ROOT/'cf003/config.json').read_text())['train_episodes']:
 a=np.load(ROOT/'cf003/data'/ep/'artifacts/dataset.npz')
 for name,mask in [('weak_negative',a['train']&(a['weight']>0)&(a['target']<=.2)),('weak_positive',a['train']&(a['weight']>0)&(a['target']>=.8))]:
  s=a['states'][mask]
  stats.append({'episode':ep,'group':name,'n':len(s),'action_mean':s[:,18:25].mean(0).tolist() if len(s) else None,'gap_quantiles':np.quantile(s[:,9]-s[:,10],[0,.5,1]).tolist() if len(s) else None})
out={'frames_dependency_probe':records,'training_state_summary':stats,'interpretation':'mean replacement is OOD dependency probe, not contact counterfactual or labels'}
(B/'audits/input_dependency.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
