import pathlib,json,hashlib,numpy as np,torch
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA');BASE=ROOT/'experiments/contact_frames';R=BASE/'cf008'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=json.loads((R/'config.json').read_text());checks=[];mx=0
for ep in cfg['train_episodes']:
 src=BASE/'cf003/data'/ep;cfg0=json.loads((src/'config.json').read_text())
 d=np.load(src/'artifacts/dataset.npz');n=np.load(R/'data'/ep/'artifacts/dataset.npz');P=np.load(src/'artifacts/reference.npz')['surface_xyz'];H=np.load(ROOT/cfg0['motion'])['hand_to_world'];O=d['visual_transforms'];N=len(O)
 rel=d['geometry_reliable']&d['state_valid'];sep=rel&(d['distance']-2*d['sigma']>.006);saved=json.loads((R/'data'/ep/'artifacts/motion_evidence.json').read_text());lookup={(x['t'],x['s']):x for x in saved};seen=set()
 yy=np.zeros(N);ww=np.zeros(N);supports=np.zeros(N)
 for t in range(1,N):
  if not rel[t]:continue
  if sep[t]:ww[t]=d['weight'][t];continue
  scores=[];weights=[]
  for s in range(max(1,t-8),min(N,t+9)):
   if s-t not in [-8,-4,-2,-1,1,2,4,8] or not rel[s]:continue
   if any(sep[k] for k in range(min(t,s),max(t,s)+1)):continue
   X=np.c_[P,np.ones(len(P))]
   a=X@O[t].T;b=X@O[s].T;c=a@(H[s]@np.linalg.inv(H[t])).T
   e0=np.sum((b[:,:3]-a[:,:3])**2,1);e1=np.sum((b[:,:3]-c[:,:3])**2,1)
   ratio=float(np.median(e0-e1)/(2*(d['sigma'][t]**2+d['sigma'][s]**2)))
   ev=float(max(0,np.tanh(ratio*.5)));rec=lookup[(t,s)];seen.add((t,s))
   mx=max(mx,abs(ratio-rec['logratio']));assert abs(ev-rec['support'])<1e-7
   scores.append(ev);weights.append(d['quality'][s])
  if scores:
   supports[t]=np.dot(scores,weights)/np.sum(weights);yy[t]=d['target'][t]*supports[t];ww[t]=d['quality'][t]*np.mean(weights)
 assert seen==set(lookup)
 assert np.allclose(yy,n['target'],atol=1e-7) and np.allclose(ww,n['weight'],atol=1e-7)
 assert np.array_equal(n['near_target'],d['target']) and np.array_equal(n['near_weight'],d['weight'])
 z=torch.zeros(N,requires_grad=True);torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor(n['target']),weight=torch.tensor(n['weight']),reduction='sum').backward()
 assert torch.all(z.grad[torch.tensor(n['weight']==0)]==0)
 checks.append({'episode':ep,'pairs':len(seen),'target_reconstructed':True,'near_aux_unchanged':True,'unknown_main_gradient_zero':True})
seeds=[]
for seed in cfg['seeds']:
 rd=R/'runs/evidence'/('seed_'+str(seed));a=torch.load(rd/'initial.pt',map_location='cpu',weights_only=False);b=torch.load(BASE/'cf005/runs/all_train_frames'/('seed_'+str(seed))/'initial.pt',map_location='cpu',weights_only=False)
 assert all(torch.equal(a['model'][k],b['model'][k]) for k in b['model'])
 assert sha(R/'plans'/('seed_'+str(seed)+'.json'))==sha(BASE/'cf005/plans'/('seed_'+str(seed)+'.json'))
 h=json.loads((rd/'history.json').read_text());ck=torch.load(rd/'last.pt',map_location='cpu',weights_only=False)
 assert len(h)==600 and ck['steps']==600 and all(int(v['step'])==600 for v in ck['optimizer']['state'].values())
 seeds.append({'seed':seed,'steps':600,'shared_initial_equal':True,'last_hash':sha(rd/'last.pt')})
for f in ['input_locked.json','source_locked.json','training_inputs_locked.json','evaluation_weights_locked.json']:
 lock=json.loads((R/f).read_text());assert all(sha(pathlib.Path(p))==v for p,v in lock.items()),f
result={'checks':checks,'max_pair_logratio_difference':mx,'seeds':seeds,'all_locks_unchanged':True,'main_unknown_not_negative':True,'low_support_does_not_assert_noncontact':True}
(R/'audits/independent_reconstruction.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
