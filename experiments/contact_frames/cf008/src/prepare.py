import pathlib,json,hashlib,shutil,ast,numpy as np
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA');BASE=ROOT/'experiments/contact_frames';RUN=BASE/'cf008'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=json.loads((RUN/'config.json').read_text());rows=[];locks={}
for ep in cfg['train_episodes']:
 src=BASE/'cf003/data'/ep;dpath=src/'artifacts/dataset.npz';refpath=src/'artifacts/reference.npz';cp=src/'config.json'
 mpath=ROOT/json.loads(cp.read_text())['motion']
 for p in [dpath,refpath,cp,mpath]:locks[str(p)]=sha(p)
 d=dict(np.load(dpath));P=np.load(refpath)['surface_xyz'];H=np.load(mpath)['hand_to_world'];O=d['visual_transforms'];N=len(O)
 rel=d['geometry_reliable']&d['state_valid'];q=d['quality'];sig=d['sigma']
 separated=rel&(d['distance']-2*sig>.006)
 y=np.zeros(N,np.float32);w=np.zeros(N,np.float32);support=np.zeros(N,np.float32);pc=np.zeros(N,np.int64);pairs=[]
 for t in range(1,N):
  if not rel[t]:continue
  if separated[t]:
   w[t]=d['weight'][t];continue
  xt=P@O[t,:3,:3].T+O[t,:3,3]
  hand=np.linalg.solve(H[t,:3,:3],(xt-H[t,:3,3]).T).T
  ss=[];qq=[]
  for lag in cfg['pair_offsets']:
   s=t+lag
   if not (0<s<N) or not rel[s] or np.any(separated[min(s,t):max(s,t)+1]):continue
   xs=P@O[s,:3,:3].T+O[s,:3,3]
   ph=hand@H[s,:3,:3].T+H[s,:3,3]
   e0=np.sum((xs-xt)**2,axis=1);e1=np.sum((xs-ph)**2,axis=1)
   logratio=float(np.median(e0-e1)/(2*(sig[t]**2+sig[s]**2)))
   ev=float(max(0,np.tanh(logratio/2)))
   ss.append(ev);qq.append(float(q[s]))
   pairs.append({'t':t,'s':s,'logratio':logratio,'support':ev,'static_median_mm':float(np.median(np.sqrt(e0))*1000),'hand_median_mm':float(np.median(np.sqrt(e1))*1000)})
  pc[t]=len(ss)
  if ss:
   support[t]=np.average(ss,weights=qq);y[t]=d['target'][t]*support[t];w[t]=q[t]*np.mean(qq)
 out=RUN/'data'/ep/'artifacts';out.mkdir(parents=True)
 d['near_target']=d['target'].copy();d['near_weight']=d['weight'].copy();d['target']=y;d['weight']=w
 d['interaction_support']=support;d['interaction_pairs']=pc
 assert np.isfinite(y).all() and np.isfinite(w).all() and np.all((y>=0)&(y<=1)) and np.all(w>=0)
 orig=np.load(dpath)
 for k in orig.files:
  if k not in ['target','weight']:assert np.array_equal(d[k],orig[k],equal_nan=True)
 np.savez_compressed(out/'dataset.npz',**d)
 (out/'motion_evidence.json').write_text(json.dumps(pairs,indent=2))
 rows.append({'episode':ep,'frames':N,'weighted':int((w>0).sum()),'effective_high_frames':np.flatnonzero((y>=.8)&(w>0)).tolist(),'effective_mid_frames':np.flatnonzero((y>=.2)&(y<.8)&(w>0)).tolist(),'supported_none_near_high':int(((d['near_target']>=.8)&(w>0)&(y<.2)).sum()),'unknown_weight_zero':int((w==0).sum()),'pairs':len(pairs),'all_other_original_arrays_unchanged':True})
(RUN/'plans').mkdir()
for seed in cfg['seeds']:
 p=BASE/'cf005/plans'/('seed_'+str(seed)+'.json');locks[str(p)]=sha(p);shutil.copyfile(p,RUN/'plans'/p.name)
 p=BASE/'cf005/runs/all_train_frames'/('seed_'+str(seed))/'initial.pt';locks[str(p)]=sha(p)
locks[str(BASE/'cf003/normalization.json')]=sha(BASE/'cf003/normalization.json')
locks[str(BASE/'cf003/src/train_and_evaluate.py')]=sha(BASE/'cf003/src/train_and_evaluate.py')
original=BASE/'cf005/src/model_all_train_frames.py';locks[str(original)]=sha(original)
tree=ast.parse(original.read_text())
for cls in tree.body:
 if isinstance(cls,ast.ClassDef):
  for fn in cls.body:
   if isinstance(fn,ast.FunctionDef) and fn.name=='__init__':
    fn.body+=ast.parse('self.interaction = torch.nn.Sequential(torch.nn.Linear(128,32), torch.nn.GELU(), torch.nn.Linear(32,1))').body
   if isinstance(fn,ast.FunctionDef) and fn.name=='forward':
    ret=next(x for x in fn.body if isinstance(x,ast.Return))
    assert len(ret.value.elts)==2
    near,aux=ret.value.elts
    ret.value=ast.Tuple(elts=[ast.parse('self.interaction(last).squeeze(-1)',mode='eval').body,aux,near],ctx=ast.Load())
tree.body.insert(0,ast.Import(names=[ast.alias(name='torch')]))
ast.fix_missing_locations(tree);code=ast.unparse(tree);compile(code,'model.py','exec');(RUN/'src/model.py').write_text(code)
assert all(sha(pathlib.Path(p))==h for p,h in locks.items())
(RUN/'input_locked.json').write_text(json.dumps(locks,indent=2))
(RUN/'train_supervision_summary.json').write_text(json.dumps({'rows':rows,'no_review_targets_read':True,'feedback_not_used':True},ensure_ascii=False,indent=2))
print(json.dumps(rows,ensure_ascii=False))
