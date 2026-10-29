from pathlib import Path
import json,hashlib,numpy as np,torch
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf009');OLD=R.parent/'cf008'
cfg=json.loads((R/'config.json').read_text());results=[]
for ep in cfg['train_episodes']:
 old=np.load(OLD/'data'/ep/'artifacts/dataset.npz')
 sd=np.load(R/'teacher_details'/ep/'supports.npz')
 pairs=json.loads((R/'teacher_details'/ep/'persistence.json').read_text())
 expected=sd['support'].copy()
 for z in pairs:
  t,s=z['t'],z['s']
  assert not np.any(sd['separation'][min(t,s):max(t,s)+1])
  assert abs(t-s)<=16 and sd['reliable'][t] and sd['reliable'][s]
  anchor=float(sd['support'][s]*old['near_target'][s]*np.clip(old['quality'][s],0,1))
  assert abs(anchor-z['anchor'])<1e-7
  c=anchor*np.exp(-(z['distance_m']**2+z['finger_delta_m']**2)/(2*z['variance']))
  assert abs(c-z['contribution'])<1e-7
  expected[t]=max(expected[t],c)
 assert np.allclose(expected,sd['persistent'],atol=1e-7)
 for arm in cfg['arms']:
  d=np.load(R/arm/'data'/ep/'artifacts/dataset.npz')
  support=sd['support'] if arm=='positive_evidence' else expected
  w=old['near_weight']*support*sd['reliable'];w[sd['separation']]=old['near_weight'][sd['separation']]
  y=old['near_target'].copy();y[sd['separation']]=0;y[w==0]=0
  assert np.allclose(d['weight'],w,atol=1e-7) and np.array_equal(y,d['target'])
  unsupported=(support==0)&~sd['separation']
  assert not np.any(d['weight'][unsupported])
  x=torch.zeros(len(y),requires_grad=True)
  loss=(torch.nn.functional.binary_cross_entropy_with_logits(x,torch.tensor(d['target']),reduction='none')*torch.tensor(d['weight'])).mean()
  loss.backward()
  assert not torch.any(x.grad[torch.tensor(d['weight']==0)])
  results.append({'episode':ep,'arm':arm,'unknown_gradient_zero':True,'reconstructed':True,'unsupported_frames':int(unsupported.sum())})
for name in ['source_locked.json','input_locked.json']:
 lock=json.loads((R/name).read_text())
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in lock.items())
(R/'audits/pre_training.json').write_text(json.dumps({'passed':True,'rows':results,'no_recursive_propagation':True,'no_separation_crossing':True},indent=2))
print(json.dumps({'passed':True,'rows':len(results)}))
