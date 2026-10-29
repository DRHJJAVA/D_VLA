from pathlib import Path
import sys,json,hashlib,random
import numpy as np,torch
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose/iter_016';O=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(O/'src'))
import train as T
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
 for f in ['source_locked.json','input_locked.json','control_locked.json']:
  for p,h in json.loads((O/f).read_text()).items():assert sha(P/p)==h,(f,p)
 m=json.loads((O/'manifest.json').read_text());tr=[r for r in m['rows'] if r['split']=='train'];va=[r for r in m['rows'] if r['split']=='val'];assert len(tr)==24 and len(va)==8
 assert {r['episode'] for r in tr}.isdisjoint({r['episode'] for r in va})
 for r in m['rows']:
  for folder in ['samples','graphs']:assert sha(O/folder/(r['key']+'.npz'))==sha(B/folder/(r['key']+'.npz'))
  with np.load(O/'samples'/(r['key']+'.npz')) as z:
   y=z['target'].astype(bool);k=z['known'];idx=z['reference_source_index'];assert np.all(z['confidence'][~k]==0) and not np.any(y&~k)
   assert len(np.unique(idx[y]))>=3 and len(np.unique(idx[k&~y]))>=16
 for name in ['model.py','pointnet2_grasp.py']:assert sha(O/'src'/name)==sha(B/'src'/name)
 torch.manual_seed(2026);net=T.RegionPoseNet().cuda();init=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest();del net
 for v in ['object_only','pose_conditioned']:assert init==json.loads((B/'runs'/('multi_'+v)/'initialization.json').read_text())['state_sha256']
 random.seed(2026);events=sorted({r['event'] for r in tr});plan=[]
 for epoch in range(1,33):
  order=events.copy();random.shuffle(order)
  plan.append([dict(event=ev,offset=[15,10,5,1][(epoch-1+events.index(ev))%4],key=next(r['key'] for r in tr if r['event']==ev and r['offset']==[15,10,5,1][(epoch-1+events.index(ev))%4])) for ev in order])
 for v in ['object_only','pose_conditioned']:
  h=json.loads((B/'runs'/('multi_'+v)/'history.json').read_text());assert [x['selected_queries'] for x in h]==plan
 checks=[]
 samples=T.loadrows(dict(rows=tr),torch.device('cuda'))
 for r,d in samples:
  s=torch.linspace(-3,3,len(d['target']),device='cuda').requires_grad_(True);s.data[~d['known']]=100
  ix=T.hard_negative_indices(s,d);neg=torch.nonzero(d['known'] & (d['target']<=.5)).flatten()
  assert len(ix)==min(64,len(neg)) and torch.all(d['known'][ix]) and torch.all(d['target'][ix]==0)
  assert torch.all(s[ix]>=torch.sort(s[neg],descending=True).values[len(ix)-1])
  loss=T.train_loss(s,d);g=torch.autograd.grad(loss,s)[0];assert torch.all(g[~d['known']]==0) and torch.isfinite(g).all()
  b=torch.nn.functional.binary_cross_entropy_with_logits(s,d['target'],reduction='none');w=d['confidence'];n=(b[neg]*w[neg]).sum()/w[neg].sum();hard=(b[ix]*w[ix]).sum()/w[ix].sum()
  assert torch.allclose(loss,T.loss(s,d)+.25*(hard-n))
  checks.append(dict(key=r['key'],selected_known_negatives=len(ix),unknown_gradient_zero=True))
 # Verify evaluation still calls original loss, not new training loss.
 import inspect
 assert 'train_loss(' not in inspect.getsource(T.evaluate)
 out=dict(status='pass',sample_pairs=32,source_gate_after_fps=True,initialization_sha256=init,matched_plan_epochs=32,matched_steps=192,evaluation_original_loss=True,gradient_checks=checks)
 (O/'audits').mkdir(exist_ok=True);(O/'audits/preflight.json').write_text(json.dumps(out,indent=2));print('PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
