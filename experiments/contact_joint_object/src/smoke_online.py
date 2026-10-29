from pathlib import Path
import torch,json,time,traceback,numpy as np,h5py
from data import B,Dataset,to_device
from network import JointModel,joint_loss
from online_targets import refresh_episode,target_for
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25);torch.manual_seed(2026)
try:
 ds=Dataset('train');r=ds.rows[0];m=JointModel().cuda().eval();t=time.time()
 exact=refresh_episode(m,ds,r,-1,point_limit=128,prune=False)
 result=refresh_episode(m,ds,r,0,point_limit=128,prune=True)
 raw=ds.get(r,47,ds.frame_objects(r,47)[0]);b=to_device(raw);target=target_for(raw,0)
 assert target is not None
 assert torch.all(target['low']<=target['high']) and torch.all(target['low']>=0) and torch.all(target['high']<=1+1e-6)
 current={k:b[k] for k in ['xyz','normal','features','graph','finger_nodes','finger_normals','finger_side']}
 with torch.no_grad():
  score=m.score_current(current);full=m(b)['score'];assert torch.equal(score['score'],full)
 loss,v,o=joint_loss(m,b,target);assert torch.isfinite(loss);loss.backward()
 baseline=Path(exact['path'])
 diff=0.
 with h5py.File(baseline,'r') as old,h5py.File(result['path'],'r') as new:
  for tt in old:
   for oid in old[tt]:
    for key in ['low','high']:
     diff=max(diff,float(np.max(np.abs(old[tt+'/'+oid+'/'+key][:]-new[tt+'/'+oid+'/'+key][:]))))
 assert diff<=3.1e-6
 result.update(ok=True,numerical_pruning_target_max_difference=diff,seconds=time.time()-t,intervals_ordered=True,current_entry_requires_no_future_fields=True,actual_joint_backward_finite=all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None),target_scope='engineering random-initialization test, not trained results')
except Exception as e:
 (B/'logs/online_smoke_traceback.log').write_text(traceback.format_exc());result={'ok':False,'error':type(e).__name__,'detail':str(e)[:250]}
(B/'manifests/online_smoke.json').write_text(json.dumps(result,indent=2))
