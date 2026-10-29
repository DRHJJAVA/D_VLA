from pathlib import Path
import os,json,time,traceback,sys
import numpy as np,torch
B=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object')
from data import Dataset,to_device
from geometry import closest,box_tri
from network import JointModel,joint_loss
torch.set_num_threads(4);torch.manual_seed(2026);torch.cuda.set_per_process_memory_fraction(.25)
try:
 ds=Dataset('train');r=ds.rows[0];t=47;objects=ds.frame_objects(r,t);samples=[]
 for oid in objects:
  b=ds.get(r,t,oid,ci=0);samples.append(b)
 b=max(samples,key=lambda x:len(x['motion_source']));raw=b;b=to_device(b)
 m=JointModel().cuda();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4)
 lo=torch.zeros(len(b['xyz']),device='cuda');hi=torch.ones_like(lo)
 loss,metrics,o=joint_loss(m,b,{'low':lo,'high':hi});assert torch.isfinite(loss)
 loss.backward();grad={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.grad is not None}
 assert all(np.isfinite(v) for v in grad.values());assert any(v>0 for k,v in grad.items() if k.startswith('relation'))
 opt.step();m.eval()
 with torch.no_grad():
  a=m(b)['score'].clone();b2=dict(b);b2['context']=torch.randn_like(b['context']);b2['finger_displacement']=torch.randn_like(b['finger_displacement']);c=m(b2)['score']
 assert torch.equal(a,c)
 tri=box_tri(np.array([1.,1.,1.]));d,p,n=closest(np.array([[2.,0,0],[0.,0,0],[1.,.2,.3]]),tri)
 assert np.allclose(d,[1,-1,0],atol=1e-6)
 cp=B/'runs/v001/smoke_only.pt';torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'not_formal_initialization':True},cp)
 m2=JointModel().cuda();m2.load_state_dict(torch.load(cp,map_location='cuda',weights_only=False)['model'])
 out={'ok':True,'episode':r['id'],'frame':t,'object':raw['object_id'],'points':len(raw['xyz']),'motion_observations':len(raw['motion_source']),'loss':metrics,'all_gradients_finite':True,'relation_gradient_nonzero':True,'current_score_future_context_invariant':True,'sdf_box_test':True,'trainable_parameters':sum(p.numel() for p in m.parameters()),'smoke_weights_discard_for_formal_training':True}
except Exception as e:
 (B/'logs/network_smoke_traceback.log').write_text(traceback.format_exc())
 out={'ok':False,'error':type(e).__name__,'detail':str(e)[:200]}
(B/'manifests/network_smoke.json').write_text(json.dumps(out,indent=2))
