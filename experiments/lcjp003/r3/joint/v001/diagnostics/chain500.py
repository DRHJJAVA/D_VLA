import os,sys,pathlib,json,traceback,time
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import torch,numpy as np
torch.set_num_threads(2)
N=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(N/'src'))
import train
from network import JointNetwork
sys.path.insert(0,str(N.parents[2]/'r2/v2/v001/src'))
# Explicit existing R2 helper path, read only.
sys.path.insert(0,'/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r2/v2/v001/src')
import common
try:
 m=JointNetwork(common.build_expert('cpu').lm_expert);m.load_state_dict(torch.load(N/'checkpoints/step0500.pt',map_location='cpu',weights_only=False)['model']);m.eval()
 episodes={e['id']:e for e in json.loads((N/'episodes.json').read_text())}
 records=json.loads((N/'eval0500.json').read_text())['records']
 a=json.loads((N/'eval0250.json').read_text())['records']
 indices=sorted(range(80),key=lambda i:records[i]['gripper_error']-a[i]['gripper_error'],reverse=True)[:4]+[0,4,40,44]
 indices=list(dict.fromkeys(indices));out=[]
 for i in indices:
  r=records[i];d,y=train.load_window(episodes[r['episode']],r['frame'])
  noise=torch.randn(1,16,6,generator=torch.Generator().manual_seed(20261028+i))
  with torch.no_grad():
   e=m.encode(d);o=m.generate(e,noise)
  if not out:
   (N/'diagnostics/schema500.json').write_text(json.dumps({'encode':{k:list(v.shape) for k,v in e.items() if hasattr(v,'shape')},'generate_type':type(o).__name__,'generate_shapes':{k:list(v.shape) for k,v in o.items()} if isinstance(o,dict) else [list(v.shape) for v in o]}))
  pred=int(e['roles'].argmax());teacher=int(y['role'].argmax());valid=d['valid']
  variants={}
  def calc(en,**kw):
   with torch.no_grad(): z=m.generate(en,noise,**kw)
   motion,grip=z
   probs=grip.softmax(-1);labels=probs.argmax(-1)
   return {'mae':float((motion[0]-y['target']).abs().mean()),'grip_error':float((labels[0]!=y['grip']).float().mean()),'grip_prediction':labels[0].tolist(),'grip_probability':probs[0,:,1].tolist()},motion
  base,act=calc(e);variants['normal']=base
  for name in ['uniform','shuffle']:
   with torch.no_grad():ee=m.encode(d,name)
   z,aa=calc(ee);z['action_delta']=float((aa-act).abs().mean())
   if pred<len(valid):z['top_overlap']=len(set(e['top'][pred].tolist()) & set(ee['top'][pred].tolist()))/8
   variants[name]=z
  for name,value in [('lambda0',0.),('lambda1',1.)]:
   z,aa=calc(e,lambda_override=value);z['action_delta']=float((aa-act).abs().mean());variants[name]=z
  z,aa=calc(e,object_index=teacher);z['action_delta']=float((aa-act).abs().mean());variants['teacher_target']=z
  al=e['alpha'];q=y['parts'];k=min(teacher,len(valid)-1);v=valid[k];p=al[k][v];t=q[k][v]
  entropy=float(-(p*p.clamp_min(1e-12).log()).sum()/np.log(max(len(p),2)))
  teacher_entropy=float(-(t*t.clamp_min(1e-12).log()).sum()/np.log(max(len(t),2)))
  mass=float(t[torch.topk(p,max(1,int(.2*len(p)))).indices].sum())
  out.append(dict(index=i,episode=r['episode'],frame=r['frame'],pred_target=pred,teacher_target=teacher,role_weight=float(y['role_weight']),part_weight=float(y['part_weight'][k]),effective_part_weight=float(y['role_weight']*y['part_weight'][k]),target_agrees=pred==teacher,score_entropy=entropy,teacher_entropy=teacher_entropy,teacher_mass_in_pred_top20=mass,grip_truth=y['grip'].tolist(),variants=variants))
  (N/'diagnostics/chain500_progress.json').write_text(json.dumps({'completed':len(out),'total':len(indices),'records':out},indent=2))
 (N/'diagnostics/chain500_complete.json').write_text(json.dumps({'cpu_only':True,'checkpoint':500,'records':out},indent=2))
except Exception as ex:
 (N/'diagnostics/chain500_failure.json').write_text(json.dumps({'error_type':type(ex).__name__,'message':str(ex)[:200],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(ex.__traceback__)]}))
