"""Train a first-pose object PointNet++ student on train-only cross-demonstration frequency cues.
No simulator contacts, object poses, meshes, names, actions, or manual region labels enter training."""
from pathlib import Path
import json,random,sys,time
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pointnet2_grasp import PointNet2Grasp
R=Path('/mnt/nvme1/drh/D_VLA')
DATA=R/'data/v3/samples/frequency_student_v1'
GRAPH_DATA=R/'data/v3/samples/visual_interaction_v1'
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
def load(split):
 items=[]
 for id in S[split]:
  path=DATA/split/(id+'.npz')
  graph=GRAPH_DATA/'graphs'/split/(id+'.npz')
  with np.load(path) as f:sample={k:f[k].copy() for k in ('xyz','dino_feat','target')}
  with np.load(graph) as f:g={k:f[k].copy() for k in f.files}
  assert sample['xyz'].shape[0]==sample['target'].shape[0]
  items.append((id,sample,g))
 return items
def tensor(item,device):
 _,s,g=item
 xyz=torch.as_tensor(s['xyz'],dtype=torch.float32,device=device)
 feat=torch.as_tensor(s['dino_feat'],dtype=torch.float32,device=device)
 target=torch.as_tensor(s['target'],dtype=torch.float32,device=device)
 graph={k:torch.as_tensor(v,dtype=torch.float32 if k.endswith('weight') else torch.long,device=device) for k,v in g.items()}
 return xyz,feat,target,graph
def loss(logits,target):
 pred=logits.sigmoid()
 mse=F.mse_loss(pred,target)
 hi=torch.topk(target,min(64,len(target)//5)).indices
 lo=torch.topk(-target,min(64,len(target)//5)).indices
 rank=F.softplus(logits[lo][None,:]-logits[hi][:,None]).mean()
 return mse+.1*rank,mse,rank
def evaluate(net,items,device):
 net.eval();rows=[]
 with torch.inference_mode():
  for item in items:
   xyz,feat,target,graph=tensor(item,device)
   logits=net(xyz,feat,graph)
   total,mse,rank=loss(logits,target)
   n=max(1,int(np.ceil(.2*len(target))))
   teacher=torch.topk(target,n).indices
   student=torch.topk(logits,n).indices
   overlap=torch.isin(teacher,student).float().mean()
   rows.append(dict(id=item[0],loss=float(total),mse=float(mse),ranking=float(rank),top20_overlap=float(overlap)))
 return dict(loss=float(np.mean([x['loss'] for x in rows])),mse=float(np.mean([x['mse'] for x in rows])),top20_overlap=float(np.mean([x['top20_overlap'] for x in rows])),rows=rows)
def main():
 torch.set_num_threads(4)
 seed=2028;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 assert not(set(S['train'])&set(S['val']) or set(S['train'])&set(S['test']) or set(S['val'])&set(S['test']))
 train=load('train');val=load('val')
 out=R/'outputs/v3_stage2/frequency_student/seed_2028'
 out.mkdir(parents=True,exist_ok=True)
 if (out/'done.json').exists():print(json.dumps(dict(reused_completed=True)),flush=True);return
 if (out/'best.pt').exists():raise RuntimeError('Partial checkpoint exists; inspect before restart')
 config=dict(model='PointNet++ single-object pointwise student',input='first-pose RGB-D cloud and frozen DINOv2 RGB patch feature; no robot, future frames or full test sequence at inference',target='leave-one-out visual frequency from five visually retrieved TRAIN demonstrations; no query future',train_episodes=len(train),val_episodes=len(val),test_episodes=len(S['test']),seed=seed,epochs=8,no_contact_training=True,no_simulator_truth=True,checkpoint='minimum validation pseudo-target loss; independent contact diagnostic separate',scope='same observed assets, training-demo visual frequency pilot; not multi-operation or unseen-asset validated')
 (out/'config.json').write_text(json.dumps(config,indent=2))
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 net=PointNet2Grasp().to(device)
 opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4)
 history=[];best=float('inf');start=time.time()
 for epoch in range(1,9):
  net.train();order=train.copy();random.shuffle(order);losses=[]
  for item in order:
   xyz,feat,target,graph=tensor(item,device)
   opt.zero_grad(set_to_none=True)
   logits=net(xyz,feat,graph)
   total,_,_=loss(logits,target)
   total.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
   opt.step();losses.append(float(total.detach()))
  valresult=evaluate(net,val,device)
  row=dict(epoch=epoch,train_loss=float(np.mean(losses)),val_loss=valresult['loss'],val_top20_overlap=valresult['top20_overlap'],elapsed_s=round(time.time()-start,1))
  history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2))
  print(json.dumps(row),flush=True)
  if valresult['loss']<best:
   best=valresult['loss'];tmp=out/'best.tmp.pt'
   torch.save(dict(state=net.state_dict(),config=config,epoch=epoch,val_pseudo=valresult),tmp)
   tmp.replace(out/'best.pt')
 done=dict(event='training_complete',best_epoch=min(history,key=lambda x:x['val_loss'])['epoch'],best_val_pseudo_loss=best)
 (out/'done.json').write_text(json.dumps(done,indent=2))
 print(json.dumps(done),flush=True)
if __name__=='__main__':main()
