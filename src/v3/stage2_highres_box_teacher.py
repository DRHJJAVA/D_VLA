"""Full-sequence object-box-attention teacher; automatic box is not a grasp-region label."""
from pathlib import Path
import json,random,time
import h5py,numpy as np,torch
from torch.nn import functional as F
from stage2_crossview_teacher import R,FEATURES,Teacher
from stage2_highres_teacher import load as base_load, SPLIT, MANIFEST
OUTPUT=R/'outputs/v3_stage2/A_highres_box/seed_2026'
class BoxTeacher(Teacher):
 def __init__(self):
  super().__init__('A')
 def forward(self,fixed,wrist,depth_fixed,depth_wrist,prior,mask):
  n=fixed.shape[0]
  f=torch.cat((fixed.reshape(n,1024,384).float(),depth_fixed.reshape(n,1024,1).float()),-1)
  w=torch.cat((wrist[:,12:20,12:20].float().mean((1,2)),depth_wrist[:,12:20,12:20].float().mean((1,2))[:,None]),-1)
  q=self.query(w)
  k=F.normalize(self.key(f),dim=-1)
  qn=F.normalize(q,dim=-1)
  logits=20*torch.einsum('nd,npd->np',qn,k)
  logits=logits.masked_fill(~mask.reshape(1,1024),-1e4)
  attn=logits.softmax(-1)
  value=self.value(f)
  context=F.normalize(torch.einsum('np,npd->nd',attn,value),dim=-1)
  pair=20*context@qn.T
  label=torch.arange(n,device=q.device)
  loss=.5*(F.cross_entropy(pair,label)+F.cross_entropy(pair.T,label))
  return loss,attn.reshape(n,32,32)
def load(entry,device):
 data,meta=base_load(entry,device)
 x1,y1,x2,y2=meta['crop']
 bx1,by1,bx2,by2=meta['source_box']
 xx=x1+(np.arange(32)+.5)*(x2-x1)/32
 yy=y1+(np.arange(32)+.5)*(y2-y1)/32
 mask=(yy[:,None]>=by1-2)&(yy[:,None]<by2+2)&(xx[None,:]>=bx1-2)&(xx[None,:]<bx2+2)
 assert mask.sum()>=4
 return (*data,torch.from_numpy(mask).to(device)),meta
def eval_loss(net,entries,device):
 net.eval();values=[]
 with torch.no_grad():
  for entry in entries:
   data,_=load(entry,device);loss,_=net(*data);values.append(float(loss))
 return float(np.mean(values))
def main():
 torch.set_num_threads(4);torch.manual_seed(2026);np.random.seed(2026);random.seed(2026)
 rows={x['id']:x for x in [json.loads(x) for x in MANIFEST.read_text().splitlines() if x]}
 split=json.loads(SPLIT.read_text())['splits'];assert set(split['train']).isdisjoint(split['val']) and set(split['test']).isdisjoint(set(split['train'])|set(split['val']))
 train=[rows[x] for x in split['train']];val=[rows[x] for x in split['val']]
 OUTPUT.mkdir(parents=True,exist_ok=True)
 if (OUTPUT/'done.json').exists():print(json.dumps(dict(event='reused_completed')),flush=True);return
 if (OUTPUT/'best.pt').exists() or (OUTPUT/'history.json').exists():raise RuntimeError('partial run; inspect before restart')
 config=dict(variant='A_highres_box',seed=2026,epochs=8,train_count=len(train),val_count=len(val),source='all original RGB-D frames; frozen DINOv2 rerun on raw object crop; global wrist DINOv2',loss='synchronized cross-view contrast',crop='raw object RGB pixels encoded at 448x448; logits restricted to fixed name-free first-pose object candidate box expanded by 2px; all frames retained',no_actions=True,no_contact_labels=True,no_simulator_truth=True,map_meaning='uncalibrated within-automatic-object-box attention, not grasp probability')
 (OUTPUT/'config.json').write_text(json.dumps(config,indent=2))
 device='cuda';net=BoxTeacher().to(device);opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4)
 history=[];best=float('inf');start=time.time()
 for epoch in range(1,9):
  net.train();order=train.copy();random.shuffle(order);losses=[]
  for entry in order:
   data,_=load(entry,device)
   opt.zero_grad(set_to_none=True);loss,_=net(*data);loss.backward()
   torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
   losses.append(float(loss.detach()));del data,loss
  v=eval_loss(net,val,device)
  row=dict(epoch=epoch,train_loss=float(np.mean(losses)),val_loss=v,elapsed_s=round(time.time()-start,1))
  history.append(row);(OUTPUT/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
  if v<best:
   best=v;tmp=OUTPUT/'best.tmp.pt';torch.save(dict(state=net.state_dict(),config=config,epoch=epoch,val_loss=v),tmp);tmp.replace(OUTPUT/'best.pt')
 done=dict(event='training_complete',best_val=best,best_epoch=min(history,key=lambda x:x['val_loss'])['epoch'])
 (OUTPUT/'done.json').write_text(json.dumps(done,indent=2));print(json.dumps(done),flush=True)
if __name__=='__main__':main()
