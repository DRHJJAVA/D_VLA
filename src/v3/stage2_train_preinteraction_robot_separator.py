"""Auxiliary self-supervised RGB-D moving-manipulator separator.
Pseudo masks come only from preinteraction optical flow in complete TRAIN RGB-D.
VAL episodes choose checkpoint using pseudo-mask consistency, not grasp contacts.
This module is not a grasp surface model or Stage 2 acceptance result.
"""
from pathlib import Path
import hashlib,json,random,time
import cv2,h5py,numpy as np,torch
import torch.nn as nn
import torch.nn.functional as F
R=Path('/mnt/nvme1/drh/D_VLA')
SRC=R/'data/v3/samples/preinteraction_robot_motion_pseudo_v1'
COMP=R/'data/v3/proposals/foreground3d_components'
OUT=R/'outputs/v3_stage2/preinteraction_robot_seg_v1'
EPOCHS=12
SEED=2026
SIZE=256
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def block(a,b):
 return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(inplace=True),
  nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(inplace=True))
class SmallUNet(nn.Module):
 def __init__(self):
  super().__init__()
  self.e1=block(4,24);self.e2=block(24,48);self.e3=block(48,96)
  self.mid=block(96,192)
  self.d3=block(192+96,96);self.d2=block(96+48,48);self.d1=block(48+24,24)
  self.out=nn.Conv2d(24,1,1)
 def forward(self,x):
  a=self.e1(x);b=self.e2(F.max_pool2d(a,2));c=self.e3(F.max_pool2d(b,2))
  m=self.mid(F.max_pool2d(c,2))
  y=self.d3(torch.cat([F.interpolate(m,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
  y=self.d2(torch.cat([F.interpolate(y,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
  y=self.d1(torch.cat([F.interpolate(y,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
  return self.out(y)
def load_split(split,rows):
 paths={}
 data=[]
 for row in rows:
  if row['split']!=split or row['status']!='usable':continue
  id=row['id'];t=row['frame']
  p=SRC/split/(id+'_'+str(t).zfill(3)+'.npz')
  with np.load(p) as z:
   rgb=z['rgb'].copy()
   pos=z['positive_core'].copy()
   neg=z['negative_far'].copy()
   mov=z['moving_component'].copy()
   assert str(z['episode_id'])==id and int(z['frame'])==t
  if id not in paths:
   with np.load(COMP/split/(id+'.npz')) as z:paths[id]=Path(str(z['rgbd_path']))
  with h5py.File(paths[id]) as f:depth=f['agentview/depth_m'][t]
  x=cv2.resize(rgb,(SIZE,SIZE),interpolation=cv2.INTER_AREA).astype(np.float32)/255
  z=cv2.resize(depth,(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
  x=np.concatenate([x,np.clip(z,0,2)[...,None]/2],axis=2)
  pos=cv2.resize(pos,(SIZE,SIZE),interpolation=cv2.INTER_NEAREST).astype(np.float32)
  neg=cv2.resize(neg,(SIZE,SIZE),interpolation=cv2.INTER_NEAREST).astype(np.float32)
  mov=cv2.resize(mov,(SIZE,SIZE),interpolation=cv2.INTER_NEAREST).astype(np.float32)
  assert pos.sum()>100 and neg.sum()>100
  data.append((id,t,x.transpose(2,0,1),pos[None],neg[None],mov[None]))
 return data
def batch_loss(logits,pos,neg):
 return (F.softplus(-logits)*pos).sum()/pos.sum().clamp_min(1)+(F.softplus(logits)*neg).sum()/neg.sum().clamp_min(1)
def eval_model(net,data,device):
 net.eval();losses=[];ious=[]
 with torch.inference_mode():
  for start in range(0,len(data),8):
   part=data[start:start+8]
   x=torch.from_numpy(np.stack([v[2] for v in part])).to(device)
   pos=torch.from_numpy(np.stack([v[3] for v in part])).to(device)
   neg=torch.from_numpy(np.stack([v[4] for v in part])).to(device)
   mov=torch.from_numpy(np.stack([v[5] for v in part])).to(device)
   pred=net(x);losses.append(float(batch_loss(pred,pos,neg)))
   binary=pred.sigmoid()>.5
   iou=((binary&mov.bool()).sum((1,2,3)).float()/(binary|mov.bool()).sum((1,2,3)).clamp_min(1)).cpu().numpy()
   ious.extend(map(float,iou))
 return float(np.mean(losses)),float(np.mean(ious))
def main():
 manifest=SRC/'manifest.json';j=json.loads(manifest.read_text())
 assert j['status']=='preinteraction_robot_motion_pseudo_v1_complete'
 assert j['no_simulator_truth'] and j['no_actions'] and j['no_manual_labels']
 train=load_split('train',j['rows']);val=load_split('val',j['rows'])
 assert len(train)==j['counts']['train']['usable'] and len(val)==j['counts']['val']['usable']
 OUT.mkdir(parents=True,exist_ok=True)
 done=OUT/'summary.json'
 if done.exists():
  print(json.dumps(dict(status='already_complete',summary=str(done))),flush=True);return
 torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
 device='cuda' if torch.cuda.is_available() else 'cpu'
 net=SmallUNet().to(device);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
 checkpoints=sorted(OUT.glob('epoch_*.pt'))
 history=[]
 if checkpoints:
  last=checkpoints[-1];state=torch.load(last,map_location=device,weights_only=False)
  assert state['manifest_sha256']==sha(manifest)
  net.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
  history=state['history'];start_epoch=state['epoch']+1
 else:start_epoch=1
 for epoch in range(start_epoch,EPOCHS+1):
  net.train();order=np.random.default_rng(SEED+epoch).permutation(len(train))
  losses=[]
  for start in range(0,len(order),8):
   part=[train[k] for k in order[start:start+8]]
   x=torch.from_numpy(np.stack([v[2] for v in part])).to(device)
   pos=torch.from_numpy(np.stack([v[3] for v in part])).to(device)
   neg=torch.from_numpy(np.stack([v[4] for v in part])).to(device)
   if np.random.default_rng(SEED+epoch*1000+start).random()<.5:
    x=torch.flip(x,[-1]);pos=torch.flip(pos,[-1]);neg=torch.flip(neg,[-1])
   opt.zero_grad(set_to_none=True)
   logits=net(x);loss=batch_loss(logits,pos,neg)
   loss.backward();opt.step();losses.append(float(loss.detach()))
  vloss,viou=eval_model(net,val,device)
  row=dict(epoch=epoch,train_loss=float(np.mean(losses)),val_pseudo_loss=vloss,
   val_pseudo_iou=viou)
  history.append(row);print(json.dumps(row),flush=True)
  dest=OUT/('epoch_'+str(epoch).zfill(2)+'.pt')
  if dest.exists():raise FileExistsError(dest)
  temp=dest.with_suffix('.partial')
  if temp.exists():raise RuntimeError(str(temp))
  torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
   history=history,manifest_sha256=sha(manifest),
   no_simulator_truth=True,no_actions=True),temp)
  temp.replace(dest)
 best=min(history,key=lambda x:x['val_pseudo_loss'])
 record=dict(status='auxiliary_robot_separator_pseudo_training_complete',
  manifest_sha256=sha(manifest),train_samples=len(train),val_samples=len(val),
  epochs=EPOCHS,best_epoch=best['epoch'],best_pseudo_val_loss=best['val_pseudo_loss'],
  best_pseudo_val_iou=best['val_pseudo_iou'],
  checkpoint=str(OUT/('epoch_'+str(best['epoch']).zfill(2)+'.pt')),
  caveat='pseudo-mask consistency is not robot segmentation truth or grasp-region accuracy; inspect contact-frame predictions separately',
  no_simulator_truth=True,no_actions=True,no_names=True,history=history)
 done.write_text(json.dumps(record,indent=2))
 print(json.dumps(dict(status='complete',best_epoch=best['epoch'])),flush=True)
if __name__=='__main__':main()
