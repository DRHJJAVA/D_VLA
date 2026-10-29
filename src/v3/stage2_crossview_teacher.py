"""Stage-2 action-free cross-view context teacher. Every frame in each source sequence participates.
The attention map is a candidate interaction cue, not a calibrated grasp probability."""
from pathlib import Path
import argparse,json,math,random,time
import h5py,numpy as np,torch
from torch import nn
from torch.nn import functional as F
R=Path('/mnt/nvme1/drh/D_VLA')
FEATURES=R/'data/v3/features/dinov2_all_frames'
class Teacher(nn.Module):
 def __init__(self,variant):
  super().__init__();self.variant=variant;self.dim=256
  self.key=nn.Sequential(nn.Linear(385,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,256))
  self.value=nn.Linear(385,256)
  self.query=nn.Sequential(nn.Linear(385,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,256))
  if variant in ('C1','C2'):
   layer=nn.TransformerEncoderLayer(256,4,512,.1,batch_first=True,norm_first=True)
   self.temporal=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
  if variant=='C2':
   self.slot_logits=nn.Linear(256,4);self.slot_mix=nn.Linear(512,256)
 def forward(self,fixed,wrist,depth_fixed,depth_wrist,prior=None):
  # fixed/wrist [time,32,32,384]; full supplied time axis stays intact.
  n=fixed.shape[0]
  f=torch.cat((fixed.reshape(n,1024,384).float(),depth_fixed.reshape(n,1024,1).float()),-1)
  w=torch.cat((wrist[:,12:20,12:20].float().mean((1,2)),depth_wrist[:,12:20,12:20].float().mean((1,2))[:,None]),-1)
  q=self.query(w)
  if self.variant in ('C1','C2'):
   t=torch.arange(n,device=q.device,dtype=q.dtype)[:,None]
   freq=torch.exp(torch.arange(0,256,2,device=q.device,dtype=q.dtype)*(-math.log(10000)/256))
   pos=torch.zeros_like(q);pos[:,0::2]=torch.sin(t*freq);pos[:,1::2]=torch.cos(t*freq)
   q=self.temporal((q+pos)[None])[0]
  if self.variant=='C2':
   assignment=torch.softmax(self.slot_logits(q),-1)
   slots=(assignment.T@q)/(assignment.sum(0)[:,None]+1e-6)
   q=self.slot_mix(torch.cat((q,assignment@slots),-1))
  k=F.normalize(self.key(f),dim=-1)
  qn=F.normalize(q,dim=-1)
  logits=20*torch.einsum('nd,npd->np',qn,k)
  if self.variant in ('B','C1','C2'):
   assert prior is not None
   logits=logits+prior
  attn=logits.softmax(-1)
  value=self.value(f)
  context=F.normalize(torch.einsum('np,npd->nd',attn,value),dim=-1)
  pair=20*context@qn.T
  label=torch.arange(n,device=q.device)
  loss=.5*(F.cross_entropy(pair,label)+F.cross_entropy(pair.T,label))
  return loss,attn.reshape(n,32,32)
def load_case(entry,device,with_prior):
 path=FEATURES/entry['asset_group']/(Path(entry['rgbd_path']).stem+'.h5')
 with h5py.File(path,'r') as f:
  assert f.attrs['status']=='complete' and int(f.attrs['frames'])==entry['frames']
  fixed=torch.from_numpy(f['agentview/patch'][:]).to(device)
  wrist=torch.from_numpy(f['robot0_eye_in_hand/patch'][:]).to(device)
  df=torch.from_numpy(f['agentview/depth_grid'][:]).to(device)
  dw=torch.from_numpy(f['robot0_eye_in_hand/depth_grid'][:]).to(device)
 prior=None
 if with_prior:
  with h5py.File(entry['rgbd_path'],'r') as source:
   def world(cam,depth,u,v):
    k=source[cam+'/K'][:].astype(np.float32)
    t=source[cam+'/camera_to_world'][:].astype(np.float32)
    x=(u-k[:,0,2,None])*depth/k[:,0,0,None]
    y=(v-k[:,1,2,None])*depth/k[:,1,1,None]
    camxyz=np.stack((x,y,depth),-1)
    return np.einsum('nij,npj->npi',t[:,:3,:3],camxyz)+t[:,None,:3,3]
   n=entry['frames'];h,w=source['agentview/rgb'].shape[1:3]
   yy,xx=np.indices((32,32));u=((xx.ravel()+.5)*w/32).astype(np.float32);v=((yy.ravel()+.5)*h/32).astype(np.float32)
   wf=world('agentview',df.cpu().numpy().reshape(n,1024),np.broadcast_to(u,(n,1024)),np.broadcast_to(v,(n,1024)))
   h,w=source['robot0_eye_in_hand/rgb'].shape[1:3]
   uw=np.full((n,1),w/2,dtype=np.float32);vw=np.full((n,1),h/2,dtype=np.float32)
   pw=world('robot0_eye_in_hand',dw[:,12:20,12:20].float().mean((1,2)).cpu().numpy()[:,None],uw,vw)
   dist=np.linalg.norm(wf-pw,axis=-1)
   prior=torch.from_numpy(np.clip(-dist*dist/(2*.12**2),-6,0).astype(np.float32)).to(device)
 return fixed,wrist,df,dw,prior
def evaluate(net,entries,device):
 net.eval();ls=[]
 with torch.no_grad():
  for x in entries:
   data=load_case(x,device,net.variant!='A')
   loss,_=net(*data)
   ls.append(float(loss))
 return float(np.mean(ls))
def train(a):
 torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
 rows={x['id']:x for x in [json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]}
 split=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
 train_rows=[rows[x] for x in split['train']];val_rows=[rows[x] for x in split['val']]
 assert not set(split['train'])&set(split['val']) and not set(split['test'])&(set(split['train'])|set(split['val']))
 out=R/'outputs/v3_stage2'/a.variant/f'seed_{a.seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'done.json').exists():
  print(json.dumps(dict(event='reused_completed',variant=a.variant,seed=a.seed)),flush=True);return
 if (out/'best.pt').exists() or (out/'history.json').exists():
  raise RuntimeError('Incomplete run already has a checkpoint/history; inspect before restarting: '+str(out))
 config=dict(variant=a.variant,seed=a.seed,epochs=a.epochs,split_file=str(R/'data/v3/manifests/stage2_episode_split.json'),train_count=len(train_rows),val_count=len(val_rows),input='all frames of both RGB-D views via frozen DINOv2; no actions/contact/region labels',loss='within-episode synchronized cross-view contrast',map_meaning='candidate cross-view attention, not grasp probability',model='256-d cross-view teacher; C1/C2 use 4-layer temporal Transformer')
 (out/'config.json').write_text(json.dumps(config,indent=2))
 device=torch.device('cuda')
 net=Teacher(a.variant).to(device);opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4)
 best=float('inf');history=[];start=time.time()
 for epoch in range(1,a.epochs+1):
  net.train();order=train_rows.copy();random.shuffle(order);losses=[]
  for i,row in enumerate(order):
   data=load_case(row,device,a.variant!='A')
   opt.zero_grad(set_to_none=True);loss,_=net(*data);loss.backward()
   torch.nn.utils.clip_grad_norm_(net.parameters(),1)
   opt.step();losses.append(float(loss.detach()))
   del data,loss
  val=evaluate(net,val_rows,device)
  item=dict(epoch=epoch,train_loss=float(np.mean(losses)),val_loss=val,elapsed_s=round(time.time()-start,1))
  history.append(item);(out/'history.json').write_text(json.dumps(history,indent=2))
  print(json.dumps(item),flush=True)
  if val<best:
   best=val;tmp=out/'best.tmp.pt'
   torch.save(dict(state=net.state_dict(),config=config,epoch=epoch,val_loss=val),tmp);tmp.replace(out/'best.pt')
 complete=dict(event='training_complete',variant=a.variant,seed=a.seed,best_val=best,best_epoch=min(history,key=lambda z:z['val_loss'])['epoch'])
 (out/'done.json').write_text(json.dumps(complete,indent=2))
 print(json.dumps(complete),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['A','B','C1','C2'],required=True);p.add_argument('--seed',type=int,default=2026);p.add_argument('--epochs',type=int,default=8);a=p.parse_args();train(a)
if __name__=='__main__':main()
