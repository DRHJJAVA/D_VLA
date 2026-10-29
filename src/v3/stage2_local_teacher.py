"""Full-sequence cross-view teacher on automatically localized RGB-D object crop.
Pilot only: its attention is an interaction cue, never a calibrated grasp probability.
No object names, text, actions, simulator contacts, or region labels in training."""
from pathlib import Path
import json,math,random,time
import h5py,numpy as np,torch
from torch.nn import functional as F
from stage2_crossview_teacher import R,FEATURES,Teacher
SPLIT=R/'data/v3/manifests/stage2_episode_split.json'
MANIFEST=R/'data/v3/manifests/original_sequences.jsonl'
OUTPUT=R/'outputs/v3_stage2/A_local/seed_2026'
def crop_box(box,height,width):
 x1,y1,x2,y2=box
 cx=(x1+x2)/2;cy=(y1+y2)/2
 side=min(min(height,width),max(64,2*max(x2-x1,y2-y1)))
 x1=max(0,min(width-side,cx-side/2));y1=max(0,min(height-side,cy-side/2))
 return (float(x1),float(y1),float(x1+side),float(y1+side))
def load(entry,device):
 source=Path(entry['rgbd_path'])
 p=R/'data/v3/proposals/motion_rgbd_rerank'/entry['asset_group']/(source.stem+'.json')
 q=json.loads(p.read_text());assert q['status']=='complete' and not q['uses_region_labels'] and not q['uses_simulator_truth']
 box=q['selected']['box']
 with h5py.File(source,'r') as f:h,w=f['agentview/rgb'].shape[1:3]
 cb=crop_box(box,h,w)
 # Frozen features cover all provided frames. The crop uses only automated image/depth localization.
 x1=max(0,int(math.floor(cb[0]*32/w)));y1=max(0,int(math.floor(cb[1]*32/h)))
 x2=min(32,int(math.ceil(cb[2]*32/w)));y2=min(32,int(math.ceil(cb[3]*32/h)))
 path=FEATURES/entry['asset_group']/(source.stem+'.h5')
 with h5py.File(path,'r') as f:
  fixed=torch.from_numpy(f['agentview/patch'][:,y1:y2,x1:x2].astype(np.float32)).to(device)
  df=torch.from_numpy(f['agentview/depth_grid'][:,y1:y2,x1:x2].astype(np.float32)).to(device)
  wrist=torch.from_numpy(f['robot0_eye_in_hand/patch'][:].astype(np.float32)).to(device)
  dw=torch.from_numpy(f['robot0_eye_in_hand/depth_grid'][:].astype(np.float32)).to(device)
 assert fixed.shape[0]==entry['frames'] and fixed.shape[1]>1 and fixed.shape[2]>1
 fixed=F.interpolate(fixed.permute(0,3,1,2),size=(32,32),mode='bilinear',align_corners=False).permute(0,2,3,1)
 df=F.interpolate(df[:,None],size=(32,32),mode='bilinear',align_corners=False)[:,0]
 return (fixed,wrist,df,dw,None),dict(source_box=box,crop=cb,feature_crop=[x1,y1,x2,y2])
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
 config=dict(variant='A_local',seed=2026,epochs=8,train_count=len(train),val_count=len(val),source='all original RGB-D frames; frozen full-frame DINOv2; name-free RGB-D object proposal crop',loss='synchronized cross-view contrast',crop='square twice selected box side, minimum 64px; existing 32x32 DINO patch grid upsampled, no new high-resolution detail',no_actions=True,no_contact_labels=True,no_simulator_truth=True,map_meaning='uncalibrated interaction attention, not grasp probability')
 (OUTPUT/'config.json').write_text(json.dumps(config,indent=2))
 device='cuda';net=Teacher('A').to(device);opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4)
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
