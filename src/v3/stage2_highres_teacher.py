"""Full-sequence cross-view teacher using actual high-resolution object-crop DINOv2 features.
No names, text, actions, simulator contacts, or region labels in training."""
from pathlib import Path
import json,random,time
import h5py,numpy as np,torch
from stage2_crossview_teacher import R,FEATURES,Teacher
SPLIT=R/'data/v3/manifests/stage2_episode_split.json'
MANIFEST=R/'data/v3/manifests/original_sequences.jsonl'
CROPS=R/'data/v3/features/dinov2_object_crops'
OUTPUT=R/'outputs/v3_stage2/A_highres/seed_2026'
def load(entry,device):
 source=Path(entry['rgbd_path'])
 crop=CROPS/entry['asset_group']/(source.stem+'.h5')
 full=FEATURES/entry['asset_group']/(source.stem+'.h5')
 with h5py.File(crop,'r') as f:
  assert f.attrs['status']=='complete' and int(f.attrs['frames'])==entry['frames']
  fixed=torch.from_numpy(f['patch'][:].astype(np.float32)).to(device)
  df=torch.from_numpy(f['depth_grid'][:].astype(np.float32)).to(device)
  box=json.loads(f.attrs['crop_box']);source_box=json.loads(Path(f.attrs['proposal']).read_text())['selected']['box']
 with h5py.File(full,'r') as f:
  wrist=torch.from_numpy(f['robot0_eye_in_hand/patch'][:].astype(np.float32)).to(device)
  dw=torch.from_numpy(f['robot0_eye_in_hand/depth_grid'][:].astype(np.float32)).to(device)
 assert fixed.shape==(entry['frames'],32,32,384)
 return (fixed,wrist,df,dw,None),dict(crop=box,source_box=source_box)
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
 config=dict(variant='A_highres',seed=2026,epochs=8,train_count=len(train),val_count=len(val),source='all original RGB-D frames; frozen DINOv2 rerun on raw object crop; global wrist DINOv2',loss='synchronized cross-view contrast',crop='raw object RGB pixels cropped per frame using a fixed name-free first-pose box, then encoded at 448x448; all frames retained',no_actions=True,no_contact_labels=True,no_simulator_truth=True,map_meaning='uncalibrated interaction attention, not grasp probability')
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
