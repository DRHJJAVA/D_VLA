"""Frozen high-resolution DINOv2 features from automatic object crops of every RGB-D frame.
Only RGB-D and name-free proposal boxes are read. Original demonstrations remain untouched."""
from pathlib import Path
import json,time,math
import h5py,numpy as np,torch,torch.nn.functional as F
from stage2_extract_dino import R,model,preproc
from stage2_local_teacher import crop_box
OUT=R/'data/v3/features/dinov2_object_crops'
def extract(entry,net):
 source=Path(entry['rgbd_path']);n=int(entry['frames'])
 proposal=R/'data/v3/proposals/motion_rgbd_rerank'/entry['asset_group']/(source.stem+'.json')
 q=json.loads(proposal.read_text())
 assert q['status']=='complete' and not q['uses_object_name'] and not q['uses_simulator_truth'] and not q['uses_region_labels']
 target=OUT/entry['asset_group']/(source.stem+'.h5');target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  with h5py.File(target,'r') as f:
   assert f.attrs['status']=='complete' and f.attrs['source_rgbd']==str(source) and int(f.attrs['frames'])==n
   assert f['patch'].shape==(n,32,32,384) and f['depth_grid'].shape==(n,32,32)
  print(json.dumps(dict(id=entry['id'],status='reused',frames=n)),flush=True);return
 tmp=target.with_suffix('.partial.h5')
 if tmp.exists():raise RuntimeError('partial output exists; inspect before resuming: '+str(tmp))
 with h5py.File(source,'r') as f:
  h,w=f['agentview/rgb'].shape[1:3]
  cb=crop_box(q['selected']['box'],h,w)
  x1=max(0,int(math.floor(cb[0])));y1=max(0,int(math.floor(cb[1])))
  x2=min(w,int(math.ceil(cb[2])));y2=min(h,int(math.ceil(cb[3])))
  assert x2-x1>=32 and y2-y1>=32
  start=time.time()
  with h5py.File(tmp,'w') as out:
   out.attrs.update(dict(status='in_progress',source_rgbd=str(source),frames=n,crop_box=json.dumps([x1,y1,x2,y2]),proposal=str(proposal),all_original_frames=True,no_object_names=True,no_region_labels=True,no_simulator_truth=True,weight='dinov2_vits14_pretrain.pth',input='original RGB crop resized to 448x448, not interpolation of full-frame feature grid'))
   patch=out.create_dataset('patch',shape=(n,32,32,384),dtype='float16',chunks=(1,32,32,384),compression='lzf')
   depthgrid=out.create_dataset('depth_grid',shape=(n,32,32),dtype='float16',chunks=(16,32,32),compression='lzf')
   for i in range(0,n,2):
    j=min(i+2,n)
    rgb=f['agentview/rgb'][i:j,y1:y2,x1:x2]
    dep=f['agentview/depth_m'][i:j,y1:y2,x1:x2].astype(np.float32)
    with torch.inference_mode():
     tokens=net.forward_features(preproc(rgb))
     feature=tokens[:,1:,:].reshape(j-i,32,32,384)
     d=F.interpolate(torch.from_numpy(dep).cuda()[:,None],size=(32,32),mode='area')[:,0]
    assert torch.isfinite(feature).all() and torch.isfinite(d).all()
    patch[i:j]=feature.cpu().numpy().astype(np.float16)
    depthgrid[i:j]=d.cpu().numpy().astype(np.float16)
   out.attrs['status']='complete';out.flush()
 tmp.replace(target)
 print(json.dumps(dict(id=entry['id'],status='complete',frames=n,crop=[x1,y1,x2,y2],elapsed_s=round(time.time()-start,1))),flush=True)
def main():
 torch.set_num_threads(4)
 entries=[json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]
 net=model()
 for e in entries:extract(e,net)
 print(json.dumps(dict(event='all_highres_crops_complete',episodes=len(entries))),flush=True)
if __name__=='__main__':main()
