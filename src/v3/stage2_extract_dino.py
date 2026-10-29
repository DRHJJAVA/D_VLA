"""Frozen DINOv2 features for every supplied RGB-D frame; no region labels."""
from pathlib import Path
import argparse,json,hashlib,time,os
import h5py,numpy as np,torch,torch.nn.functional as F,timm
R=Path('/mnt/nvme1/drh/D_VLA')
WEIGHTS=Path('/home/drh/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')
CAMS=('agentview','robot0_eye_in_hand')
OUT=R/'data/v3/features/dinov2_all_frames'
def model():
 net=timm.create_model('vit_small_patch14_dinov2',pretrained=False,dynamic_img_size=True)
 w=torch.load(WEIGHTS,map_location='cpu',weights_only=False)
 state=net.load_state_dict(w,strict=False)
 assert not state.missing_keys and state.unexpected_keys==['mask_token'],state
 return net.eval().cuda()
def preproc(rgb):
 x=torch.from_numpy(np.ascontiguousarray(rgb)).cuda(non_blocking=True).permute(0,3,1,2).float()/255
 if x.shape[-2:]!=(448,448):x=F.interpolate(x,size=(448,448),mode='bilinear',align_corners=False)
 mean=x.new_tensor((.485,.456,.406))[None,:,None,None]
 std=x.new_tensor((.229,.224,.225))[None,:,None,None]
 return (x-mean)/std
def extract(entry,net,probe=False):
 path=Path(entry['rgbd_path']);n=int(entry['frames'])
 target=OUT/entry['asset_group']/(path.stem+'.h5');target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists() and not probe:
  with h5py.File(target,'r') as f:
   assert f.attrs['status']=='complete' and int(f.attrs['frames'])==n and f.attrs['source_rgbd']==str(path)
   assert all(f[c+'/patch'].shape==(n,32,32,384) for c in CAMS)
  print(json.dumps(dict(id=entry['id'],status='reused',frames=n)),flush=True);return
 if probe:n=min(n,2)
 tmp=target.with_suffix('.partial.h5')
 if tmp.exists() and not probe:tmp.unlink()
 context=h5py.File(tmp,'w') if not probe else None
 start=time.time()
 try:
  with h5py.File(path,'r') as source:
   if context:
    context.attrs.update(dict(status='in_progress',source_rgbd=str(path),frames=n,weight_name=WEIGHTS.name,patch_grid='32x32',input_resize='448x448; 256 inputs upsampled',all_original_frames=True,no_labels=True))
   for cam in CAMS:
    dst=context.create_group(cam) if context else None
    feature=dst.create_dataset('patch',shape=(n,32,32,384),dtype='float16',chunks=(1,32,32,384),compression='lzf') if dst else None
    depth=dst.create_dataset('depth_grid',shape=(n,32,32),dtype='float16',chunks=(16,32,32),compression='lzf') if dst else None
    for i in range(0,n,2):
     j=min(i+2,n)
     rgb=source[cam+'/rgb'][i:j]
     dep=source[cam+'/depth_m'][i:j].astype(np.float32)
     with torch.inference_mode():
      tokens=net.forward_features(preproc(rgb))
      patches=tokens[:,1:,:].reshape(j-i,32,32,384)
      d=F.interpolate(torch.from_numpy(dep).cuda()[:,None],size=(32,32),mode='area')[:,0]
     assert torch.isfinite(patches).all() and torch.isfinite(d).all()
     if dst:
      feature[i:j]=patches.cpu().numpy().astype(np.float16)
      depth[i:j]=d.cpu().numpy().astype(np.float16)
    print(json.dumps(dict(id=entry['id'],camera=cam,frames=n,elapsed_s=round(time.time()-start,1))),flush=True)
   if context:
    context.attrs['status']='complete';context.flush()
 finally:
  if context:context.close()
 if not probe:tmp.replace(target)
 print(json.dumps(dict(id=entry['id'],status='probe_passed' if probe else 'complete',frames=n,seconds=round(time.time()-start,1))),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');p.add_argument('--case');a=p.parse_args()
 torch.set_num_threads(4)
 rows=[json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]
 if a.case:rows=[x for x in rows if x['id']==a.case]
 if a.probe:rows=rows[:1]
 assert rows
 net=model()
 for entry in rows:extract(entry,net,a.probe)
if __name__=='__main__':main()
