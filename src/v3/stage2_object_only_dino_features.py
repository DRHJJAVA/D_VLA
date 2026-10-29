"""Strict single-object cutout DINO features for train, validation and pose development.
TRAIN full RGB-D supplies separate visual target; scoring input is only the object
surface point cloud plus DINO features from an RGB crop with all non-object pixels
replaced by a neutral constant. No names, actions, contacts or simulator truth.
"""
from pathlib import Path
import hashlib,json,math,sys
import h5py,numpy as np,torch
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_extract_dino import model,preproc
SRC=R/'data/v3/samples/visual_interaction_uniform448_3dcc'
POSE=R/'data/v3/samples/poseval16_firstframe_multicandidate'
COMP=R/'data/v3/proposals/foreground3d_components'
MAN=R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json'
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
OUT=R/'data/v3/samples/object_only_dino_v1'
NEUTRAL=np.asarray([124,116,104],dtype=np.uint8)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def square(pixel,h,w):
 lo=pixel.min(0);hi=pixel.max(0)+1
 cx=float(lo[0]+hi[0])/2;cy=float(lo[1]+hi[1])/2
 side=int(math.ceil(max(64,1.25*max(hi-lo))))
 side=min(side,h,w)
 x1=int(round(np.clip(cx-side/2,0,w-side)))
 y1=int(round(np.clip(cy-side/2,0,h-side)))
 return [x1,y1,x1+side,y1+side]
def build(split,id,source,rgbd,net):
 with np.load(source) as f:
  xyz=f['xyz'].copy();pixel=f['pixel'].copy()
  original_box=f['source_box'].copy()
  target=f['target'].copy() if split in ('train','val') else None
 assert len(xyz)==len(pixel)>=64
 with h5py.File(rgbd) as f:rgb=f['agentview/rgb'][0]
 assert rgb.shape[:2]==(448,448)
 h,w=rgb.shape[:2]
 x1,y1,x2,y2=square(pixel,h,w)
 crop=np.broadcast_to(NEUTRAL,(y2-y1,x2-x1,3)).copy()
 selected=(pixel[:,0]>=x1)&(pixel[:,0]<x2)&(pixel[:,1]>=y1)&(pixel[:,1]<y2)
 assert selected.all()
 crop[pixel[:,1]-y1,pixel[:,0]-x1]=rgb[pixel[:,1],pixel[:,0]]
 with torch.inference_mode():
  patch=net.forward_features(preproc(crop[None]))[:,1:,:].reshape(32,32,384).cpu().numpy()
 gx=np.clip(((pixel[:,0]-x1)*32/max(1,x2-x1)).astype(int),0,31)
 gy=np.clip(((pixel[:,1]-y1)*32/max(1,y2-y1)).astype(int),0,31)
 feat=patch[gy,gx].astype(np.float32)
 dest=OUT/split/(id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
 tmp=dest.with_suffix('.partial.npz')
 if tmp.exists():raise RuntimeError('inspect partial: '+str(tmp))
 record=dict(xyz=xyz.astype(np.float32),pixel=pixel.astype(np.int16),
  dino_feat=feat,source_box=original_box,object_crop_box=np.asarray([x1,y1,x2,y2]),
  object_rgb=rgb[pixel[:,1],pixel[:,0]],non_object_pixels_neutral=np.asarray(True),
  source_sha256=np.asarray(sha(source)),query_frame=np.asarray(0))
 if target is not None:record['target']=target.astype(np.float32)
 np.savez_compressed(tmp,**record);tmp.replace(dest)
 return dict(id=id,split=split,points=len(pixel),crop_box=[x1,y1,x2,y2])
def main():
 assert json.loads((SRC/'manifest.json').read_text())['status']=='visual_interaction_uniform448_3dcc_complete'
 pose=json.loads(MAN.read_text())['rows'];assert len(pose)==16
 jobs=[]
 for split in ('train','val'):
  for id in S[split]:
   source=SRC/split/(id+'.npz')
   with np.load(COMP/split/(id+'.npz')) as z:rgbd=Path(str(z['rgbd_path']))
   jobs.append((split,id,source,rgbd))
 for e in pose:jobs.append(('poseval16',e['id'],POSE/e['id']/'rank_1.npz',Path(e['rgbd_path'])))
 assert len(jobs)==40
 torch.set_num_threads(4);net=None;rows=[]
 for split,id,source,rgbd in jobs:
  dest=OUT/split/(id+'.npz')
  if dest.exists():
   with np.load(dest) as old:
    assert str(old['source_sha256'])==sha(source)
    assert int(old['query_frame'])==0 and bool(old['non_object_pixels_neutral'])
    n=len(old['xyz'])
   row=dict(id=id,split=split,status='reused',points=n)
  else:
   if net is None:net=model()
   row=build(split,id,source,rgbd,net);row['status']='built'
  rows.append(row);print(json.dumps(row),flush=True)
 manifest=OUT/'manifest.json'
 record=dict(status='strict_object_only_features_complete',
  train=20,val=4,poseval16=16,query_frame=0,no_simulator_truth=True,
  no_names=True,non_object_pixels_neutral=True,source_manifest_sha256=sha(SRC/'manifest.json'),
  pose_manifest_sha256=sha(MAN),rows=rows)
 if manifest.exists():
  old=json.loads(manifest.read_text())
  assert old['pose_manifest_sha256']==record['pose_manifest_sha256']
 else:manifest.write_text(json.dumps(record,indent=2))
 print(json.dumps(dict(status='complete',samples=len(rows))),flush=True)
if __name__=='__main__':main()
