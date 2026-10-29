"""Frozen DINO point features for up to three first-frame RGB-D pose-validation surfaces.
The query pipeline reads frame zero only. Complete query sequence is never read here.
"""
from pathlib import Path
import hashlib,json,math,sys
import h5py,numpy as np,torch
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_extract_dino import model,preproc
from stage2_local_teacher import crop_box
MAN=R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json'
SURF=R/'data/v3/proposals/poseval16_surface_candidates'
OUT=R/'data/v3/samples/poseval16_firstframe_multicandidate'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(e,rank,net):
 id=e['id'];source=SURF/id/(f'rank_{rank}.npz')
 with np.load(source) as f:
  pixel=f['pixel'].copy();xyz=f['xyz'].copy()
  outer=f['outer_box'].copy();box=f['component_box'].copy()
 with h5py.File(e['rgbd_path']) as f:
  rgb=f['agentview/rgb'][0]
 assert len(pixel)==len(xyz)>=64
 cb=crop_box(outer.tolist(),*rgb.shape[:2])
 x1,y1,x2,y2=int(math.floor(cb[0])),int(math.floor(cb[1])),int(math.ceil(cb[2])),int(math.ceil(cb[3]))
 with torch.inference_mode():
  patch=net.forward_features(preproc(rgb[None,y1:y2,x1:x2]))[:,1:,:].reshape(32,32,384).cpu().numpy()
 gx=np.clip(((pixel[:,0]-x1)*32/max(1,x2-x1)).astype(int),0,31)
 gy=np.clip(((pixel[:,1]-y1)*32/max(1,y2-y1)).astype(int),0,31)
 feat=patch[gy,gx].astype(np.float32)
 dest=OUT/id/(f'rank_{rank}.npz');dest.parent.mkdir(parents=True,exist_ok=True)
 tmp=dest.with_suffix('.partial.npz')
 if tmp.exists():raise RuntimeError('inspect partial: '+str(tmp))
 np.savez_compressed(tmp,xyz=xyz.astype(np.float32),dino_feat=feat,
   pixel=pixel.astype(np.int16),rgb=rgb[pixel[:,1],pixel[:,0]],
   source_box=box,outer_box=outer,source_sha256=np.asarray(sha(source)),
   query_frame=np.asarray(0),no_query_future=np.asarray(True))
 tmp.replace(dest)
 return len(pixel)
def main():
 entries=json.loads(MAN.read_text())['rows']
 assert len(entries)==16
 torch.set_num_threads(4);net=None;rows=[]
 for e in entries:
  id=e['id']
  for rank in (1,2,3):
   source=SURF/id/(f'rank_{rank}.npz')
   if not source.exists():
    rows.append(dict(id=id,rank=rank,status='no_surface'));continue
   dest=OUT/id/(f'rank_{rank}.npz')
   if dest.exists():
    with np.load(dest) as f:
     assert str(f['source_sha256'])==sha(source)
     assert int(f['query_frame'])==0
     n=len(f['xyz'])
    status='reused'
   else:
    if net is None:net=model()
    n=build(e,rank,net);status='built'
   row=dict(id=id,rank=rank,status=status,points=n)
   rows.append(row);print(json.dumps(row),flush=True)
 manifest=OUT/'manifest.json'
 record=dict(status='poseval16_firstframe_multicandidate_features_complete',
   query_frame=0,no_query_future=True,no_simulator_truth=True,no_names=True,
   source_manifest_sha256=sha(MAN),surface_manifest_sha256=sha(SURF/'manifest.json'),
   rows=rows)
 if manifest.exists():
  old=json.loads(manifest.read_text());assert old['source_manifest_sha256']==record['source_manifest_sha256']
 else:manifest.write_text(json.dumps(record,indent=2))
 print(json.dumps(dict(status='complete',samples=len([x for x in rows if x['status']!='no_surface']))),flush=True)
if __name__=='__main__':main()
