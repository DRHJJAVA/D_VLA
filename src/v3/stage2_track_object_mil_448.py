"""Visual-only single-instance RGB tracker on complete TRAIN/VAL 448 RGB-D sequences.
Automatic first-frame component boxes are the only initialization. No action,
object names, simulator contacts/truth, or manual region labels are read.
Tracker outputs are candidate boxes; success flags do not prove identity.
"""
from pathlib import Path
import json,time
import cv2,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
COMP=R/'data/v3/proposals/foreground3d_components'
OUT=R/'data/v3/tracking/mil_448_v1'
def one(split,id):
 source=COMP/split/(id+'.npz')
 with np.load(source) as z:
  path=Path(str(z['rgbd_path']))
  box=z['component_box'].astype(int)
 init=(int(box[0]),int(box[1]),int(box[2]-box[0]),int(box[3]-box[1]))
 target=OUT/split/(id+'.npz')
 target.parent.mkdir(parents=True,exist_ok=True)
 with h5py.File(path) as f:n=len(f['agentview/rgb'])
 if target.exists():
  with np.load(target) as z:
   assert z['boxes'].shape==(n,4) and z['ok'].shape==(n,)
   assert str(z['rgbd_path'])==str(path)
   assert np.array_equal(z['initial_box'],np.asarray(init))
  return dict(split=split,id=id,status='reused',frames=n,path=str(target))
 start=time.time()
 with h5py.File(path) as f:
  rgb=f['agentview/rgb']
  tracker=cv2.TrackerMIL_create()
  tracker.init(cv2.cvtColor(rgb[0],cv2.COLOR_RGB2BGR),init)
  boxes=np.zeros((n,4),dtype=np.int16);ok=np.zeros(n,dtype=bool)
  boxes[0]=init;ok[0]=True
  for i in range(1,n):
   success,b=tracker.update(cv2.cvtColor(rgb[i],cv2.COLOR_RGB2BGR))
   boxes[i]=np.asarray(b,dtype=np.int16);ok[i]=success
 tmp=target.with_suffix('.partial.npz')
 if tmp.exists():raise RuntimeError(str(tmp))
 np.savez_compressed(tmp,boxes=boxes,ok=ok,initial_box=np.asarray(init,dtype=np.int16),
  rgbd_path=np.asarray(str(path)),first_frame_automatic_component=np.asarray(True),
  no_simulator_truth=np.asarray(True),no_manual_labels=np.asarray(True),no_actions=np.asarray(True))
 tmp.replace(target)
 return dict(split=split,id=id,status='complete',frames=n,ok_fraction=float(ok.mean()),
  max_center_displacement_px=float(np.max(np.linalg.norm(boxes[:,:2]+boxes[:,2:]/2-(boxes[0,:2]+boxes[0,2:]/2),axis=1))),
  seconds=round(time.time()-start,2),path=str(target))
def main():
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   row=one(split,id);rows.append(row);print(json.dumps(row),flush=True)
 manifest=OUT/'manifest.json'
 record=dict(status='mil_448_v1_complete',method='OpenCV MIL visually tracks automatic first-frame object bounding box through complete RGB-D demonstration',rows=rows,
  no_names=True,no_manual_labels=True,no_actions=True,no_simulator_truth=True,
  caveat='track boxes not verified object identity; success flags cannot prove correct tracking; validation used only for visual audit')
 if manifest.exists():
  old=json.loads(manifest.read_text());assert old['status']==record['status']
 else:manifest.write_text(json.dumps(record,indent=2))
 print(json.dumps(dict(status='complete',episodes=len(rows))),flush=True)
if __name__=='__main__':main()
