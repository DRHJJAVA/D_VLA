"""Uniform first-frame RGB-D GrabCut on 448 train and validation replay.
Only derived visual data and automatic motion boxes; no simulator truth or robot actions."""
from pathlib import Path
import json,sys
import h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_grabcut_component_dev import segment
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
E={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
OUT=R/'data/v3/proposals/grabcut_components/train_val_448'
RENDER=R/'data/v3/trainval_rgbd_448/manifest.json'
def main():
 assert json.loads(RENDER.read_text())['status']=='train_val_448_complete','wait for replay completion'
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   e=E[id];g=e['asset_group'];ep=e['source_episode']
   source=Path(e['rgbd_path'])
   with h5py.File(source) as f:original_size=int(f['agentview/rgb'].shape[1])
   path=source if original_size==448 else R/'data/v3/trainval_rgbd_448'/g/(ep+'.h5')
   assert path.exists(),path
   with h5py.File(path) as f:
    rgb=f['agentview/rgb'][0];depth=f['agentview/depth_m'][0]
   h,w=depth.shape;assert h==w==448
   p=R/'data/v3/proposals/motion_rgbd_rerank'/g/(ep+'.json')
   b=json.loads(p.read_text())['selected']['box']
   base=[int(round(v*448/original_size)) for v in b]
   x1,y1,x2,y2=base;dx=int(round((x2-x1)*.25));dy=int(round((y2-y1)*.25))
   outer=[max(0,x1-dx),max(0,y1-dy),min(w,x2+dx),min(h,y2+dy)]
   pixel,box,z=segment(rgb,depth,base,outer)
   dest=OUT/split/(id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():
    with np.load(dest) as old:assert np.array_equal(old['pixel'],pixel)
   else:np.savez_compressed(dest,pixel=pixel,component_box=np.asarray(box),base_box=np.asarray(base),outer_box=np.asarray(outer),seed_depth=np.asarray(z),query_frame=np.asarray(0),no_simulator_truth=np.asarray(True),rgbd_path=np.asarray(str(path)))
   row=dict(id=id,split=split,points=len(pixel),box=box,base_box=base,outer_box=outer,rgbd_path=str(path))
   rows.append(row);print(json.dumps(row),flush=True)
 manifest=OUT/'manifest.json'
 data=dict(status='uniform_448_train_val_grabcut_complete',no_simulator_truth=True,rows=rows)
 if manifest.exists():assert json.loads(manifest.read_text())['rows']==rows
 else:manifest.write_text(json.dumps(data,indent=2))
 print(json.dumps(dict(status='complete',train=len(S['train']),val=len(S['val']))),flush=True)
if __name__=='__main__':main()
