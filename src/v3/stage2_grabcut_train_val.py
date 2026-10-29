"""Uniform first-frame RGB-D foreground extraction on train/val demonstrations.
Uses automatic full-sequence motion box for TRAIN input preparation only;
does not read simulator contacts, object truth, names, or robot actions."""
from pathlib import Path
import json,sys
import h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_grabcut_component_dev import segment
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
E={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
OUT=R/'data/v3/proposals/grabcut_components/train_val'
def expanded(b,shape):
 x1,y1,x2,y2=b;h,w=shape
 dx=int(round((x2-x1)*.25));dy=int(round((y2-y1)*.25))
 return [max(0,x1-dx),max(0,y1-dy),min(w,x2+dx),min(h,y2+dy)]
def main():
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   e=E[id];g=e['asset_group'];ep=e['source_episode']
   p=R/'data/v3/proposals/motion_rgbd_rerank'/g/(ep+'.json')
   base=json.loads(p.read_text())['selected']['box']
   with h5py.File(e['rgbd_path']) as f:
    rgb=f['agentview/rgb'][0];depth=f['agentview/depth_m'][0]
   outer=expanded(base,depth.shape)
   pixel,box,z=segment(rgb,depth,base,outer)
   dest=OUT/split/(id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():
    with np.load(dest) as old:assert np.array_equal(old['pixel'],pixel)
   else:np.savez_compressed(dest,pixel=pixel,component_box=np.asarray(box),base_box=np.asarray(base),outer_box=np.asarray(outer),seed_depth=np.asarray(z),query_frame=np.asarray(0),no_simulator_truth=np.asarray(True))
   row=dict(id=id,split=split,points=len(pixel),box=box,base_box=base,outer_box=outer)
   rows.append(row);print(json.dumps(row),flush=True)
 manifest=OUT/'manifest.json'
 if not manifest.exists():manifest.write_text(json.dumps(dict(status='visual_only_train_val_grabcut',rows=rows),indent=2))
 print(json.dumps(dict(status='complete',train=len(S['train']),val=len(S['val']))),flush=True)
if __name__=='__main__':main()
