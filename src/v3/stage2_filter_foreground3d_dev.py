"""Keep the first-frame 3D-connected visual object surface for development queries.
Reuses frozen DINO patch features; no future query frame, names, or simulator truth."""
from pathlib import Path
import json
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
COMP=R/'data/v3/proposals/foreground3d_components/dev'
SOURCE=R/'data/v3/samples/grabcut_firstframe_dev/val'
OUT=R/'data/v3/samples/foreground3d_firstframe_dev/val'
def main():
 assert json.loads((COMP.parent/'manifest.json').read_text())['status']=='foreground3d_visual_seeded_complete'
 rows=[]
 for e in json.loads(M.read_text())['rows']:
  id=e['id']
  with np.load(COMP/(id+'.npz')) as z:pixel=z['pixel'].copy();box=z['component_box'].copy()
  with np.load(SOURCE/(id+'.npz')) as z:old={k:z[k].copy() for k in ('xyz','dino_feat','pixel')}
  packed=old['pixel'][:,1].astype(np.int64)*448+old['pixel'][:,0]
  selected=pixel[:,1].astype(np.int64)*448+pixel[:,0]
  keep=np.isin(packed,selected)
  assert keep.sum()==len(selected),(id,keep.sum(),len(selected))
  dest=OUT/(id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():
   with np.load(dest) as z:assert np.array_equal(z['pixel'],old['pixel'][keep])
  else:np.savez_compressed(dest,xyz=old['xyz'][keep],dino_feat=old['dino_feat'][keep],pixel=old['pixel'][keep],source_box=box,first_frame_only=np.asarray(True),proposal_kind=np.asarray('foreground3d_seeded_visual'))
  row=dict(id=id,points=int(keep.sum()),source_box=box.tolist());rows.append(row);print(json.dumps(row),flush=True)
 manifest=OUT.parent/'manifest.json'
 if not manifest.exists():manifest.write_text(json.dumps(dict(status='foreground3d_dev_samples_complete',no_query_future=True,no_simulator_truth=True,rows=rows),indent=2))
 print(json.dumps(dict(status='complete',episodes=len(rows))),flush=True)
if __name__=='__main__':main()
