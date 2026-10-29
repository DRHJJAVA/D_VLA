"""Re-render train/validation RGB-D at 448 in an isolated derived directory.
Official HDF5 and prior 256/448 renderings remain untouched."""
from pathlib import Path
import json,subprocess,sys
import h5py
R=Path('/mnt/nvme1/drh/D_VLA')
LIBERO=Path('/home/drh/project/ECV-VLA/vendor/LIBERO')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
E={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
OUT=R/'data/v3/trainval_rgbd_448'
def main():
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   row=E[id];source=Path(row['rgbd_path'])
   with h5py.File(source) as f:oldsize=int(f['agentview/rgb'].shape[1])
   if oldsize==448:
    rows.append(dict(id=id,split=split,status='existing_448',path=str(source),frames=row['frames']))
    print(json.dumps(rows[-1]),flush=True);continue
   assert oldsize==256,oldsize
   dest=OUT/row['asset_group']/(row['source_episode']+'.h5')
   dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():
    with h5py.File(dest) as f:
     assert int(f.attrs['num_frames'])==row['frames']
     assert int(f['agentview/rgb'].shape[1])==448
    rows.append(dict(id=id,split=split,status='reused_448',path=str(dest),frames=row['frames']))
    print(json.dumps(rows[-1]),flush=True);continue
   partial=dest.with_suffix(dest.suffix+'.part')
   if partial.exists():raise RuntimeError('Inspect partial first: '+str(partial))
   log=R/'logs'/('stage2_trainval448_render_'+id+'.log')
   cmd=[sys.executable,str(R/'src/render_rgbd_episode.py'),'--input',row['source_path'],'--output',str(dest),'--libero-root',str(LIBERO),'--episode',row['source_episode'],'--size','448','--bddl-subdir',Path(row['source_path']).parent.name]
   print(json.dumps(dict(id=id,event='start',log=str(log))),flush=True)
   with log.open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=R)
   with h5py.File(dest) as f:
    assert int(f.attrs['num_frames'])==row['frames']
    assert int(f['agentview/rgb'].shape[1])==448
   rows.append(dict(id=id,split=split,status='rendered_448',path=str(dest),frames=row['frames']))
   print(json.dumps(rows[-1]),flush=True)
 manifest=OUT/'manifest.json'
 if not manifest.exists():manifest.write_text(json.dumps(dict(status='train_val_448_complete',source='official_read_only',rows=rows),indent=2))
 print(json.dumps(dict(event='train_val_448_complete',episodes=len(rows))),flush=True)
if __name__=='__main__':main()
