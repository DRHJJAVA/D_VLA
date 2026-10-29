"""Render predeclared pose-validation demonstrations in an isolated 448 RGB-D directory.
Official HDF5 is read-only; no reserved blind episode is opened."""
from pathlib import Path
import json,subprocess,sys
import h5py
R=Path('/mnt/nvme1/drh/D_VLA')
LIBERO=Path('/home/drh/project/ECV-VLA/vendor/LIBERO')
M=R/'data/v3/manifests/stage2_pose_validation_16.json'
OUT=R/'data/v3/pose_validation_rgbd_448'
def main():
 m=json.loads(M.read_text());assert m['status']=='predeclared_pose_development_validation_not_blind'
 rows=[]
 for row in m['rows']:
  dest=OUT/row['asset_group']/(row['source_episode']+'.h5')
  dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():
   with h5py.File(dest) as f:
    assert int(f.attrs['num_frames'])==row['source_frames']
    assert f['agentview/rgb'].shape[1:3]==(448,448)
   status='reused'
  else:
   partial=dest.with_suffix(dest.suffix+'.part')
   if partial.exists():raise RuntimeError('inspect partial first: '+str(partial))
   log=R/'logs'/('stage2_poseval_render_'+row['id']+'.log')
   cmd=[sys.executable,str(R/'src/render_rgbd_episode.py'),'--input',row['source_path'],'--output',str(dest),'--libero-root',str(LIBERO),'--episode',row['source_episode'],'--size','448','--bddl-subdir',Path(row['source_path']).parent.name]
   print(json.dumps(dict(id=row['id'],event='start',log=str(log))),flush=True)
   with log.open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=R)
   with h5py.File(dest) as f:
    assert int(f.attrs['num_frames'])==row['source_frames']
    assert f['agentview/rgb'].shape[1:3]==(448,448)
   status='rendered'
  info=dict(id=row['id'],status=status,path=str(dest),frames=row['source_frames'])
  rows.append(info);print(json.dumps(info),flush=True)
 manifest=OUT/'manifest.json'
 if manifest.exists():assert json.loads(manifest.read_text())['rows']==rows
 else:manifest.write_text(json.dumps(dict(status='pose_validation_16_rgbd_complete',selection_manifest=str(M),source_read_only=True,rows=rows),indent=2))
 print(json.dumps(dict(event='pose_validation_16_rgbd_complete',episodes=len(rows))),flush=True)
if __name__=='__main__':main()
