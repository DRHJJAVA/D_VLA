"""Render the preregistered eight untouched official episodes to remote RGB-D derivatives.
Never changes official HDF5 files. Completed outputs are reused; partial files stop the run."""
from pathlib import Path
import json,subprocess,sys,time
import h5py
R=Path('/mnt/nvme1/drh/D_VLA')
MANIFEST=R/'data/v3/manifests/stage2_fresh_blind_8.json'
LIBERO=Path('/home/drh/project/ECV-VLA/vendor/LIBERO')
def main():
 m=json.loads(MANIFEST.read_text())
 assert m['status']=='frozen_before_render_or_prediction' and len(m['rows'])==8
 for row in m['rows']:
  dest=Path(row['rgbd_path']);dest.parent.mkdir(parents=True,exist_ok=True)
  partial=dest.with_suffix(dest.suffix+'.part')
  if dest.exists():
   with h5py.File(dest) as f:
    assert f.attrs['episode']==row['source_episode'] and int(f.attrs['num_frames'])==row['source_frames']
    assert f['agentview/rgb'].shape[0]==row['source_frames']
   print(json.dumps(dict(id=row['id'],status='reused')),flush=True);continue
  if partial.exists():raise RuntimeError('Inspect incomplete render before retry: '+str(partial))
  log=R/'logs'/('stage2_fresh_render_'+row['id']+'.log')
  cmd=[sys.executable,str(R/'src/render_rgbd_episode.py'),'--input',row['source_path'],'--output',str(dest),'--libero-root',str(LIBERO),'--episode',row['source_episode'],'--size','448','--bddl-subdir',Path(row['source_path']).parent.name]
  with log.open('w') as f:
   print(json.dumps(dict(event='start',id=row['id'],output=str(dest),log=str(log))),flush=True)
   subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=R)
  with h5py.File(dest) as f:assert int(f.attrs['num_frames'])==row['source_frames']
  print(json.dumps(dict(event='complete',id=row['id'],frames=row['source_frames'])),flush=True)
 print(json.dumps(dict(event='all_fresh_rgbd_complete',count=len(m['rows']))),flush=True)
if __name__=='__main__':main()
