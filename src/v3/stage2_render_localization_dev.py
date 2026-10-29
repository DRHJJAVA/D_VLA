"""Render separate development episodes; official HDF5 remains read-only."""
from pathlib import Path
import json,subprocess,sys
import h5py
R=Path('/mnt/nvme1/drh/D_VLA')
LIBERO=Path('/home/drh/project/ECV-VLA/vendor/LIBERO')
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
def main():
    m=json.loads(M.read_text());assert m['status']=='development_set_not_blind_test'
    for row in m['rows']:
        dest=Path(row['rgbd_path']);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            with h5py.File(dest) as f:assert int(f.attrs['num_frames'])==row['source_frames']
            print(json.dumps(dict(id=row['id'],status='reused')),flush=True);continue
        partial=dest.with_suffix(dest.suffix+'.part')
        if partial.exists():raise RuntimeError('Inspect partial first: '+str(partial))
        log=R/'logs'/('stage2_locdev_render_'+row['id']+'.log')
        cmd=[sys.executable,str(R/'src/render_rgbd_episode.py'),'--input',row['source_path'],'--output',str(dest),'--libero-root',str(LIBERO),'--episode',row['source_episode'],'--size','448','--bddl-subdir',Path(row['source_path']).parent.name]
        print(json.dumps(dict(id=row['id'],event='start',log=str(log))),flush=True)
        with log.open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=R)
        with h5py.File(dest) as f:assert int(f.attrs['num_frames'])==row['source_frames']
        print(json.dumps(dict(id=row['id'],event='complete',frames=row['source_frames'])),flush=True)
    print(json.dumps(dict(event='development_rgbd_complete',episodes=len(m['rows']))),flush=True)
if __name__=='__main__':main()
