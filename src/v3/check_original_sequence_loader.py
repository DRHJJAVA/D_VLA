"""Read-only source audit. Manifest contains references, never edited recordings."""
from pathlib import Path
import json,datetime,sys
import h5py,numpy as np
from episode_dataset import audit_loader
R=Path('/mnt/nvme1/drh/D_VLA')
out=R/'data/v3/manifests';out.mkdir(parents=True,exist_ok=True)
rows=[];seen=set();stats={}
for p in sorted((R/'data/v2/stage1_rgbd').glob('*/*.h5')):
 with h5py.File(p,'r') as f:
  assert '_generated_tail' not in f
  n=int(f.attrs.get('frames',f.attrs.get('num_frames')))
  source=Path(str(f.attrs.get('source',f.attrs.get('source_hdf5'))))
  if not source.is_absolute():source=R/source
  ep=str(f.attrs['episode']);key=(str(source.resolve()),ep)
  assert key not in seen;seen.add(key)
  for x in [p,source]:
   stats[str(x)]=(x.stat().st_size,x.stat().st_mtime_ns)
  with h5py.File(source,'r') as original:
   states=original['data'][ep]['states']
   assert len(states)==n
   tt=np.asarray(states[:,0],dtype=float)
  assert len(tt)==n and np.isfinite(tt).all() and (np.diff(tt)>0).all()
  if 'frame_indices' in f:assert np.array_equal(f['frame_indices'][:],np.arange(n))
  for cam in ['agentview','robot0_eye_in_hand']:
   k=f[cam+'/K'][:];t=f[cam+'/camera_to_world'][:]
   assert np.isfinite(k).all() and np.isfinite(t).all()
   assert (k[:,0,0]>0).all() and (k[:,1,1]>0).all()
   assert np.allclose(t[:,3,:],[0,0,0,1],atol=1e-5)
   assert np.allclose(t[:,:3,:3].transpose(0,2,1)@t[:,:3,:3],np.eye(3),atol=1e-4)
  rows.append(dict(id=p.parent.name+'_'+ep,asset_group=p.parent.name,source_path=str(source),source_episode=ep,rgbd_path=str(p),frames=n,timestamps_seconds=tt.tolist(),depth_provenance='existing replay from official saved simulation states',interaction_count=None,action_types=None,split='unassigned',all_supplied_frames=True))
manifest=out/'original_sequences.jsonl'
manifest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
report=R/'audits/v3/original_sequence_loader_check.json'
result=audit_loader(manifest,report)
for path,before in stats.items():
 p=Path(path);assert (p.stat().st_size,p.stat().st_mtime_ns)==before
result.update(generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_files_opened_read_only=True,source_size_and_mtime_unchanged=True,duplicate_source_episode_count=0,synthetic_frames_in_manifest=0,fixed_phase_gate=False,interaction_count_gate=False,source_timestamps_monotonic=True,visual_alignment_rechecked=False,depth_provenance='simulation-state replay, not official provided depth',scope='existing 30 pick/place sequences only; mixed-action content coverage not established')
report.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)
