from pathlib import Path
import json,hashlib,h5py,numpy as np,time
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018')
cases=json.loads((R/'episode_manifest.json').read_text())['cases']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
checked={}
for c in cases:
 for typ in ['rgbd','raw']:
  p=Path(c[typ+'_path'])
  if str(p) not in checked:checked[str(p)]=sha(p)
  assert checked[str(p)]==c[typ+'_sha256'],(c['case'],typ)
 with h5py.File(c['rgbd_path']) as a,h5py.File(c['raw_path']) as b:
  n=c['frames']
  assert np.array_equal(a['frame_indices'][:],np.arange(n))
  assert len(b[f"data/demo_{c['demo_index']}/actions"])==n
  for cam in c['camera_views']:
   assert a[f'{cam}/rgb'].shape==(n,448,448,3)
 for folder_name in ['prepared','prepared_fine']:
  folder=R/folder_name/c['case']
  sensors=json.loads((folder/'sensors.json').read_text())
  hashes=json.loads((folder/'images_sha256.json').read_text())
  wanted=range(0,n,2) if folder_name=='prepared' else range(n)
  target=set(wanted);target.add(n-1)
  assert [x['frame'] for x in sensors['rows']]==sorted(target)
  assert sensors['rows'][0]['state_valid']==False
  assert all(x['obs_index']==x['frame']-1 for x in sensors['rows'][1:])
  assert len(hashes)==2*len(target)
  for k,v in hashes.items():assert sha(folder/k)==v,(folder_name,c['case'],k)
out={'stage':'complete','time':time.strftime('%Y-%m-%d %H:%M:%S%z'),'unique_source_files':len(checked),'cases':len(cases),'image_files_checked':sum(2*(len(set(range(0,c['frames'],2))|{c['frames']-1})+c['frames']) for c in cases),'human_ranges_used':False,'reserved_test_access':False}
(R/'input_preflight.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False))
