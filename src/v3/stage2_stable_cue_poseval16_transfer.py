"""Frozen stable-cue frequency transfer to previously viewed pose development 16.
Exploratory only: these episodes and old contact diagnostics were already inspected.
No query future, contact, action, name, text or simulator truth enters prediction.
"""
from pathlib import Path
import hashlib,json,time,sys
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_build_frequency_targets import transfer
F=R/'data/v3/samples/object_only_dino_v1'
C=R/'data/v3/samples/stable_near_surface_cue_3dcc_v2'
MAN=R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json'
OUT=R/'outputs/v3_stage2/stable_cue_object_only_poseval16'
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(split,id,donor=False):
 p=F/split/(id+'.npz')
 with np.load(p) as f:
  assert int(f['query_frame'])==0 and bool(f['non_object_pixels_neutral'])
  d={k:f[k].copy() for k in ('xyz','dino_feat','pixel','source_box')}
 if donor:
  with np.load(C/split/(id+'.npz')) as z:
   assert np.array_equal(d['pixel'],z['pixel'])
   d['target']=z['cue'].copy()
 d['id']=id
 v=d['dino_feat'].mean(0)
 d['embedding']=v/max(np.linalg.norm(v),1e-8)
 return d
def main():
 assert json.loads((C/'manifest.json').read_text())['status']=='stable_near_surface_cue_3dcc_v2_complete'
 entries=json.loads(MAN.read_text())['rows'];assert len(entries)==16
 donors=[read('train',id,True) for id in S['train']]
 lock=OUT/'predictions_locked.json'
 if lock.exists():
  old=json.loads(lock.read_text())
  for key,digest in old['prediction_sha256'].items():assert sha(OUT/(key+'.npz'))==digest
  print('already locked');return
 hashes={};rows=[]
 for e in entries:
  id=e['id'];q=read('poseval16',id)
  score,sources=transfer(q,donors)
  dest=OUT/(id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():raise FileExistsError(dest)
  temp=dest.with_suffix('.partial.npz')
  if temp.exists():raise RuntimeError(str(temp))
  np.savez_compressed(temp,score=score,pixel=q['pixel'],xyz=q['xyz'],
   source_box=q['source_box'],query_frame=np.asarray(0),
   has_query_future=np.asarray(False),
   object_only_feature_sha256=np.asarray(sha(F/'poseval16'/(id+'.npz'))))
  temp.replace(dest);hashes[id]=sha(dest)
  rows.append(dict(id=id,points=len(score),sources=sources))
  print(json.dumps(rows[-1]),flush=True)
 record=dict(status='stable_cue_object_only_poseval16_locked',
  previously_viewed_development=True,no_query_future=True,no_simulator_truth=True,
  no_names=True,no_actions=True,no_reserved_blind=True,
  train_donors=len(donors),cue_manifest_sha256=sha(C/'manifest.json'),
  query_manifest_sha256=sha(MAN),prediction_sha256=hashes,
  rows=rows,locked_unix=time.time())
 tmp=lock.with_suffix('.partial')
 if tmp.exists():raise RuntimeError(str(tmp))
 tmp.write_text(json.dumps(record,indent=2));tmp.replace(lock)
 print(json.dumps(dict(status='locked',predictions=len(rows))),flush=True)
if __name__=='__main__':main()
