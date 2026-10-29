"""Object-only frequency transfer from TRAIN stable near-surface visual evidence.
No query future, contacts, truth, actions, names, or text enters prediction.
Train leave-one-out and old validation only; no reserved blind episodes touched.
"""
from pathlib import Path
import hashlib,json,time,sys
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_build_frequency_targets import transfer
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
F=R/'data/v3/samples/object_only_dino_v1'
C=R/'data/v3/samples/stable_near_surface_cue_v1'
OUT=R/'outputs/v3_stage2/stable_cue_object_only_transfer_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(split,id,donor=False):
 with np.load(F/split/(id+'.npz')) as f:
  assert int(f['query_frame'])==0 and bool(f['non_object_pixels_neutral'])
  d={k:f[k].copy() for k in ('xyz','dino_feat','pixel','source_box')}
 if donor:
  with np.load(C/split/(id+'.npz')) as c:
   assert np.array_equal(d['pixel'],c['pixel'])
   assert np.allclose(d['xyz'],c['xyz'],atol=1e-5)
   d['target']=c['cue'].copy()
 d['id']=id
 v=d['dino_feat'].mean(0)
 d['embedding']=v/max(np.linalg.norm(v),1e-8)
 return d
def main():
 assert json.loads((C/'manifest.json').read_text())['status']=='stable_near_surface_cue_v1_complete'
 donors=[load('train',id,True) for id in S['train']]
 lock=OUT/'predictions_locked.json'
 if lock.exists():
  old=json.loads(lock.read_text())
  for key,digest in old['prediction_sha256'].items():assert sha(OUT/(key+'.npz'))==digest
  print(json.dumps(dict(status='already_locked',rows=len(old['prediction_sha256']))));return
 rows=[];hashes={}
 for split in ('train','val'):
  for id in S[split]:
   q=load(split,id)
   score,sources=transfer(q,donors)
   key=('train_loo' if split=='train' else split)+'/'+id
   dest=OUT/(key+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():raise FileExistsError(dest)
   tmp=dest.with_suffix('.partial.npz')
   if tmp.exists():raise RuntimeError(str(tmp))
   np.savez_compressed(tmp,score=score,pixel=q['pixel'],xyz=q['xyz'],
    query_frame=np.asarray(0),has_query_future=np.asarray(False),
    source_feature_sha256=np.asarray(sha(F/split/(id+'.npz'))))
   tmp.replace(dest);hashes[key]=sha(dest)
   rows.append(dict(id=id,split=split,points=len(score),donors=sources))
   print(json.dumps(rows[-1]),flush=True)
 record=dict(status='stable_cue_object_only_predictions_locked',train_donors=len(donors),
  no_query_future=True,no_names=True,no_actions=True,no_simulator_truth=True,
  no_reserved_blind=True,
  cue_manifest_sha256=sha(C/'manifest.json'),
  feature_manifest_sha256=sha(F/'manifest.json'),
  prediction_sha256=hashes,rows=rows,locked_unix=time.time())
 temp=lock.with_suffix('.partial')
 if temp.exists():raise RuntimeError(str(temp))
 temp.write_text(json.dumps(record,indent=2));temp.replace(lock)
 print(json.dumps(dict(status='locked',rows=len(rows))),flush=True)
if __name__=='__main__':main()
