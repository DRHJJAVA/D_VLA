"""Strict single-object visual-frequency transfer with predictions locked before audit.
TRAIN targets come from complete TRAIN RGB-D only; query features are masked
single-object first-frame RGB-D cutouts and object surface points.
"""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_build_frequency_targets import transfer
SRC=R/'data/v3/samples/object_only_dino_v1'
OUT=R/'outputs/v3_stage2/object_only_frequency_v1'
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
POSE=json.loads((R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json').read_text())['rows']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path,id,donor=False):
 with np.load(path) as z:
  keys=['xyz','dino_feat','pixel','source_box']+(['target'] if donor else [])
  d={k:z[k].copy() for k in keys}
  assert int(z['query_frame'])==0 and bool(z['non_object_pixels_neutral'])
 d['id']=id
 v=d['dino_feat'].mean(0);d['embedding']=v/max(np.linalg.norm(v),1e-8)
 return d
def main():
 assert json.loads((SRC/'manifest.json').read_text())['status']=='strict_object_only_features_complete'
 donors=[read(SRC/'train'/(id+'.npz'),id,True) for id in S['train']]
 assert len(donors)==20 and len(POSE)==16
 queries=[('train_loo',id,SRC/'train'/(id+'.npz')) for id in S['train']]
 queries += [('val',id,SRC/'val'/(id+'.npz')) for id in S['val']]
 queries += [('poseval16',e['id'],SRC/'poseval16'/(e['id']+'.npz')) for e in POSE]
 assert len(queries)==40
 lock=OUT/'predictions_locked.json'
 if lock.exists():
  old=json.loads(lock.read_text())
  assert old['source_manifest_sha256']==sha(SRC/'manifest.json')
  for key,digest in old['prediction_sha256'].items():assert sha(OUT/(key+'.npz'))==digest
  print(json.dumps(dict(status='already_locked',predictions=len(old['prediction_sha256']))),flush=True);return
 rows=[];hashes={}
 for split,id,path in queries:
  q=read(path,id,False)
  score,sources=transfer(q,donors)
  assert len(score)==len(q['xyz']) and np.isfinite(score).all()
  key=split+'/'+id;dest=OUT/(key+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():
   with np.load(dest) as z:
    assert str(z['query_sample_sha256'])==sha(path)
    assert np.allclose(z['score'],score,atol=1e-7,rtol=0)
  else:
   tmp=dest.with_suffix('.partial.npz')
   if tmp.exists():raise RuntimeError('inspect partial: '+str(tmp))
   np.savez_compressed(tmp,score=score,xyz=q['xyz'],pixel=q['pixel'],
     source_box=q['source_box'],query_sample_sha256=np.asarray(sha(path)),
     query_frame=np.asarray(0),has_future=np.asarray(False))
   tmp.replace(dest)
  hashes[key]=sha(dest)
  row=dict(split=split,id=id,points=len(score),sources=sources)
  rows.append(row);print(json.dumps(row),flush=True)
 record=dict(status='strict_object_only_frequency_predictions_locked',
  query_frame=0,non_object_pixels_neutral=True,no_query_future=True,
  no_simulator_truth=True,no_names=True,train_donors=len(donors),
  source_manifest_sha256=sha(SRC/'manifest.json'),
  pose_manifest_sha256=sha(R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json'),
  prediction_sha256=hashes,rows=rows,locked_unix=time.time())
 temp=lock.with_suffix('.partial')
 if temp.exists():raise RuntimeError('inspect partial: '+str(temp))
 temp.write_text(json.dumps(record,indent=2));temp.replace(lock)
 print(json.dumps(dict(status='locked',predictions=len(rows))),flush=True)
if __name__=='__main__':main()
