"""Hash-lock train-only visual frequency transfer on first-frame pose-validation surfaces.
Each query rank is scored independently. No query future, sim contacts, names, or actions.
"""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_build_frequency_targets import transfer
MAN=R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json'
SOURCE=R/'data/v3/samples/visual_interaction_uniform448_3dcc'
QUERY=R/'data/v3/samples/poseval16_firstframe_multicandidate'
OUT=R/'outputs/v3_stage2/visual_frequency_poseval16'
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path,id,donor=False):
 with np.load(path) as z:
  keys=['xyz','dino_feat','pixel','source_box']+(['target'] if donor else [])
  d={k:z[k].copy() for k in keys}
  if not donor:assert int(z['query_frame'])==0 and bool(z['no_query_future'])
 d['id']=id
 v=d['dino_feat'].mean(0);d['embedding']=v/max(np.linalg.norm(v),1e-8)
 return d
def main():
 assert json.loads((QUERY/'manifest.json').read_text())['status']=='poseval16_firstframe_multicandidate_features_complete'
 entries=json.loads(MAN.read_text())['rows'];assert len(entries)==16
 donors=[read(SOURCE/'train'/(id+'.npz'),id,True) for id in S['train']]
 assert len(donors)==20
 lock=OUT/'predictions_locked.json'
 if lock.exists():
  old=json.loads(lock.read_text())
  assert old['query_manifest_sha256']==sha(MAN)
  for key,digest in old['prediction_sha256'].items():
   assert sha(OUT/(key+'.npz'))==digest
  print(json.dumps(dict(status='already_locked',predictions=len(old['prediction_sha256']))),flush=True);return
 rows=[];hashes={}
 for e in entries:
  id=e['id']
  for rank in (1,2,3):
   path=QUERY/id/(f'rank_{rank}.npz')
   if not path.exists():continue
   q=read(path,id,False);score,sources=transfer(q,donors)
   assert np.isfinite(score).all() and len(score)==len(q['xyz'])
   key=f'rank_{rank}/{id}'
   dest=OUT/(key+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():
    with np.load(dest) as z:
     assert str(z['query_sample_sha256'])==sha(path)
     assert np.allclose(z['score'],score,rtol=0,atol=1e-7)
   else:
    tmp=dest.with_suffix('.partial.npz')
    if tmp.exists():raise RuntimeError('inspect partial: '+str(tmp))
    np.savez_compressed(tmp,score=score,pixel=q['pixel'],xyz=q['xyz'],
      source_box=q['source_box'],query_sample_sha256=np.asarray(sha(path)),
      query_frame=np.asarray(0),has_future=np.asarray(False))
    tmp.replace(dest)
   hashes[key]=sha(dest)
   row=dict(id=id,rank=rank,points=len(score),score_mean=float(score.mean()),
     score_max=float(score.max()),source_demos=sources)
   rows.append(row);print(json.dumps(row),flush=True)
 assert len(rows)==48
 record=dict(status='poseval16_multicandidate_visual_frequency_locked',
   query_manifest_sha256=sha(MAN),query_sample_manifest_sha256=sha(QUERY/'manifest.json'),
   train_visual_manifest_sha256=sha(SOURCE/'manifest.json'),
   no_query_future=True,no_simulator_truth=True,no_names=True,
   query_frame=0,train_donors=len(donors),prediction_sha256=hashes,
   method='direct 3D+DINO visual frequency transfer from complete TRAIN RGB-D evidence',
   rows=rows,locked_unix=time.time())
 temp=lock.with_suffix('.partial')
 if temp.exists():raise RuntimeError('inspect partial: '+str(temp))
 temp.write_text(json.dumps(record,indent=2))
 temp.replace(lock)
 print(json.dumps(dict(status='locked',episodes=len(entries),predictions=len(rows))),flush=True)
if __name__=='__main__':main()
