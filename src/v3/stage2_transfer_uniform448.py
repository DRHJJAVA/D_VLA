"""Transfer complete-sequence visual interaction frequency from uniform 448 train donors.
Train leave-one-out, validation, and separate development queries use only first-frame
RGB-D at inference. Scores are hash-locked before any simulator-contact diagnostic."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_build_frequency_targets import S,transfer
SOURCE=R/'data/v3/samples/visual_interaction_uniform448'
DEV=R/'data/v3/samples/grabcut_firstframe_dev/val'
OUT=R/'outputs/v3_stage2/visual_frequency_uniform448'
DEV_MANIFEST=R/'data/v3/manifests/stage2_localization_dev_8.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path,id,donor=False):
 with np.load(path) as z:
  keys=['xyz','dino_feat','pixel','source_box']+(['target'] if donor else [])
  d={k:z[k].copy() for k in keys}
 d['id']=id
 v=d['dino_feat'].mean(0);d['embedding']=v/max(np.linalg.norm(v),1e-8)
 return d
def main():
 assert json.loads((SOURCE/'manifest.json').read_text())['status']=='visual_interaction_uniform448_complete'
 donors=[read(SOURCE/'train'/(id+'.npz'),id,True) for id in S['train']]
 assert len(donors)==20
 queries=[('train_loo',id,SOURCE/'train'/(id+'.npz')) for id in S['train']]
 queries += [('val',id,SOURCE/'val'/(id+'.npz')) for id in S['val']]
 queries += [('dev',e['id'],DEV/(e['id']+'.npz')) for e in json.loads(DEV_MANIFEST.read_text())['rows']]
 rows=[];hashes={}
 for split,id,path in queries:
  q=read(path,id,False)
  score,sources=transfer(q,donors)
  assert np.isfinite(score).all() and len(score)==len(q['xyz'])
  dest=OUT/split/(id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():
   with np.load(dest) as z:assert np.allclose(z['score'],score,rtol=0,atol=1e-7)
  else:
   temp=dest.with_suffix('.partial.npz')
   if temp.exists():raise RuntimeError('inspect partial: '+str(temp))
   np.savez_compressed(temp,score=score,pixel=q['pixel'],xyz=q['xyz'],source_box=q['source_box'],query_frame=np.asarray(0))
   temp.replace(dest)
  hashes[split+'/'+id]=sha(dest)
  row=dict(split=split,id=id,points=len(score),sources=sources)
  if split in ('train_loo','val'):
   with np.load(path) as z:visual=z['target']
   k=max(1,int(np.ceil(.2*len(score))))
   top=np.argpartition(score,-k)[-k:]
   row['top20_visual_mass']=float(visual[top].sum()/max(visual.sum(),1e-8))
   row['random20']=k/len(score)
  rows.append(row);print(json.dumps(row),flush=True)
 lock=OUT/'predictions_locked.json'
 if lock.exists():raise FileExistsError(lock)
 lock.write_text(json.dumps(dict(status='uniform448_visual_frequency_locked_before_old_contact_audit',no_simulator_truth=True,no_query_future=True,train_donors=len(donors),source_manifest_sha256=sha(SOURCE/'manifest.json'),dev_manifest_sha256=sha(DEV_MANIFEST),prediction_sha256=hashes,rows=rows,created_unix=time.time()),indent=2))
 print(json.dumps(dict(status='locked',train_loo=len(S['train']),val=len(S['val']),dev=8)),flush=True)
if __name__=='__main__':main()
