"""Build train-only cross-demo frequency targets for a first-pose student.
The query's own future is never used; validation targets come solely from TRAIN demonstrations."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
SOURCE=R/'data/v3/samples/visual_interaction_v1'
OUT=R/'data/v3/samples/frequency_student_v1'
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
def load(split,id,donor=False):
 with np.load(SOURCE/split/(id+'.npz')) as f:
  keys=['xyz','dino_feat','pixel','source_box']+(['target'] if donor else [])
  d={k:f[k].copy() for k in keys}
 d['id']=id
 v=d['dino_feat'].mean(0);d['embedding']=v/np.linalg.norm(v)
 return d
def transfer(q,donors):
 pool=[d for d in donors if d['id']!=q['id']]
 sims=np.asarray([q['embedding']@d['embedding'] for d in pool])
 nearest=np.argsort(-sims)[:5]
 weights=np.exp((sims[nearest]-sims[nearest].max())/.03);weights/=weights.sum()
 xyz=q['xyz']-q['xyz'].mean(0)
 feature=q['dino_feat'];feature=feature/np.maximum(np.linalg.norm(feature,axis=1,keepdims=True),1e-8)
 score=np.zeros(len(xyz))
 for j,w in zip(nearest,weights):
  d=pool[j];ref=d['xyz']-d['xyz'].mean(0)
  dist,ix=cKDTree(ref).query(xyz,k=min(16,len(ref)))
  if ix.ndim==1:ix=ix[:,None];dist=dist[:,None]
  f=d['dino_feat'];f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)
  cosine=np.einsum('nd,nkd->nk',feature,f[ix])
  local=np.exp(-dist/.02)*np.exp(np.clip((cosine-1)/.08,-30,0))
  score+=w*np.sum(local*d['target'][ix],axis=1)/np.maximum(local.sum(1),1e-8)
 return np.clip(score,0,1).astype(np.float32),[(pool[j]['id'],float(w)) for j,w in zip(nearest,weights)]
def main():
 donors=[load('train',id,True) for id in S['train']]
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   q=load(split,id,False)
   target,sources=transfer(q,donors)
   sample=OUT/split/(id+'.npz');sample.parent.mkdir(parents=True,exist_ok=True)
   if sample.exists():raise FileExistsError(sample)
   np.savez_compressed(sample,xyz=q['xyz'],dino_feat=q['dino_feat'],target=target,pixel=q['pixel'],source_box=q['source_box'])
   row=dict(id=id,split=split,points=len(target),target_mean=float(target.mean()),target_std=float(target.std()),source_demos=sources)
   rows.append(row);print(json.dumps(row),flush=True)
 manifest=dict(target='average full-sequence visual interaction cues transferred from visually retrieved TRAIN demonstrations',query_future_used=False,validation_future_used=False,no_names=True,no_simulator_truth=True,train_leave_one_out=True,rows=rows)
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
 print(json.dumps(dict(complete=True,train=len(S['train']),val=len(S['val']))),flush=True)
if __name__=='__main__':main()
