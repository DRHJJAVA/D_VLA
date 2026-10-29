"""Compare multimodal aggregation of train visual interaction maps on TRAIN leave-one-out
and VAL visual targets. No simulator contacts or development outcomes are read."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_build_frequency_targets import S
SOURCE=R/'data/v3/samples/visual_interaction_uniform448_3dcc'
AUDIT=R/'audits/v3/stage2_multimodal_visual_only.json'
MODES=('weighted_mean','unweighted_mean','top2_mean','max','median','mean_plus_max')
def read(path,id):
 with np.load(path) as z:d={k:z[k].copy() for k in ('xyz','dino_feat','target')}
 d['id']=id;v=d['dino_feat'].mean(0);d['embedding']=v/max(np.linalg.norm(v),1e-8)
 return d
def donor_maps(q,donors):
 pool=[d for d in donors if d['id']!=q['id']]
 sims=np.asarray([q['embedding']@d['embedding'] for d in pool])
 nearest=np.argsort(-sims)[:5]
 weights=np.exp((sims[nearest]-sims[nearest].max())/.03);weights/=weights.sum()
 xyz=q['xyz']-q['xyz'].mean(0)
 feature=q['dino_feat'];feature=feature/np.maximum(np.linalg.norm(feature,axis=1,keepdims=True),1e-8)
 maps=[]
 for j in nearest:
  d=pool[j];ref=d['xyz']-d['xyz'].mean(0)
  dist,ix=cKDTree(ref).query(xyz,k=min(16,len(ref)))
  if ix.ndim==1:ix=ix[:,None];dist=dist[:,None]
  f=d['dino_feat'];f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)
  cosine=np.einsum('nd,nkd->nk',feature,f[ix])
  local=np.exp(-dist/.02)*np.exp(np.clip((cosine-1)/.08,-30,0))
  maps.append(np.sum(local*d['target'][ix],axis=1)/np.maximum(local.sum(1),1e-8))
 return np.stack(maps),weights
def aggregate(maps,weights,mode):
 if mode=='weighted_mean':return weights@maps
 if mode=='unweighted_mean':return maps.mean(0)
 if mode=='top2_mean':return np.sort(maps,axis=0)[-2:].mean(0)
 if mode=='max':return maps.max(0)
 if mode=='median':return np.median(maps,axis=0)
 if mode=='mean_plus_max':return .5*(weights@maps)+.5*maps.max(0)
 raise ValueError(mode)
def metric(score,target):
 k=max(1,int(np.ceil(.2*len(score))));ix=np.argpartition(score,-k)[-k:]
 return float(target[ix].sum()/max(target.sum(),1e-8))
def main():
 assert not AUDIT.exists()
 donors=[read(SOURCE/'train'/(id+'.npz'),id) for id in S['train']]
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   q=read(SOURCE/split/(id+'.npz'),id)
   maps,weights=donor_maps(q,donors)
   row=dict(id=id,split=split,points=len(q['xyz']))
   for mode in MODES:row[mode]=metric(aggregate(maps,weights,mode),q['target'])
   rows.append(row);print(json.dumps(row),flush=True)
 summary={split:{mode:float(np.mean([r[mode] for r in rows if r['split']==split])) for mode in MODES} for split in ('train','val')}
 AUDIT.write_text(json.dumps(dict(status='visual_only_no_simulator_truth',modes=MODES,summary=summary,rows=rows),indent=2))
 print(json.dumps(dict(status='complete',summary=summary)),flush=True)
if __name__=='__main__':main()
