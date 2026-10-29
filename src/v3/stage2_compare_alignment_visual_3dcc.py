"""Compare train-only visual interaction transfer alignments without simulator truth.
Validation own future-generated visual cue is used for selection only, never as model input."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_build_frequency_targets import S,load
OUT=R/'audits/v3/stage2_alignment_visual_3dcc_only.json'
assert not OUT.exists()
MODES=('world_centered','robust_axis','robust_iso','box_pixel')
def xy(d,mode):
 x=d['xyz'].astype(np.float32)
 if mode=='world_centered':return x-x.mean(0)
 if mode=='robust_axis':
  lo=np.percentile(x,5,axis=0);hi=np.percentile(x,95,axis=0)
  scale=np.maximum(hi-lo,.025)
  return (x-np.median(x,axis=0))/scale
 if mode=='robust_iso':
  center=np.median(x,axis=0);r=np.median(np.linalg.norm(x-center,axis=1))
  return (x-center)/max(r,.025)
 b=d['source_box'].astype(float);pix=d['pixel'].astype(float)
 return (pix-b[:2])/np.maximum(b[2:]-b[:2],1)
def transfer(q,donors,mode):
 pool=[d for d in donors if d['id']!=q['id']]
 sims=np.asarray([q['embedding']@d['embedding'] for d in pool])
 nearest=np.argsort(-sims)[:5]
 weights=np.exp((sims[nearest]-sims[nearest].max())/.03);weights/=weights.sum()
 x=xy(q,mode)
 feature=q['dino_feat'];feature=feature/np.maximum(np.linalg.norm(feature,axis=1,keepdims=True),1e-8)
 score=np.zeros(len(x),np.float64)
 length={'world_centered':.02,'robust_axis':.3,'robust_iso':.4,'box_pixel':.15}[mode]
 for j,w in zip(nearest,weights):
  d=pool[j];ref=xy(d,mode)
  dist,ix=cKDTree(ref).query(x,k=min(16,len(ref)))
  if ix.ndim==1:ix=ix[:,None];dist=dist[:,None]
  f=d['dino_feat'];f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)
  cosine=np.einsum('nd,nkd->nk',feature,f[ix])
  local=np.exp(-dist/length)*np.exp(np.clip((cosine-1)/.08,-30,0))
  score+=w*np.sum(local*d['target'][ix],axis=1)/np.maximum(local.sum(1),1e-8)
 return np.clip(score,0,1)
def metric(score,target):
 n=len(score);ix=np.argpartition(score,-max(1,int(np.ceil(.2*n))))[-max(1,int(np.ceil(.2*n))):]
 mass=float(target[ix].sum()/max(target.sum(),1e-8))
 rank=float(spearmanr(score,target).statistic)
 return dict(top20_visual_mass=mass,random20=len(ix)/n,spearman=rank)
def main():
 donors=[load('train',id,True) for id in S['train']]
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   q=load(split,id,True)
   one=dict(split=split,id=id,points=len(q['xyz']))
   for mode in MODES:
    score=transfer(q,donors,mode)
    one[mode]=metric(score,q['target'])
   rows.append(one);print(json.dumps(one),flush=True)
 summary={}
 for split in ('train','val'):
  rr=[r for r in rows if r['split']==split]
  summary[split]={m:dict(top20_visual_mass=float(np.mean([r[m]['top20_visual_mass'] for r in rr])),spearman=float(np.mean([r[m]['spearman'] for r in rr]))) for m in MODES}
 OUT.write_text(json.dumps(dict(status='visual_cue_comparison_no_simulator_truth',modes=MODES,summary=summary,rows=rows),indent=2))
 print(json.dumps(dict(status='complete',summary=summary)),flush=True)
if __name__=='__main__':main()
