"""Exploratory alignment and multi-mode comparison on pose-development visual targets.
Only TRAIN automatic RGB-D interaction maps supply scores. Pose-validation full sequence
is loaded solely as a frozen audit target, never as a query feature. No sim truth.
"""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
SRC=R/'data/v3/samples/visual_interaction_uniform448_3dcc'
Q=R/'data/v3/samples/poseval16_firstframe_multicandidate'
TARGET=R/'audits/v3/stage2_poseval16_visual_targets'
OUT=R/'audits/v3/stage2_poseval16_visual_method_comparison.json'
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
M=json.loads((R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json').read_text())['rows']
MODES=('world_weighted','world_top2','world_max','robust_axis','robust_iso','box_pixel')
def load(path,id,donor=False):
 with np.load(path) as z:
  keys=['xyz','dino_feat','pixel','source_box']+(['target'] if donor else [])
  d={k:z[k].copy() for k in keys}
 d['id']=id
 v=d['dino_feat'].mean(0);d['embedding']=v/max(np.linalg.norm(v),1e-8)
 return d
def coords(d,mode):
 x=d['xyz']
 if mode=='world':return x-x.mean(0)
 if mode=='robust_axis':
  lo=np.percentile(x,5,axis=0);hi=np.percentile(x,95,axis=0)
  return (x-np.median(x,axis=0))/np.maximum(hi-lo,.025)
 if mode=='robust_iso':
  c=np.median(x,axis=0);rad=np.median(np.linalg.norm(x-c,axis=1))
  return (x-c)/max(rad,.025)
 b=d['source_box'].astype(float);p=d['pixel'].astype(float)
 return (p-b[:2])/np.maximum(b[2:]-b[:2],1)
def maps(q,donors,geometry):
 sims=np.asarray([q['embedding']@d['embedding'] for d in donors])
 near=np.argsort(-sims)[:5]
 w=np.exp((sims[near]-sims[near].max())/.03);w/=w.sum()
 a=coords(q,geometry)
 f=q['dino_feat'];f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)
 length={'world':.02,'robust_axis':.3,'robust_iso':.4,'box_pixel':.15}[geometry]
 result=[]
 for j in near:
  d=donors[j];b=coords(d,geometry)
  dist,ix=cKDTree(b).query(a,k=min(16,len(b)))
  if ix.ndim==1:ix=ix[:,None];dist=dist[:,None]
  df=d['dino_feat'];df=df/np.maximum(np.linalg.norm(df,axis=1,keepdims=True),1e-8)
  cosine=np.einsum('nd,nkd->nk',f,df[ix])
  local=np.exp(-dist/length)*np.exp(np.clip((cosine-1)/.08,-30,0))
  result.append(np.sum(local*d['target'][ix],axis=1)/np.maximum(local.sum(1),1e-8))
 return np.stack(result),w
def mass(score,target):
 k=max(1,int(np.ceil(.2*len(score))))
 top=np.argpartition(score,-k)[-k:]
 return float(target[top].sum()/max(target.sum(),1e-8))
def main():
 assert not OUT.exists()
 donors=[load(SRC/'train'/(id+'.npz'),id,True) for id in S['train']]
 assert len(donors)==20 and len(M)==16
 rows=[]
 for e in M:
  id=e['id'];q=load(Q/id/'rank_1.npz',id)
  with np.load(TARGET/(id+'.npz')) as z:t=z['target'].copy()
  assert len(q['xyz'])==len(t)
  a,w=maps(q,donors,'world')
  scores={'world_weighted':w@a,'world_top2':np.sort(a,axis=0)[-2:].mean(0),
   'world_max':a.max(0)}
  for geometry in ('robust_axis','robust_iso','box_pixel'):
   arr,weights=maps(q,donors,geometry)
   scores[geometry]=weights@arr
  row=dict(id=id,points=len(t))
  for key in MODES:row[key]=mass(scores[key],t)
  rows.append(row);print(json.dumps(row),flush=True)
 summary={key:float(np.mean([x[key] for x in rows])) for key in MODES}
 result=dict(status='exploratory_pose_development_visual_only_not_blind',
  query_frame=0,validation_future_for_audit_only=True,no_simulator_truth=True,
  budget='top20% surface points',summary=summary,rows=rows)
 OUT.write_text(json.dumps(result,indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
