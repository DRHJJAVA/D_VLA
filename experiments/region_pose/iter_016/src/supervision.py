from pathlib import Path
import json,numpy as np
from scipy.spatial import cKDTree
from tracking_math import rigid,apply
P=Path('/mnt/nvme1/drh/D_VLA/experiments/region_pose')
def reference(row):
 cases=json.loads((P/'iter_013/targets'/row['split']/'summary.json').read_text())['rows']
 return next(c for c in cases if c['event']==row['event'])
def targets(row,z):
 ref=np.load(reference(row)['target_source']);tree=cKDTree(ref['query_xyz'])
 _,idx=tree.query(z['mapped_reference_xyz'])
 if row['query_frame']==row['reference_query']:idx=np.arange(len(z['query_xyz']))
 conf=ref['confidence'][idx].astype(np.float64).copy();known=z['diagnostic_known'].astype(bool);y=z['diagnostic_target'].astype(bool)
 if row['query_frame']!=row['reference_query']:
  a=z['query_match_xyz'][z['inlier']];b=z['reference_match_xyz'][z['inlier']]
  fits=0
  for j in range(len(a)):
   aa=np.delete(a,j,axis=0);bb=np.delete(b,j,axis=0);scale=np.linalg.svd(aa-aa.mean(0),compute_uv=False)/np.sqrt(len(aa))
   if len(aa)<7 or scale[1]<.003:continue
   _,ii=tree.query(apply(z['query_xyz'],rigid(aa,bb)));conf=np.minimum(conf,ref['confidence'][ii]);fits+=1
  assert fits>=7
 conf*=np.exp(-z['position_sensitivity_m']/.002);conf[~known]=0
 assert np.all(conf[known]>0) and np.isfinite(conf).all()
 return y,known,conf.astype(np.float32),idx
