import json,hashlib
from pathlib import Path
import numpy as np
P=Path('/mnt/nvme1/drh/D_VLA')
O=P/'experiments/region_pose/iter_035'
LAST_SEED=None
FILTER_ARRAYS={}
def augment_features(ident,*args,**kwargs):
 global LAST_SEED
 import cv2
 actual=cv2.goodFeaturesToTrack(*args,**kwargs)
 actual=np.empty((0,2),np.float32) if actual is None else actual.reshape(-1,2)
 with np.load(O/'seeds'/(ident+'.npz'),allow_pickle=False) as z:
  assert np.array_equal(kwargs['mask'],z['mask'])
  assert np.array_equal(actual,z['original_uv'])
  alluv=z['initial_uv'].copy()
 LAST_SEED={'episode':ident,'original':len(actual),'total':len(alluv)}
 return alluv.reshape(-1,1,2) if len(alluv) else None
def capture_stage(ident,frame,stage,values):
 key=ident+'_frame'+str(int(frame))
 dest=FILTER_ARRAYS.setdefault(key,{})
 for n in ['ok','alive','tracked','endpoint','corr','uv','xy']:
  a=values.get(n)
  if isinstance(a,np.ndarray) and a.dtype.kind in 'biuf':
   dest[stage+'__'+n]=a.copy()
 if stage=='corr':
  np.savez_compressed(O/'train/filter_masks'/(key+'.npz'),**dest)
  del FILTER_ARRAYS[key]
