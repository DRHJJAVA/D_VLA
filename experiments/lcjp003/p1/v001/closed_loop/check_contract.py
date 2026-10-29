
from adapters import *
import traceback
try:
 eid='pick_up_the_bbq_sauce_and_place_it_in_the_basket__demo_0'
 live=Dataset(cached=False);live.old=HERE/'validation_raw';live._cache=collections.OrderedDict()
 a=live.frame(eid,0)
 ref=Dataset(cached=True);b=ref.frame(eid,0)
 diffs={}
 for k in a:
  if isinstance(a[k],np.ndarray) and k in b:
   x=a[k];y=b[k]
   diffs[k]={'shape_match':x.shape==y.shape,'equal':bool(np.array_equal(x,y,equal_nan=True)),'max_abs':float(np.nanmax(np.abs(x.astype(float)-y.astype(float)))) if x.shape==y.shape else None}
 report={'first_frame':diffs,'live_objects':int(np.unique(a['object_id'][a['point_valid']]).size),'cached_objects':int(np.unique(b['object_id'][b['point_valid']]).size)}
 # Causal window grouping/state logic is reused verbatim; compare it on fixed frames.
 direct=Dataset(cached=False)
 x=direct.load_window(eid,0);y=ref.window(eid,0)
 report['window_geometry']={k:bool(np.array_equal(x[k],y[k],equal_nan=True)) for k in KEYS if k!='semantic'}
 report['action_roundtrip_max_abs']=float(np.max(np.abs(((direct.actions[eid]-direct.action_mean)/direct.action_std)*direct.action_std+direct.action_mean-direct.actions[eid])))
 (HERE/'input_contract.json').write_text(json.dumps(report,indent=2))
except Exception as e:
 (HERE/'contract_failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e)[:400],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]},indent=2))
