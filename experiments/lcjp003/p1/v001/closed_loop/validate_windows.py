
from adapters import *
import types,traceback
try:
 eid='pick_up_the_bbq_sauce_and_place_it_in_the_basket__demo_0'
 ref=Dataset(cached=True)
 # Build the live adapter without loading an encoder; supply only causal recorded frame fixtures.
 obj=LiveInput.__new__(LiveInput);obj.eid=eid;obj.device='cpu'
 obj.ds=Dataset.__new__(Dataset);obj.ds.cached=False;obj.ds.rows={eid:dict(ref.rows[eid])};obj.ds.rows[eid]['frames']=600
 obj.ds.actions={eid:np.zeros((600,7),np.float32)}
 obj.ds.action_mean=ref.action_mean;obj.ds.action_std=ref.action_std
 obj.ds.instructions=ref.instructions;obj.ds.adapter_sha256=ref.adapter_sha256;obj.ds.input_protocol_version=ref.input_protocol_version
 text=np.load(P/'text/train_task.npz');obj.text={k:text[k] for k in text.files}
 results={}
 for t in [0,3,15,20]:
  seen=[]
  def frame(ds,e,i,n_points=2048):
   assert i<=t
   seen.append(i)
   return ref.frame(e,i,n_points)
  obj.ds.frame=types.MethodType(frame,obj.ds)
  obj.history={i:ref.frame(eid,i)['semantic'] for i in range(max(0,t-15),t+1)}
  actual=obj.batch(t);expected=ref.window(eid,t)
  equal={k:bool(np.array_equal(actual[k][0].numpy(),expected[k],equal_nan=True)) for k in actual}
  (HERE/'window_difference.json').write_text(json.dumps({'t':t,'fields':equal,'differences':{k:float(np.max(np.abs(actual[k][0].numpy().astype(float)-expected[k].astype(float)))) for k in actual if not equal[k]}},indent=2))
  assert all(equal.values())
  results[str(t)]={'fields_exact':equal,'max_frame_read':max(seen),'future_reads':False}
 (HERE/'causal_window_validation.json').write_text(json.dumps({'passed':True,'cases':results},indent=2))
except Exception as e:
 (HERE/'causal_window_failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e)[:400],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]},indent=2))
