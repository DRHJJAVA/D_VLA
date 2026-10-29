
from adapters import *
import traceback
try:
 torch.set_num_threads(4);torch.manual_seed(2026);torch.backends.cudnn.allow_tf32=True
 proc=LiveProposals(device='cuda');sam=proc.proc.sam
 eid='pick_up_the_bbq_sauce_and_place_it_in_the_basket__demo_0'
 with h5py.File(OLD/'cache/rgbd'/(eid+'.h5'),'r') as f:im=f['agentview/rgb'][0]
 orig=sam.predictor._transforms.postprocess_masks
 captures=[]
 def wrapped(masks,*args,**kwargs):
  before=masks.detach().cpu().numpy().copy()
  out=orig(masks,*args,**kwargs)
  after=out.detach().cpu().numpy().copy()
  captures.append((before,after))
  return out
 sam.predictor._transforms.postprocess_masks=wrapped
 baseline=None;results=[]
 for iteration in range(3):
  captures.clear()
  with torch.inference_mode():masks=sam.generate(im)
  current=[(x.copy(),y.copy()) for x,y in captures]
  row={'iteration':iteration,'postprocess_calls':len(current),'max_hole_area':sam.predictor._transforms.max_hole_area,'max_sprinkle_area':sam.predictor._transforms.max_sprinkle_area,'training_modules':sum(m.training for m in sam.predictor.model.modules())}
  if baseline:
   row.update(raw_logits_max_abs=max(float(np.max(np.abs(x-a))) for (x,y),(a,b) in zip(current,baseline)),postprocessed_max_abs=max(float(np.max(np.abs(y-b))) for (x,y),(a,b) in zip(current,baseline)),binary_mask_xor=sum(int(((y>0)!=(b>0)).sum()) for (x,y),(a,b) in zip(current,baseline)))
  else:baseline=current
  results.append(row)
 (HERE/'sam_operator_probe.json').write_text(json.dumps(results,indent=2))
except Exception as e:
 (HERE/'sam_operator_probe_failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e)[:500],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]},indent=2))
