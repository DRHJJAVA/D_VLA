
from adapters import *
import traceback
try:
 torch.set_num_threads(4);torch.manual_seed(2026)
 proc=LiveProposals(device='cuda')
 eid='pick_up_the_bbq_sauce_and_place_it_in_the_basket__demo_0'
 with h5py.File(OLD/'cache/rgbd'/(eid+'.h5'),'r') as f:im=f['agentview/rgb'][0]
 results=[];previous=None
 for deterministic in [False,False,False,True,True]:
  torch.use_deterministic_algorithms(deterministic)
  torch.backends.cudnn.allow_tf32=True
  with torch.inference_mode():masks=proc.proc.sam.generate(im)
  packed=[np.packbits(m['segmentation']) for m in masks]
  row={'deterministic_algorithms':deterministic,'masks':len(masks),'hashes':[hashlib.sha256(x.tobytes()).hexdigest() for x in packed]}
  if previous is not None and len(previous)==len(packed):
   row['xor_vs_previous']=sum(int(np.unpackbits(x^y).sum()) for x,y in zip(previous,packed))
  results.append(row);previous=packed
  (HERE/'sam_repeat_probe.json').write_text(json.dumps({'same_model_same_image':True,'runs':results},indent=2))
except Exception as e:
 (HERE/'sam_repeat_probe_failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e)[:600],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]},indent=2))
