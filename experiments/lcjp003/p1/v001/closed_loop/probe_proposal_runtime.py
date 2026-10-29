
from adapters import *
import traceback,datetime
try:
 eid='pick_up_the_bbq_sauce_and_place_it_in_the_basket__demo_0'
 results=[]
 for name,threads,seed in [('threads4_seed2026',4,2026),('threads2_seed2026',2,2026),('threads4_seed2026_repeat',4,2026)]:
  torch.set_num_threads(threads);torch.manual_seed(seed)
  work=HERE/('probe_'+name+'_'+str(time.time_ns()));work.mkdir()
  with h5py.File(OLD/'cache/rgbd'/(eid+'.h5'),'r') as raw:
   render=types.SimpleNamespace(work=work,eid=eid,out=raw)
   proc=LiveProposals(device='cuda');proc.frame(render,0)
  with h5py.File(OLD/'cache/observations'/(eid+'.h5'),'r') as a,h5py.File(work/'cache/observations'/(eid+'.h5'),'r') as b:
   ga=a['frames'][list(a['frames'])[0]];gb=b['frames'][list(b['frames'])[0]]
   detail=[];total=0
   for k in ga:
    if not k.startswith('object_'):continue
    for mk in ga[k]:
     if mk.startswith('mask_'):
      x=np.unpackbits(ga[k][mk][:]);y=np.unpackbits(gb[k][mk][:]);num=int((x!=y).sum());total+=num
      detail.append({'object':k,'mask':mk,'different_pixels':num})
   results.append({'name':name,'threads':threads,'seed':seed,'total_mask_xor_pixels':total,'detail':detail,'work':str(work)})
  del proc;torch.cuda.empty_cache()
  (HERE/'proposal_runtime_probe.json').write_text(json.dumps(results,indent=2))
except Exception as e:
 (HERE/'proposal_runtime_probe_failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e)[:400],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]},indent=2))
