import sys,pathlib,json,traceback
P=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/p1/v001');A=P/'analysis_user_stop';A.mkdir(exist_ok=True)
sys.path.insert(0,str(P));sys.path.insert(0,str(P/'source'))
try:
 import torch,numpy as np
 from common import Dataset,batch,cpu_setup
 from model import LCJP003,ModelConfig
 cpu_setup();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ds=Dataset(cached=True);pairs=json.loads((P/'evaluation_lock.json').read_text())['pairs'];x=batch(ds,[pairs[0]])
 def schema(x):
  if isinstance(x,dict):return {k:schema(v) for k,v in x.items()}
  if isinstance(x,(list,tuple)):return [schema(v) for v in x]
  return {'shape':list(x.shape),'dtype':str(x.dtype)} if hasattr(x,'shape') else type(x).__name__
 m=LCJP003(ModelConfig()).cuda().eval();s=torch.load(P/'step3000.pt',map_location='cpu',weights_only=False);m.load_state_dict(s['model']);del s
 with torch.inference_mode():y=m(x[0],torch.randn(1,1,16,7,device='cuda'),return_diagnostics=True)
 result={'batch_schema':schema(x),'output_schema':schema(y)}
 (A/'score_probe_schema.json').write_text(json.dumps(result,indent=2))
except Exception as e:
 (A/'score_probe_schema_failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e)[:300],'frames':[{'file':t.filename,'line':t.lineno} for t in traceback.extract_tb(e.__traceback__)]}))
