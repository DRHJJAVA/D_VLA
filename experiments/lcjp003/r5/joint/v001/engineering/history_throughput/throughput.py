import sys,pathlib,json,time,traceback
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r5/joint/v001');E=B/'engineering/history_throughput'
sys.path.insert(0,str(B/'src'));sys.path.insert(0,str(E))
import torch,train,io_runtime as io
from optimized_history import install
def main():
 io.setup();ck=torch.load(E/'reference0500.pt',map_location='cpu',weights_only=False)
 m=io.build();m.load_state_dict(ck['model']);m.train()
 schedule=json.loads((B/'sampling_schedule.json').read_text());rows=[]
 for j,mode in enumerate([0,8,16,16,0,8,8,16,0]):
  io.restore_rng(ck['rng']);m.zero_grad(set_to_none=True)
  undo=install(train,mode) if mode else None
  times=[]
  for rec in schedule[500]:
   torch.cuda.synchronize();t=time.perf_counter()
   d=io.move(io.load_record(rec));prev=train.previous_encoding(m,rec);noise=torch.randn(8,16,6,device='cuda')
   loss,metrics,e,pred=train.objective(m,d,noise,prev);(loss/8).backward()
   torch.cuda.synchronize();times.append(time.perf_counter()-t)
   del d,prev,loss,e,pred
  if undo:undo()
  rows.append({'batch':mode,'seconds':sum(times),'time':time.time()})
  (E/'throughput.json').write_text(json.dumps({'done':False,'rows':rows},indent=2))
 (E/'throughput.json').write_text(json.dumps({'done':True,'rows':rows},indent=2))
try:main()
except Exception as e:
 (E/'throughput_failure.json').write_text(json.dumps({'type':type(e).__name__,'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]}))
