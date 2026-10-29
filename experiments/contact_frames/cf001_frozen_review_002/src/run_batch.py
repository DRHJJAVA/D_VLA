import os,sys,json,traceback,subprocess,datetime,hashlib
from pathlib import Path
ROOT=Path('/mnt/nvme1/drh/D_VLA')
B=ROOT/'experiments/contact_frames/cf001_frozen_review_002'
PY='/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def infer(case):
 import numpy as np,torch
 sys.path.insert(0,str(ROOT/'experiments/contact_frames/cf001/src'))
 from model import ContactFrameNet
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.08)
 d=np.load(case/'artifacts/dataset.npz');n=len(d['images'])
 idx=np.array([np.maximum(np.arange(t-8,t+1),0) for t in range(n)])
 ps=[]
 for seed in [2026,2027,2028]:
  ck=torch.load(ROOT/f'experiments/contact_frames/cf001/artifacts/seed_{seed}/last.pt',map_location='cpu',weights_only=False)
  m=ContactFrameNet(ck['state_dim']).cuda().eval();m.load_state_dict(ck['model'])
  st=(d['states']-ck['state_mean'])/ck['state_std']
  out=[]
  with torch.no_grad():
   for t in range(n):
    a=idx[t]
    z,_=m(torch.from_numpy(d['images'][a][None]).cuda(),torch.from_numpy(d['points'][a][None]).cuda(),torch.from_numpy(st[a][None].astype('float32')).cuda())
    out.append(float(torch.sigmoid(z)[0]))
  ps.append(out);del m
 p=np.array(ps,dtype='float32');lo=p.min(0);hi=p.max(0)
 reliable=d['geometry_reliable']&d['state_valid']
 pos=(lo>=.8)&reliable
 neg=(hi<=.2)&reliable&((d['distance']-2*d['sigma'])>.006)
 unk=~(pos|neg)
 np.savez_compressed(case/'artifacts/frame_predictions.npz',probabilities=p,mean=p.mean(0),low=lo,high=hi,contact_candidate=pos,observed_separation_candidate=neg,unknown=unk,context_indices=idx)
 def runs(mask):
  ix=np.flatnonzero(mask);return [[int(x[0]),int(x[-1])] for x in np.split(ix,np.where(np.diff(ix)>1)[0]+1) if len(x)]
 result={'episode':case.name,'frames':n,'contact_candidate':int(pos.sum()),'separation_candidate':int(neg.sum()),'unknown':int(unk.sum()),'contact_ranges':runs(pos),'timestamp':now(),'weights_unchanged':True,'trained_on_this_video':False}
 save(case/'inference_summary.json',result)
 return result
def main():
 plan=json.loads((B/'plan.json').read_text());summ=[]
 for ep in plan['cases']:
  case=B/ep;save(B/'stage_state.json',{'stage':'preparing','episode':ep,'timestamp':now()})
  with (case/'prepare.log').open('w') as f:
   cp=subprocess.run([PY,str(case/'src/prepare.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  if cp.returncode:raise RuntimeError('prepare exit '+str(cp.returncode)+' '+ep)
  save(B/'stage_state.json',{'stage':'inference','episode':ep,'timestamp':now()})
  summ.append(infer(case));save(B/'summary.json',summ)
 locks=json.loads((B/'weights_locked.json').read_text())
 assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in locks.items())
 save(B/'stage_state.json',{'stage':'inference_complete','timestamp':now(),'cases':len(summ)})
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(B/'failure.json',{'type':type(e).__name__,'message':str(e),'stack':[{'file':x.filename,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(e.__traceback__)]})
  save(B/'stage_state.json',{'stage':'failed','timestamp':now()})
