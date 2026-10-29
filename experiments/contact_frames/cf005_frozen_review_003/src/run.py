from pathlib import Path
import json,datetime,hashlib,subprocess,sys,traceback,os,importlib.util
import numpy as np,torch
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/contact_frames/cf005_frozen_review_003';C=R/'experiments/contact_frames/cf005';PY='/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python'
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def intervals(mask):
 ids=np.flatnonzero(mask);return [[int(x[0]),int(x[-1])] for x in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(x)]
def main():
 torch.set_num_threads(4)
 plan=json.loads((B/'plan.json').read_text());weights=json.loads((B/'weights_locked.json').read_text())
 assert all(sha(Path(p))==v for p,v in weights.items())
 files=[B/'src/run.py',B/'plan.json',C/'src/model_all_train_frames.py']
 for ep in plan['cases']:
  base=B/ep;cfg=json.loads((base/'config.json').read_text());files.extend([base/'config.json',base/'src/prepare.py',base/'src/common.py']+[R/cfg[k] for k in ['rgbd','component','motion','frozen_tracker']])
 lock={str(p):sha(p) for p in files};save(B/'input_source_locked.json',lock)
 spec=importlib.util.spec_from_file_location('cf005net',C/'src/model_all_train_frames.py');net=importlib.util.module_from_spec(spec);spec.loader.exec_module(net)
 rows=[]
 for ep in plan['cases']:
  base=B/ep
  save(B/'stage_state.json',{'stage':'preparing','episode':ep,'updated':now()})
  env=dict(os.environ,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
  with (base/'prepare.log').open('w') as out:
   result=subprocess.run([PY,str(base/'src/prepare.py')],stdout=out,stderr=subprocess.STDOUT,env=env)
  if result.returncode!=0:
   failure=json.loads((base/'failure.json').read_text()) if (base/'failure.json').exists() else {'type':'preparation_nonzero_exit','code':result.returncode}
   save(B/'preparation_failure.json',{'episode':ep,'type':failure.get('type',failure.get('error_type','unknown'))});raise RuntimeError('preparation failed; details retained remotely')
  d=dict(np.load(base/'artifacts/dataset.npz'));n=len(d['states']);ix=np.maximum(0,np.arange(n)[:,None]+np.arange(-8,1)[None,:]);allp=[]
  save(B/'stage_state.json',{'stage':'predicting','episode':ep,'updated':now()})
  for seed in [2026,2027,2028]:
   ck=torch.load(C/f'runs/all_train_frames/seed_{seed}/last.pt',map_location='cpu',weights_only=False)
   model=net.ContactFrameNet(50).cuda().eval();model.load_state_dict(ck['model'])
   st=(d['states']-np.asarray(ck['state_mean']))/np.asarray(ck['state_std']);prob=[]
   with torch.no_grad():
    for j in range(0,n,16):
     k=ix[j:j+16]
     z,_=model(torch.as_tensor(d['images'][k],device='cuda'),torch.as_tensor(d['points'][k],device='cuda'),torch.as_tensor(st[k],device='cuda',dtype=torch.float32))
     prob.extend(z.sigmoid().cpu().numpy().reshape(-1).tolist())
   allp.append(prob)
  p=np.asarray(allp,np.float32);lo=p.min(0);hi=p.max(0);rel=d['geometry_reliable']&d['state_valid']
  pos=(lo>=.8)&rel;neg=(hi<=.2)&rel&(d['distance']-2*d['sigma']>.006);unknown=~(pos|neg)
  assert np.isfinite(p).all()
  np.savez_compressed(base/'artifacts/frame_predictions.npz',probabilities=p,mean=p.mean(0),low=lo,high=hi,contact_candidate=pos,observed_separation_candidate=neg,unknown=unknown,context_indices=ix)
  row={'episode':ep,'frames':n,'candidate':int(pos.sum()),'separation':int(neg.sum()),'unknown':int(unknown.sum()),'candidate_intervals':intervals(pos),'separation_intervals':intervals(neg),'unknown_intervals':intervals(unknown),'raw_unanimous_high':int((lo>=.8).sum()),'geometry_available':int(rel.sum()),'new_training':False,'physical_success_claim':False}
  save(base/'inference_summary.json',row);rows.append(row)
 assert all(sha(Path(p))==v for p,v in weights.items());assert all(sha(Path(p))==v for p,v in lock.items())
 save(B/'summary.json',rows);save(B/'audits/completion.json',{'time':now(),'inputs_and_sources_unchanged':True,'frozen_weights_unchanged':True,'no_training':True,'no_threshold_change':True,'reserved_demos_opened':False})
 save(B/'stage_state.json',{'stage':'inference_complete','updated':now()})
if __name__=='__main__':
 try:main()
 except Exception as e:
  (B/'failure.log').write_text(traceback.format_exc());save(B/'failure.json',{'type':type(e).__name__,'message':str(e),'time':now()})
  save(B/'stage_state.json',{'stage':'failed','type':type(e).__name__,'updated':now()});sys.exit(1)
