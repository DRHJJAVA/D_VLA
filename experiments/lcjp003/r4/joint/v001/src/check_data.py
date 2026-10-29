import json,time
import torch,numpy as np,h5py
from data import RUN,ROOT,R3,CFG,legacy,load_sample,prepare_one,atomic_json,fail,digest,obs_file
def main():
 episodes=json.loads((RUN/'episodes.json').read_text());lookup={e['id']:e for e in episodes}
 recs=json.loads((RUN/'evaluation_windows.json').read_text());g,s=legacy()
 worst=0.;visible=0;qualities=[];tasks=set();phasecounts=[0,0];switchpositive=0;switchcount=0
 for rec in recs:
  e=lookup[rec['episode']];prepare_one((e,rec['frame']));q=load_sample(rec);d=q['inputs'];y=q['labels']
  assert not any(k in d for k in ['target','grip','future','part','mask','teacher']),'future_input'
  assert torch.isfinite(d['features']).all()
  assert tuple(d['features'].shape)==(2,32,32,384)
  qualities.append(q['metadata']['teacher_confidence']);visible+=int(y['mask'][:,0].sum()>0);tasks.add(e['task'])
  # Full depth unprojection must match actual cached RGB-D points, without object pose truth.
  with h5py.File(obs_file(e,rec['frame']),'r') as hf:
   fg=hf['frames/'+str(rec['frame'])]
   for name in fg:
    if not name.startswith('object_'):continue
    o=fg[name];cameras=o['camera'][:];uv=o['uv'][:];points=o['xyz'][:]
    for c in range(2):
     use=np.flatnonzero(cameras==c)[::127]
     if not len(use):continue
     pix=uv[use].astype(int);z=d['depth'][c,pix[:,1],pix[:,0]].numpy();k=d['K'][c].numpy();T=d['T'][c].numpy()
     cp=np.stack([(pix[:,0]+.375-k[0,2])*z/k[0,0],(pix[:,1]+.875-k[1,2])*z/k[1,1],z],-1)
     world=cp@T[:3,:3].T+T[:3,3]
     worst=max(worst,float(np.linalg.norm(world-points[use],axis=-1).max()))
 assert worst<1e-4,'unprojection_coordinate_mismatch'
 for e in episodes:
  if e['split']!='train':continue
  with h5py.File(ROOT/e['raw'],'r') as h:a=h['data/'+e['episode']+'/actions'][:,6]>0
  phasecounts[0]+=int((~a).sum());phasecounts[1]+=int(a.sum());switchpositive+=int((a[1:]!=a[:-1]).sum());switchcount+=len(a)-1
 inv=np.asarray(phasecounts,dtype=float);weights=inv.sum()/(2*inv)
 cfg=json.loads((RUN/'config.json').read_text());cfg['grip_class_weights']=weights.tolist();cfg['switch_positive_weight']=float(min(20,max(1,(switchcount-switchpositive)/max(switchpositive,1))))
 (RUN/'config.json').write_text(json.dumps(cfg,indent=2))
 atomic_json('checks/input_contract.json',{'passed':True,'time':time.time(),'fixed_windows':len(recs),'tasks':len(tasks),'visible_teacher_windows':visible,
  'teacher_confidence_mean':float(np.mean(qualities)),'teacher_confidence_min':float(np.min(qualities)),'depth_unprojection_max_m':worst,
  'heldout_read':False,'class_counts_training_only':phasecounts,'switch_positive':switchpositive,'switch_count':switchcount,
  'input_label_separation':True,'teacher_accuracy_claimed':False})
if __name__=='__main__':
 try:main()
 except Exception as ex:fail('data_contract',ex);raise SystemExit(1)
