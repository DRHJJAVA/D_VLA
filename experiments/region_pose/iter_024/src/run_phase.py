from pathlib import Path
import sys,json,hashlib,time,traceback,os
import numpy as np
import audit_regions as reg
import audit_loo as loo
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';O=B/'iter_024'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def verify(path,root):
 d=json.loads(path.read_text())
 for rel,h in d.items():
  p=Path(rel);p=p if p.is_absolute() else root/p
  assert sha(p)==h,str(p)
 return len(d)
def main(split):
 assert split in ['train','val']
 assert not (O/split/'summary.json').exists(),'completed split must not be rerun'
 verify(O/'source_locked.json',P)
 if split=='val':
  fr=json.loads((O/'pre_validation_freeze.json').read_text())
  for key,path in [('train_summary_sha256',O/'train/summary.json'),('train_loo_summary_sha256',O/'train_loo/summary.json'),('source_lock_sha256',O/'source_locked.json')]:assert fr[key]==sha(path)
  assert fr['train_audit_passed']
 source=B/('iter_022' if split=='train' else 'iter_023')
 oldcount=verify(source/'completed_locked.json',P)
 cases=[c for c in json.loads((source/split/'summary.json').read_text())['cases'] if c['valid']]
 assert len(cases)==(11 if split=='train' else 3)
 refs={r['id']:r for r in json.loads((B/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
 paths=[source/'completed_locked.json',source/split/'summary.json',source/split/'locked.json',B/'iter_003/evidence/base/protocol.json']
 for c in cases:
  ident=c['id'];assert c['split']==split
  stem=c['event']+'_query'+str(c['query']);ref=refs[ident]
  paths += [source/split/'correspondences'/(stem+'.npz'),source/split/'audits'/(stem+'.json'),source/'seeds'/(ident+'.npz'),B/'iter_003/evidence/motion'/(ident+'.npz'),Path(ref['component']),Path(ref['rgbd'])]
 save(O/(split+'_input_locked.json'),{str(p.relative_to(P)):sha(p) for p in dict.fromkeys(paths)})
 save(O/(split+'_cases_locked.json'),[{'episode':c['id'],'query':c['query'],'selected_frames':loo.expected_frames(c),'source':str(source)} for c in cases])
 # Numerical sanity uses synthetic coordinates solely to verify transform implementation, not training examples.
 rng=np.random.default_rng(2026);a=rng.normal(size=(20,3))*.04
 theta=.4;T=np.eye(4);T[:3,:3]=[[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]];T[:3,3]=[.1,-.2,.3]
 recovered=reg.rigid(a,reg.apply(a,T));err=float(np.max(np.abs(recovered-T)))
 assert err<1e-10
 assert np.max(np.abs(reg.apply(reg.apply(a,T),np.linalg.inv(T))-a))<1e-10
 save(O/'audits'/(split+'_preflight.json'),{'passed':True,'old_completed_lock_files':oldcount,'cases':len(cases),'rigid_recovery_max_abs':err,'source_locked_before_reading_frames':True,'no_tracking_or_scoring_training':True,'time':time.time()})
 reg.main(split)
 loo.main(split)
 verify(O/(split+'_input_locked.json'),P);verify(O/'source_locked.json',P)
 verify(O/split/'locked.json',O/split);verify(O/(split+'_loo')/'locked.json',O/(split+'_loo'))
 verify(source/'completed_locked.json',P)
 summary=json.loads((O/split/'summary.json').read_text());ls=json.loads((O/(split+'_loo')/'summary.json').read_text())
 assert ls['events']==len(cases)
 save(O/'audits'/(split+'_completion.json'),{'passed':True,'cases':len(cases),'window_prefix_verified':True,'query_cloud_exact':True,'source_pixel_world_geometry_roundtrip_checks':True,'all_locks_unchanged':True,'time':time.time()})
 if split=='train':
  save(O/'pre_validation_freeze.json',{'time':time.time(),'train_summary_sha256':sha(O/'train/summary.json'),'train_loo_summary_sha256':sha(O/'train_loo/summary.json'),'train_lock_sha256':sha(O/'train/locked.json'),'train_loo_lock_sha256':sha(O/'train_loo/locked.json'),'source_lock_sha256':sha(O/'source_locked.json'),'input_lock_sha256':sha(O/'train_input_locked.json'),'train_audit_passed':True,'no_validation_started':True})
 save(O/'stage_state.json',{'stage':split+'_complete','time':time.time(),'training_performed':False})
 print(json.dumps({'phase_complete':split,'cases':len(cases)}))
if __name__=='__main__':
 try:main(sys.argv[1])
 except Exception as e:
  save(O/'failure.json',{'type':type(e).__name__,'message':str(e),'time':time.time()})
  print('SAFE_FAILURE '+type(e).__name__+': '+str(e));sys.exit(1)
