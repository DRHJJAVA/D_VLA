from pathlib import Path
import json,hashlib,time,sys,traceback,importlib.util
import numpy as np
from scipy.spatial import cKDTree
import build_targets as bt
from sampling import fps
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/region_pose';B=P/'iter_025'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def verify(lock):
 for k,v in json.loads(Path(lock).read_text()).items():assert sha(R/k)==v,k
def lock(paths):return {str(p.relative_to(R)):sha(p) for p in paths}
def run(split):
 verify(B/'source_locked.json');verify(B/(split+'_input_locked.json'))
 if split=='val':
  f=json.loads((B/'pre_validation_freeze.json').read_text())
  assert f['sampling_summary']==sha(B/'sampling/train/summary.json')
 bt.main(split)
 cases=json.loads((P/'iter_024'/split/'summary.json').read_text())['cases']
 ts=json.loads((B/'targets'/split/'summary.json').read_text())['rows']
 out=B/'sampling'/split;out.mkdir(parents=True,exist_ok=False)
 rows=[]
 for c in cases:
  key=c['event']+'_query'+str(c['query'])
  z=np.load(P/'iter_024'/split/(key+'.npz'))
  y=np.load(B/'targets'/split/(key+'.npz'))
  query=z['query_xyz']; n=len(query);frames=z['selected_frames']
  assert np.array_equal(query,y['query_xyz'])
  # Sampling is completed using only current query geometry before reading target arrays.
  sel=fps(query,2048) if n>2048 else np.arange(n)
  sel2=fps(query.copy(),2048) if n>2048 else np.arange(n)
  assert np.array_equal(sel,sel2) and len(np.unique(sel))==len(sel)
  pp=[];nn=[];ee=[];pw=[];nw=[];ss=[]
  for j,t in enumerate(frames):
   t=int(t)
   l=np.load(P/'iter_024'/(split+'_loo')/(key+'_frame'+str(t)+'.npz'))
   Q=z['query_to_current_hand_'+str(t)];alts=l['loo_Q']
   assert np.array_equal(Q,l['nominal_Q'])
   positions=np.stack([bt.apply(query,q) for q in [Q,*alts]])
   spread=np.linalg.norm(positions-positions[0],axis=2).max(axis=0)
   q=z['finger_q_'+str(t)]
   inside1=bt.inside(positions,q,.001).all(axis=0)
   inside4=bt.inside(positions,q,.004).any(axis=0)
   stable=(spread<=.002)&(len(alts)>=7)
   source=(z['frame']==t)&z['supported_candidate']
   dist=cKDTree(z['mapped_query_xyz'][source]).query(query)[0] if source.any() else np.full(n,np.inf)
   pos=(dist<=.003)&inside1&stable
   neg=z['query_visible_at_frames'][j]&(~inside4)&stable
   pp.append(pos);nn.append(neg);ee.append(inside4);ss.append(spread)
   pw.append(np.where(pos,np.exp(-spread/.003)*np.exp(-dist/.003),0))
   nw.append(np.where(neg,np.exp(-spread/.003),0))
  ps=np.sum(pp,axis=0);ns=np.sum(nn,axis=0)
  positive=ps>=2;negative=(ns>=2)&(~np.any(ee,axis=0))
  conflict=positive&negative;positive[conflict]=False;negative[conflict]=False
  known=positive|negative
  conf=np.where(positive,np.sum(pw,axis=0)/np.maximum(ps,1),np.where(negative,np.sum(nw,axis=0)/np.maximum(ns,1),0)).astype(np.float32)
  assert np.array_equal(y['target'].astype(bool),positive)
  assert np.array_equal(y['known'].astype(bool),known)
  assert np.allclose(y['confidence'],conf,rtol=1e-6,atol=1e-7)
  assert (conf[~known]==0).all() and not (positive&negative).any()
  ids=np.arange(n,dtype=np.int64);pixel=z['query_pixel']
  assert len(np.unique(pixel,axis=0))==n
  p=positive[sel];neg=negative[sel]
  eligible=int(p.sum())>=3 and int(neg.sum())>=16
  np.savez_compressed(out/(key+'.npz'),source_index=ids,selected_index=sel,query_xyz=query,query_pixel=pixel,query_rgb=z['query_rgb'],sample_xyz=query[sel],sample_pixel=pixel[sel],target=y['target'][sel],known=y['known'][sel],confidence=y['confidence'][sel])
  row=dict(episode=c['id'],event=c['event'],query=c['query'],selected_frames=frames.tolist(),full_points=n,full_positive=int(positive.sum()),full_negative=int(negative.sum()),full_unknown=int((~known).sum()),sample_points=len(sel),sample_positive_sources=int(p.sum()),sample_negative_sources=int(neg.sum()),sample_unknown=int((~known[sel]).sum()),supported_1mm_voxels=c['supported_1mm_voxels'],eligible=eligible,distinct_sources_are_query_pixels_not_independent_grasps=True,independent_target_reconstruction=True,causal_fps_repeat_exact=True,confidence_max_difference=float(np.max(abs(y['confidence']-conf))))
  rows.append(row);dump(out/(key+'.json'),row)
 result=dict(split=split,rows=rows,eligible_events=sum(r['eligible'] for r in rows),candidate_events=len(rows),official_episode_count=20 if split=='train' else 4,training_performed=False,not_physical_contact_truth=True)
 dump(out/'summary.json',result);dump(out/'locked.json',lock(list(out.glob('*.npz'))+list(out.glob('*.json'))))
 verify(B/'source_locked.json');verify(B/(split+'_input_locked.json'))
 dump(B/'audits'/(split+'_completion.json'),dict(time=time.time(),source_and_input_hashes_unchanged=True,independent_target_reconstruction=True,causal_sampling=True,events=len(rows),eligible_events=result['eligible_events']))
 if split=='train':
  dump(B/'pre_validation_freeze.json',dict(time=time.time(),train_summary=sha(B/'targets/train/summary.json'),train_lock=sha(B/'targets/train/locked.json'),source_lock=sha(B/'source_locked.json'),input_lock=sha(B/'train_input_locked.json'),sampling_summary=sha(out/'summary.json'),sampling_lock=sha(out/'locked.json')))
 dump(B/'stage_state.json',dict(stage=split+'_complete',time=time.time(),scoring_training=False))
 print(json.dumps(dict(split=split,eligible_events=result['eligible_events'],events=len(rows))))
if __name__=='__main__':
 try:run(sys.argv[1])
 except Exception as e:
  dump(B/('failure_'+sys.argv[1]+'.json'),dict(error_type=type(e).__name__,message=str(e),time=time.time()))
  with (B/('failure_'+sys.argv[1]+'.log')).open('w') as f:traceback.print_exc(file=f)
  print('PHASE_FAILED',type(e).__name__);sys.exit(1)
