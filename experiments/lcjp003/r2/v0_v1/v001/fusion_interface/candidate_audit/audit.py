import os,sys,json,time,hashlib,traceback,fcntl
from pathlib import Path
from collections import Counter
ROOT=Path('/mnt/nvme1/drh/D_VLA')
HERE=Path(__file__).resolve().parent
FI=HERE.parent;RUN=FI.parent;M2M=RUN/'segmentation_diagnostic/m2m_all'
os.chdir(ROOT);sys.path.insert(0,str(FI));sys.path.insert(0,str(ROOT/'experiments/lcjp003/src'))
import numpy as np,h5py,torch
import data as p0
from fusion_module import LEGACY
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def fhash(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 p=HERE/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)

def make_prefix():
 import ast,copy
 tree=ast.parse((RUN/'v0_corrected_check.py').read_text())
 original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_data')
 prefix=[n for n in original.body if n.lineno<78]
 fn=copy.deepcopy(original);fn.name='input_prefix';fn.body=copy.deepcopy(prefix)+[ast.parse('def f():\n return obj, np.asarray(hist,dtype=np.float32), hist_idx, parity').body[0].body[0]]
 assert [ast.dump(n,include_attributes=False) for n in fn.body[:-1]]==[ast.dump(n,include_attributes=False) for n in prefix]
 ns=LEGACY.copy();exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<readonly_original_input_prefix>','exec'),ns)
 return ns['input_prefix']
INPUT_PREFIX=make_prefix()
def verify_prefix(dataset,entry,mu,sd):
 obs=ROOT/'experiments/contact_joint_object/cache/observations'/(entry['id']+'.h5');derived=HERE/'case000_observations.h5'
 original_file=h5py.File;original_factory=p0.P0Dataset
 def redirected(path,*args,**kw):
  if isinstance(path,(str,Path)) and Path(path).resolve()==obs.resolve():
   assert (args[0] if args else kw.get('mode','r'))=='r';return original_file(derived,*args,**kw)
  return original_file(path,*args,**kw)
 h5py.File=redirected;p0.P0Dataset=lambda:dataset;dataset._cache.clear();LEGACY['save']=lambda n,z:None
 try:
  a,ha,_,_,ia=LEGACY['make_data'](entry,0,mu,sd)
  b,hb,ib,_=INPUT_PREFIX(entry,0,mu,sd)
 finally:h5py.File=original_file;p0.P0Dataset=original_factory
 assert len(a)==len(b) and np.array_equal(ha,hb) and np.array_equal(ia,ib)
 for x,y in zip(a,b):
  assert x[0]==y[0] and np.array_equal(x[1],y[1]) and np.array_equal(x[2],y[2])
 result={'status':'passed','case':0,'candidate_count':len(a),'point_arrays_exact':True,'masks_exact':True,'history_exact':True,'history_indices_exact':True,'max_abs_difference':0.0,'optimizer_updates':0,'scope':'original input prefix exact AST body preserved; future action-label tail omitted only for this input audit'}
 save('input_prefix_parity.json',result)

def one(ci,case,recs,dataset,entry,mu,sd):
 t0=time.perf_counter();eid=case['id'];ti=case['frame'];base=ROOT/'experiments/contact_joint_object/cache'
 obs=base/'observations'/(eid+'.h5');rgb=base/'rgbd'/(eid+'.h5')
 derived=HERE/f'case{ci:03d}_observations.h5';rows=[];views=[];masks=[];checks=[]
 with h5py.File(obs,'r') as a,h5py.File(derived,'w') as b,h5py.File(rgb,'r') as r:
  fg=b.create_group('frames/'+str(ti))
  for vi,cn in enumerate(['agentview','robot0_eye_in_hand']):
   rec=recs[vi];assert rec['episode_id']==eid and rec['frame']==ti and rec['camera']==vi
   with np.load(ROOT/rec['union_npz']) as z:mm=z['masks']
   with np.load(RUN/f'coverage_case{ci}_view{vi}_grid32.npz') as z:old=z['masks']
   with np.load(ROOT/rec['m2m_npz']) as z:new=z['masks']
   assert mm.dtype==bool and mm.shape[1:]==(256,256)
   assert np.array_equal(mm[:len(old)],old) and np.array_equal(mm[len(old):],new)
   assert len(mm)==len(rec['union_sources'])
   dep=r[cn+'/depth_m'][ti];rob=r[cn+'/robot_mask'][ti].astype(bool)
   raw=r[cn+'/rgb'][ti];K=r[cn+'/K'][ti];C=r[cn+'/camera_to_world'][ti]
   assert int(r['frame_indices'][ti])==ti
   assert ahash(raw)==rec['raw_rgb_sha256']==rec['page_raw_sha256']
   assert np.isfinite(K).all() and np.isfinite(C).all() and K.shape==(3,3) and C.shape==(4,4)
   feat=a[f'frames/{ti}/features_{vi}'][:];assert feat.shape==(32,32,384)
   fg.create_dataset('features_'+str(vi),data=feat)
   maskarr=[];dups=0
   for j,m in enumerate(mm):
    source=rec['union_sources'][j];assert source['index']==j
    prev=source.get('duplicate_of')
    if prev is not None:assert 0<=prev<j and np.array_equal(m,mm[prev]);dups+=1
    gi=len(rows);vv,uu=np.nonzero(m);uv=np.stack([uu,vv],1).astype(np.int64)
    gg=fg.create_group('object_'+str(gi));gg.create_dataset('uv',data=uv);gg.create_dataset('camera',data=np.full(len(uv),vi,np.int8));gg.create_dataset('mask_'+str(vi),data=np.packbits(m))
    valid=m&np.isfinite(dep)&(dep>0)&~rob
    rows.append({'global_candidate':gi,'view':vi,'view_candidate':j,'source':source['source'],'source_index':source['source_index'],'duplicate_of':prev,'mask_pixels':int(m.sum()),'finite_positive_depth_pixels':int((m&np.isfinite(dep)&(dep>0)).sum()),'robot_overlap_pixels':int((m&rob).sum()),'nonrobot_valid_depth_pixels':int(valid.sum())})
    masks.append(m);maskarr.append(m)
   views.append({'view':vi,'frame_index':int(r['frame_indices'][ti]),'raw_rgb_exact':True,'K_finite':True,'C_finite':True,'K_determinant':float(np.linalg.det(K)),'C_rotation_orthogonality_max':float(np.abs(C[:3,:3].T@C[:3,:3]-np.eye(3)).max()),'RGB_shape':list(raw.shape),'depth_shape':list(dep.shape),'raw_rgb_sha256':ahash(raw),'depth_sha256':ahash(dep),'K_sha256':ahash(K),'C_sha256':ahash(C),'robot_mask_sha256':ahash(rob),'dense_features_sha256':ahash(feat),'union_masks_sha256':ahash(mm),'union_npz_sha256':fhash(ROOT/rec['union_npz']),'baseline_retained_exact':True,'new_retained_exact':True,'baseline_count':len(old),'new_count':len(new),'union_count':len(mm),'exact_duplicates':dups})
 original_file=h5py.File;original_factory=p0.P0Dataset
 read_log=[]
 def redirected(path,*args,**kw):
  mode=args[0] if args else kw.get('mode','r')
  if isinstance(path,(str,Path)) and Path(path).resolve()==obs.resolve():
   assert mode=='r';return original_file(derived,*args,**kw)
  return original_file(path,*args,**kw)
 dataset._cache.clear();dataset.read_log=[]
 p0.P0Dataset=lambda:dataset;h5py.File=redirected
 captures={};LEGACY['save']=lambda n,z:captures.update({n:z})
 try:
  obj,h,hidx,parity=INPUT_PREFIX(entry,ti,mu,sd)
  sf=dataset.frame(eid,ti)
 finally:h5py.File=original_file;p0.P0Dataset=original_factory
 assert max(hidx)<=ti and len(hidx)==16
 assert sf['point_valid'].shape==(2048,)
 out={o[0]:o for o in obj};dropped=[];invalid=[]
 with original_file(rgb,'r') as r:
  for row,m in zip(rows,masks):
   vi=row['view'];gi=row['global_candidate'];oid=gi*2+vi
   own=(sf['object_id']==gi)&sf['point_valid'];camera=(sf['camera_id']==vi)&sf['point_valid']
   pix=np.rint(sf['uv']).astype(np.int64).clip(0,255)
   support=camera&m[pix[:,1],pix[:,0]]
   xy=sf['depth_uv'][support].astype(np.int64);unique=np.unique(xy,axis=0)
   rob=r[(['agentview','robot0_eye_in_hand'][vi])+'/robot_mask'][ti].astype(bool)
   row.update({'p0_object_id_allocated_points':int(np.sum(sf['object_id']==gi)),'p0_object_id_valid_points':int(own.sum()),'actual_mask_support_points':int(support.sum()),'actual_distinct_depth_pixels':len(unique),'supported_robot_pixels':int(rob[unique[:,1],unique[:,0]].sum()) if len(unique) else 0,'assembled_id':oid,'assembled':oid in out})
   if oid in out:
    x=out[oid][1];v=out[oid][2]
    assert x.shape==(32,32,396) and v.shape==(32,32)
    row['assembled_valid_entries']=int(v.sum());row['valid_features_all_finite']=bool(np.isfinite(x[v]).all());row['invalid_padding_max']=float(abs(x[~v]).max()) if (~v).any() else 0.0
    assert row['valid_features_all_finite'] and row['invalid_padding_max']==0
   else:
    row['assembled_valid_entries']=0
    if row['nonrobot_valid_depth_pixels']>0:dropped.append(gi)
    else:invalid.append(gi)
 point_valid=sf['point_valid'];assert np.isfinite(sf['xyz_base'][point_valid]).all()
 result={'case':ci,'episode':eid,'frame':ti,'status':'issues_found' if dropped else 'passed','optimizer_updates':0,'views':views,'candidate_count':len(rows),'assembled_count':len(obj),'candidates':rows,'valid_support_candidates_omitted':dropped,'empty_or_no_nonrobot_depth_candidates_omitted':invalid,'point_budget':2048,'actual_valid_p0_points':int(point_valid.sum()),'actual_padding_points':int((~point_valid).sum()),'history_indices':np.asarray(hidx).tolist(),'future_observation_read':False,'geometry_parity':{'world_backprojection_max_error':parity,'input_prefix_preserved':True,'action_labels_not_required':True},'input_protocol':p0.INPUT_PROTOCOL_VERSION,'derived_observation_sha256':fhash(derived),'elapsed_seconds':time.perf_counter()-t0,'no_gpu':True}
 save(f'case{ci:03d}.json',result);return result
def main():
 torch.set_num_threads(4);torch.manual_seed(20261027);np.random.seed(20261027)
 lock=open(HERE/'audit.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 sampling=json.loads((RUN/'v1_sampling_lock.json').read_text())['cases'];assert len(sampling)==100
 assert all(c['demo'] in ['demo_45','demo_46'] for c in sampling)
 entries=json.loads((ROOT/'experiments/contact_joint_object/manifests/episodes.json').read_text())['episodes']
 known={e['id']:e.copy() for e in entries if e['id'] in {c['id'] for c in sampling}}
 d=p0.P0Dataset();d.rows=known;d.actions={}
 for eid,e in known.items():
  e['source']=str(ROOT/e['raw']);assert e['split']=='dev'
  with h5py.File(e['source'],'r') as f:d.actions[eid]=f['data'][e['episode']]['actions'][:]
 n=json.loads((FI/'inherited_action_normalization.json').read_text());assert n['development_used'] is False
 mu=np.array(n['mean'],np.float32);sd=np.array(n['std'],np.float32)
 verify_prefix(d,known[sampling[0]['id']],mu,sd)
 completed={int(p.stem[4:]):json.loads(p.read_text()) for p in HERE.glob('case[0-9][0-9][0-9].json')};started=time.time()
 while len(completed)<100:
  data=json.loads((M2M/'results.json').read_text());by={}
  for r in data['records']:by.setdefault(r['case'],{})[r['camera']]=r
  for ci,recs in sorted(by.items()):
   if ci in completed or len(recs)!=2:continue
   completed[ci]=one(ci,sampling[ci],recs,d,known[sampling[ci]['id']],mu,sd)
   save('state.json',{'status':'running','completed_cases':len(completed),'total_cases':100,'completed_views':len(completed)*2,'optimizer_updates':0,'elapsed_seconds':time.time()-started,'current_case':ci})
  if len(completed)<100:time.sleep(10)
 allrows=[r for c in completed.values() for r in c['candidates']]
 issues=[{'case':c['case'],'episode':c['episode'],'frame':c['frame'],'candidate':r} for c in completed.values() for r in c['candidates'] if (r['nonrobot_valid_depth_pixels']>0 and not r['assembled']) or r['supported_robot_pixels']>0]
 counts={'cases':100,'views':200,'union_candidates':len(allrows),'baseline_candidates':sum(r['source']=='baseline32' for r in allrows),'new_candidates':sum(r['source']=='m2m' for r in allrows),'exact_duplicates':sum(r['duplicate_of'] is not None for r in allrows),'raw_robot_overlap_candidates':sum(r['robot_overlap_pixels']>0 for r in allrows),'raw_robot_overlap_pixels':sum(r['robot_overlap_pixels'] for r in allrows),'nonrobot_valid_depth_candidates':sum(r['nonrobot_valid_depth_pixels']>0 for r in allrows),'empty_or_invalid_candidates':sum(r['nonrobot_valid_depth_pixels']==0 for r in allrows),'assembled_candidates':sum(r['assembled'] for r in allrows),'valid_support_candidates_omitted':sum(r['nonrobot_valid_depth_pixels']>0 and not r['assembled'] for r in allrows),'supported_robot_pixels':sum(r['supported_robot_pixels'] for r in allrows),'candidate_count_min':min(c['candidate_count'] for c in completed.values()),'candidate_count_max':max(c['candidate_count'] for c in completed.values()),'actual_valid_points_min':min(c['actual_valid_p0_points'] for c in completed.values()),'actual_valid_points_max':max(c['actual_valid_p0_points'] for c in completed.values())}
 checks=[{'name':'fixed_100_cases_200_views','passed':True},{'name':'raw_RGB_matches_registered_page_pixels','passed':True},{'name':'original_baseline_and_new_masks_retained_exact','passed':True},{'name':'camera_and_depth_frame_indices_match','passed':True},{'name':'original_corrected_geometry_and_visibility_code_reused','passed':True},{'name':'padding_invalid_and_valid_features_finite','passed':True},{'name':'nonrobot_support_candidates_not_silently_omitted','passed':counts['valid_support_candidates_omitted']==0},{'name':'no_robot_pixels_in_valid_model_support','passed':counts['supported_robot_pixels']==0},{'name':'zero_optimizer_updates_and_no_GPU','passed':True}]
 result={'status':'passed' if all(x['passed'] for x in checks) else 'issues_found','optimizer_updates':0,'formal_training_started':False,'GPU_used':False,'scope':'真实新并集候选的像素登记、原CPU几何/部位输入组装、预算与padding；不判定候选物体完整性、语义识别或评分学习效果。','counts':counts,'checks':checks,'failures':issues,'cases':[{'case':ci,'report':f'case{ci:03d}.json','status':c['status'],'sha256':fhash(HERE/f'case{ci:03d}.json')} for ci,c in sorted(completed.items())],'source_hashes':{'sampling_lock':fhash(RUN/'v1_sampling_lock.json'),'m2m_results':fhash(M2M/'results.json'),'P0_data':fhash(ROOT/'experiments/lcjp003/src/data.py'),'V0_assembly':fhash(RUN/'v0_corrected_check.py'),'normalization':fhash(FI/'inherited_action_normalization.json'),'audit_script':fhash(__file__),'input_prefix_parity':fhash(HERE/'input_prefix_parity.json')},'elapsed_seconds':time.time()-started}
 save('input_audit.json',result);save('complete.json',{'status':result['status'],'completed_cases':100,'completed_views':200,'optimizer_updates':0,'input_audit_sha256':fhash(HERE/'input_audit.json')})
 save('state.json',{'status':'complete','completed_cases':100,'completed_views':200,'optimizer_updates':0})
 print(json.dumps({'status':result['status'],'counts':counts,'elapsed_seconds':result['elapsed_seconds']}))
try:main()
except Exception as e:
 save('failure.json',{'type':type(e).__name__,'message':str(e)[:250],'locations':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
 print(json.dumps({'status':'failed','type':type(e).__name__}));raise SystemExit(1)
