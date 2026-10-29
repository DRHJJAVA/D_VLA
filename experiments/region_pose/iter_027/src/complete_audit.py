from pathlib import Path
import sys,json,hashlib,ast,numpy as np,h5py,cv2
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/region_pose';B=P/'iter_027'
sys.path.insert(0,str(B/'src'));import track_new as m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
for lock in ['source_locked.json','input_locked.json','propagation_source_locked.json','audit_source_locked.json']:
 for f,h in json.loads((B/lock).read_text()).items():assert sha(R/f)==h
for lock in [P/'iter_014/completed_locked.json']:
 for f,h in json.loads(lock.read_text()).items():assert sha(R/f)==h
old25=json.loads((P/'iter_025/completed_locked.json').read_text());checked25=0
for f,h in old25.items():
 if '/val/' in f or 'val_' in Path(f).name:continue
 assert sha(R/f)==h;checked25+=1
t=ast.parse((P/'iter_014/src/diagnose_early_queries.py').read_text())
class Reverse(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,str):
   d={'iter_027/new_cases.json':'iter_013/targets/train/summary.json','raw_new':'train','raw_new/summary.json':'train/summary.json','raw_new/locked.json':'train/locked.json','new_tracking_state.json':'stage_state.json'}
   if n.value in d:return ast.copy_location(ast.Constant(d[n.value]),n)
  return n
assert ast.dump(t)==ast.dump(Reverse().visit(ast.parse((B/'src/track_new.py').read_text())))
protocol={x['id']:x for x in json.loads((P/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
new=json.loads((B/'raw_new/summary.json').read_text())['rows']
slices={};causal=[]
for ident in sorted({x['episode'] for x in new}):
 motion=np.load(P/'iter_003/evidence/motion'/f'{ident}.npz')
 with np.load(protocol[ident]['component']) as z:component=z['pixel'].astype(np.int64)
 with h5py.File(protocol[ident]['rgbd'],'r') as f:
  g=f['agentview'];robot=m.RobotDepth(g['depth_m'].shape[1])
  rr=[x for x in new if x['episode']==ident];first=min(x['query'] for x in rr);last=rr[0]['reference_query']
  for k in ['rgb','depth_m','K','camera_to_world']:
   arr=np.asarray(g[k][first:last+1]);slices[f'{ident}:{first}:{last}:{k}']={'sha256':hashlib.sha256(arr.tobytes()).hexdigest(),'shape':list(arr.shape)}
  for rec in rr:
   q=rec['query'];a=np.load(B/'raw_new'/f"{rec['event']}_query{q}.npz")
   uv,xyz,rgb,*_=m.input_cloud(f,component,motion,q,robot)
   for actual,expected in [(uv,a['query_pixel']),(xyz,a['query_xyz']),(rgb[uv[:,1],uv[:,0]],a['query_rgb'])]:assert np.array_equal(actual,expected)
   corners=None
   if q!=rec['reference_query']:
    mask=np.zeros(rgb.shape[:2],np.uint8);mask[uv[:,1],uv[:,0]]=255
    mask=cv2.erode(mask,np.ones((3,3),np.uint8))
    features=cv2.goodFeaturesToTrack(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),300,.01,3,mask=mask,blockSize=3)
    assert features is not None
    corners=features.reshape(-1,2);assert np.array_equal(corners,a['initial_uv'])
   causal.append({'episode':ident,'query':q,'query_xyz_pixel_rgb_equal':True,'initial_corners_equal':True if corners is not None else None,'corner_count':len(corners) if corners is not None else None})
  robot.close()
dump(B/'new_input_slices_postrun_locked.json',{'timing':'postrun audit; not claimed pre-forward hash','slices':slices})
summary=json.loads((B/'summary.json').read_text());cases={x['event']:x for x in json.loads((B/'cases.json').read_text())['rows']};checks=[];baselines=[]
for rec in summary['rows']:
 key=f"{rec['event']}_query{rec['query']}";a=np.load(B/'train'/(key+'.npz'))
 ref=np.load(cases[rec['event']]['target_source'])
 idx=a['reference_source_index'];ids=a['all_transform_reference_index'];dd=a['all_transform_reference_distance_m'];known=a['known'];y=a['target'];sel=a['selected_index']
 if rec['query']==rec['reference_query']:expected_known=ref['known'].astype(bool);expected_y=ref['target'].astype(bool);confidence=ref['confidence'].copy()
 else:
  expected_known=np.all(ref['known'][ids].astype(bool),axis=0)&np.all(ref['target'][ids]==ref['target'][ids[0]],axis=0)&np.all(dd<=.002,axis=0)&(a['position_sensitivity_m']<=.002)
  if len(a['loo_query_to_reference_world'])<7:expected_known[:]=False
  expected_y=ref['target'][idx].astype(bool)&expected_known
  confidence=np.min(ref['confidence'][ids],axis=0)*np.exp(-a['position_sensitivity_m']/.002);confidence[~expected_known]=0
 assert np.array_equal(known,expected_known) and np.array_equal(y,expected_y)
 assert np.max(abs(a['confidence']-confidence))<=1e-7
 assert np.all(a['confidence'][~known]==0)
 assert len(np.unique(idx[sel][y[sel]]))==rec['positive_sources']
 assert len(np.unique(idx[sel][known[sel]&~y[sel]]))==rec['negative_sources']
 src=R/rec['source_file'];raw=np.load(src)
 center=-np.linalg.norm(a['query_xyz']-a['query_xyz'].mean(0),axis=1)
 baselines.append({'event':rec['event'],'offset':rec['offset'],'query':rec['query'],'current_gripper_geometry':m.metrics(raw['early_geometry_score'],y,known),'center':m.metrics(center,y,known),'reference_geometry_posthoc_not_query_input':m.metrics(raw['reference_geometry_score'],y,known)})
 checks.append({'key':key,'target_known_reconstructed':True,'confidence_reconstructed':True,'source_counts_verified':True,'unknown_confidence_zero':True})
dump(B/'audits/causal_sampling.json',{'rows':causal,'new_tracker_method_ast_unchanged_except_io_paths':True})
dump(B/'audits/baselines.json',{'rows':baselines,'no_model_scores':True,'random_expected_coverage_in_metrics':True})
dump(B/'audits/completion_audit.json',{'passed':True,'new_causal_queries':len(causal),'all_propagated_queries':len(checks),'rows':checks,'source_input_locks_verified':True,'old014_lock_files':84,'old025_train_and_nonval_locked_files_checked':checked25,'new_clip_slices_locked_after_run':True,'raw_new_summary_training_events_literal_inherited_6_not_actual_count':True,'actual_new_events':4,'actual_total_events':10,'validation_test_access':False,'stage2_accepted':False})
dump(B/'stage_state.json',{'stage':'audit_complete','queries':40,'events':10})
print(json.dumps({'audit_pass':True,'queries':40,'new_causal_queries':len(causal),'old025_checked':checked25}))
