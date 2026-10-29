from pathlib import Path
import ast,copy,sys,os,json,hashlib,time
import numpy as np,cv2,h5py
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';O=B/'iter_023'
os.environ['MUJOCO_GL']='egl';sys.path.insert(0,str(P/'src/v3'))
from stage2_robot_only_depth import RobotDepth
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def checklock(p,root):
 d=json.loads(p.read_text())
 for rel,h in d.items():
  q=Path(rel);q=q if q.is_absolute() else root/q
  assert sha(q)==h,str(q)
 return len(d)
assert json.loads((B/'iter_022/pre_validation_freeze.json').read_text()).get('eligible',True)
n22=checklock(B/'iter_022/completed_locked.json',P)
n11=checklock(B/'iter_011/val/locked.json',B/'iter_011/val')
cases=json.loads((B/'iter_011/val/summary.json').read_text())['cases']
assert {c['id'] for c in cases}=={'alphabet_soup_demo_1','porcelain_mug_demo_2','white_yellow_mug_demo_1'}
runtime=[]
for c in cases:
 d={k:c[k] for k in ['id','event','split','query','source_fusion','anchor_frame','selected_held_frames','early_support_start','original_failure']}
 d['bridges']=[{'frame':int(f)} for f in c['selected_held_frames']];runtime.append(d)
save(O/'cases_runtime.json',runtime)
src=B/'iter_011/src/extend_cotracker.py';tree=ast.parse(src.read_text())
main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
loop=next(x for x in main.body if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='case')
selected=[copy.deepcopy(x) for x in loop.body if isinstance(x,ast.With) and x.lineno in [73,74,75]]
selected[-1].body=[x for x in selected[-1].body if x.lineno<=80]
assert len(selected)==3
code=compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(src),'exec')
refs={x['id']:x for x in json.loads((B/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
paths=[src,B/'iter_022/pre_validation_freeze.json',B/'iter_022/comparison_summary.json',B/'iter_022/completed_locked.json',B/'iter_011/val/summary.json',B/'iter_011/val/locked.json',B/'iter_008/val/summary.json',B/'iter_003/evidence/base/protocol.json',P/'src/v3/stage2_robot_only_depth.py',P/'src/v3/stage2_probe_motion_fingerpads.py',B/'iter_010/weights/scaled_offline.pth']
for rel in json.loads((B/'iter_011/val/locked.json').read_text()):paths.append(B/'iter_011/val'/rel)
for c in cases:
 ref=refs[c['id']]
 paths.extend([Path(ref['component']),Path(ref['rgbd']),B/'iter_003/evidence/motion'/(c['id']+'.npz'),Path(c['source_fusion'])])
paths=list(dict.fromkeys(paths));save(O/'input_locked.json',{str(p):sha(p) for p in paths})
rows=[];slices={}
for c in cases:
 ident=c['id'];q=int(c['query']);ref=refs[ident]
 e=dict(np=np,cv2=cv2,h5py=h5py,RobotDepth=RobotDepth,OLD=B/'iter_003',ident=ident,q=q,ref=ref)
 exec(code,e)
 if hasattr(e.get('robot'),'close'):e['robot'].close()
 mask=e['mask'];a=e['features'];original=np.empty((0,2),np.float32) if a is None else a.reshape(-1,2).astype(np.float32)
 oldpath=B/'iter_011/val/trajectories'/(c['event']+'_query'+str(q)+'.npz')
 old=np.load(oldpath);assert np.array_equal(original,old['initial_uv'])
 v,u=np.where(mask>0);cells=sorted(set(zip((v//8).tolist(),(u//8).tolist())));points=list(original.copy())
 for cy,cx in cells:
  if len(points)>=300:break
  ix=np.where((v//8==cy)&(u//8==cx))[0]
  ds=(u[ix]-(cx*8+3.5))**2+(v[ix]-(cy*8+3.5))**2
  k=ix[np.lexsort((u[ix],v[ix],ds))[0]]
  point=np.array([u[k],v[k]],np.float32)
  if len(points) and np.min(np.linalg.norm(np.asarray(points)-point,axis=1))<3:continue
  points.append(point)
 points=np.asarray(points,np.float32).reshape(-1,2)
 assert np.array_equal(points[:len(original)],original)
 np.savez_compressed(O/'seeds'/(ident+'.npz'),mask=mask,original_uv=original,initial_uv=points,source_is_grid=np.arange(len(points))>=len(original),query_rgb=e['rgb'])
 with h5py.File(ref['rgbd'],'r') as h:
  g=h['agentview']
  for name,frame in [('depth_m',0),('depth_m',q),('rgb',q)]:
   ar=g[name][frame];slices[ident+':'+name+':'+str(frame)]={'shape':list(ar.shape),'sha256':ah(ar)}
  for name in ['K','camera_to_world']:
   ar=g[name][()];slices[ident+':'+name]={'shape':list(ar.shape),'sha256':ah(ar)}
 rows.append({'episode':ident,'query':q,'original':len(original),'grid':len(points)-len(original),'total':len(points),'mask_pixels':int((mask>0).sum()),'prefix_exact':True,'fixed_endpoints':c['selected_held_frames']})
save(O/'input_slices_locked.json',slices)
save(O/'seed_locked.json',{str(p.relative_to(P)):sha(p) for p in sorted((O/'seeds').glob('*.npz'))})
assert all(sha(p)==h for p,h in json.loads((O/'input_locked.json').read_text()).items())
save(O/'audits/preflight.json',{'passed':True,'cases':rows,'prior_022_lock_files':n22,'prior_011_val_lock_files':n11,'query_mask_from_same_original_operations':True,'seed_lock_before_forward':True,'time':time.time(),'training_or_targets':False})
save(O/'stage_state.json',{'stage':'prepared','time':time.time()})
print(json.dumps({'prepared':True,'cases':rows}))
