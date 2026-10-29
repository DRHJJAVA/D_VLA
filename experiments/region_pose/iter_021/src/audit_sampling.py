from pathlib import Path
import ast,copy,sys,os,json,hashlib,time
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,cv2,h5py
P=Path('/mnt/nvme1/drh/D_VLA');O=P/'experiments/region_pose/iter_021'
sys.path.insert(0,str(P/'src/v3'))
from stage2_robot_only_depth import RobotDepth
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
src=P/'experiments/region_pose/iter_011/src/extend_cotracker.py'
tree=ast.parse(src.read_text())
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
loop=next(n for n in main.body if isinstance(n,ast.For) and getattr(n.target,'id','')=='case')
selected=copy.deepcopy([n for n in loop.body if 73<=n.lineno<=75])
assert len(selected)==3 and all(isinstance(n,ast.With) for n in selected)
selected[-1].body=[n for n in selected[-1].body if n.lineno<=80]
assert not any(isinstance(n,ast.Call) and getattr(n.func,'id','') in ['model','CoTrackerPredictor'] for x in selected for n in ast.walk(x))
env={'Path':Path,'sys':sys,'np':np,'cv2':cv2,'h5py':h5py,'RobotDepth':RobotDepth,'__builtins__':__builtins__}
sys.argv=['audit','train']
for n in tree.body:
 if isinstance(n,ast.Assign) and n.lineno==5:
  exec(compile(ast.Module(body=[n],type_ignores=[]),str(src),'exec'),env)
# original preparation only; snapshots added after original mask assignments
def snap_mask(mask,label):
 snaps.append((label,np.array(mask,copy=True)))
newbody=[]
for n in selected[-1].body:
 newbody.append(n)
 if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='mask' or isinstance(t,ast.Subscript) and getattr(t.value,'id','')=='mask' for t in n.targets):
  er=any(isinstance(a,ast.Call) and getattr(a.func,'attr','')=='erode' for a in ast.walk(n))
  call=ast.Expr(value=ast.Call(func=ast.Name(id='snap_mask',ctx=ast.Load()),args=[ast.Name(id='mask',ctx=ast.Load()),ast.Constant(value='eroded' if er else 'mask_step')],keywords=[]))
  newbody.append(call)
selected[-1].body=newbody
compiled=compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(src),'exec')
validnode=next(n for n in selected[-1].body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='valid' for t in n.targets))
def leaves(n):
 if isinstance(n,ast.BinOp) and isinstance(n.op,ast.BitAnd):return leaves(n.left)+leaves(n.right)
 if isinstance(n,ast.BoolOp) and isinstance(n.op,ast.And):return sum([leaves(v) for v in n.values],[])
 return [n]
terms=leaves(validnode.value)
termmeta=[]
for n in terms:
 names={x.id for x in ast.walk(n) if isinstance(x,ast.Name)}
 label='robot_depth' if 'rd' in names else 'initial_current_depth_consistency' if 'd0' in names else 'current_depth_validity'
 termmeta.append({'label':label,'expression_remote_only':ast.unparse(n)})
save(O/'audits/original_operation_metadata.json',{'original_script':str(src),'sha256':sha(src),'selected_line_range':[73,80],
 'valid_terms':termmeta,'no_tracker_constructor_or_forward':True})
cases=read(P/'experiments/region_pose/iter_011/train/summary.json')['cases']
train=read(P/'data/v3/manifests/stage2_episode_split.json')['splits']['train']
protocol=read(P/'experiments/region_pose/iter_003/evidence/base/protocol.json')
refs={r['id']:r for r in protocol['episodes'] if r['split']=='train'}
assert len(cases)==16 and all(r['id'] in train for r in cases)
paths={src,P/'src/v3/stage2_robot_only_depth.py',P/'src/v3/stage2_probe_motion_fingerpads.py',
 P/'experiments/region_pose/iter_011/train/summary.json',P/'experiments/region_pose/iter_003/evidence/base/protocol.json'}
for r in cases:
 ep=r['id'];paths.add(Path(refs[ep]['component']))
 paths.add(P/f'experiments/region_pose/iter_003/evidence/motion/{ep}.npz')
 p=P/'experiments/region_pose/iter_011/train/trajectories'/(r['event']+'_query'+str(r['query'])+'.npz')
 if p.exists():paths.add(p)
save(O/'input_locked.json',{str(p):sha(p) for p in sorted(paths)})
save(O/'source_locked.json',{str(p):sha(p) for p in [O/'设计方案.md',O/'src/audit_sampling.py',O/'config.json']})
# Hash precisely the permitted HDF5 slices before analyzing; do not hash/read future image datasets.
slice_lock={}
for r in cases:
 ref=refs[r['id']];q=r['query']
 with h5py.File(ref['rgbd'],'r') as f:
  g=f['agentview'];slices={}
  for key,ix in [('rgb',q),('depth_m',q),('depth_m',0),('K',q),('camera_to_world',q)]:
   a=g[key][ix];slices[key+':'+str(ix)]={'sha256':ah(a),'shape':list(np.shape(a))}
  slice_lock[r['id']]={'path':ref['rgbd'],'slices':slices,'query':q}
save(O/'input_slices_locked.json',slice_lock)
def stats(mask):
 v,u=np.where(mask>0)
 return {'pixels':len(u),'grid8_cells':len(np.unique(np.stack([u//8,v//8],1),axis=0)) if len(u) else 0,
 'span_px':[int(np.ptp(u)),int(np.ptp(v))] if len(u) else None}
rows=[]
env['snap_mask']=snap_mask
for r in cases:
 ident=r['id'];q=r['query'];snaps=[]
 env.update(ident=ident,q=q,ref=refs[ident])
 exec(compiled,env)
 env['robot'].close()
 pix=env['pix'];valid=env['valid'];features=env['features'];mask=env['mask']
 # evaluate original boolean leaves without changing their order or semantics
 cumulative=np.ones(np.shape(valid),dtype=bool);stages=[]
 for n,meta in zip(terms,termmeta):
  a=np.asarray(eval(compile(ast.Expression(body=n),str(src),'eval'),env),dtype=bool)
  before=int(cumulative.sum());cumulative &= a
  stages.append({'stage':meta['label'],'standalone_pass':int(a.sum()),'sequential_remaining':int(cumulative.sum()),'sequential_removed':before-int(cumulative.sum())})
 assert np.array_equal(cumulative,valid)
 uv=np.empty((0,2),dtype=np.float32) if features is None else features.reshape(-1,2)
 old=P/'experiments/region_pose/iter_011/train/trajectories'/(r['event']+'_query'+str(q)+'.npz')
 if old.exists():
  with np.load(old) as z: equal=np.array_equal(uv,z['initial_uv'])
 else: equal=len(uv)==r.get('corners')
 # Preserve all exact masks/pixels only on remote.
 np.savez_compressed(O/'cases'/(ident+'.npz'),initial_pixel=pix,valid=valid,
  mask_before_erosion=snaps[-2][1],mask_after_erosion=mask,initial_uv=uv)
 row={'episode':ident,'event':r['event'],'query':q,'initial_component_pixels':len(pix),
 'stage_counts':stages,'mask_steps':[{'stage':label,**stats(m)} for label,m in snaps],
 'corners':len(uv),'recorded_corners':r.get('initial_corners',r.get('corners')),
 'exact_original_corner_reproduction':bool(equal),
 'gray_range_inside_eroded_mask':[int(env['gray0'][mask>0].min()),int(env['gray0'][mask>0].max())] if np.any(mask) else None}
 rows.append(row);save(O/'cases'/(ident+'.json'),row)
 print(ident,'foreground',int(valid.sum()),'eroded',int((mask>0).sum()),'corners',len(uv),'equal',equal,flush=True)
save(O/'summary.json',{'split':'train','events':16,'rows':rows,'tracking':False,'training':False,'new_labels':False})
input_ok=all(sha(Path(p))==h for p,h in read(O/'input_locked.json').items())
source_ok=all(sha(Path(p))==h for p,h in read(O/'source_locked.json').items())
slice_ok=True
for ep,row in slice_lock.items():
 with h5py.File(row['path'],'r') as f:
  for k,v in row['slices'].items():
   key,ix=k.split(':');slice_ok &= ah(f['agentview'][key][int(ix)])==v['sha256']
audit={'input_hashes_unchanged':input_ok,'source_hashes_unchanged':source_ok,'permitted_slices_unchanged':bool(slice_ok),
 'all_16_corner_arrays_reproduced':all(r['exact_original_corner_reproduction'] for r in rows),
 'no_future_rgbd_frames_read':True,'validation_test_read':False,'tracking_training':False}
save(O/'audits/completion_audit.json',audit)
save(O/'stage_state.json',{'stage':'audit_complete' if all([input_ok,source_ok,slice_ok,audit['all_16_corner_arrays_reproduced']]) else 'audit_requires_review','time':time.time()})
