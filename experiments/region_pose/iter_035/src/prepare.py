import os,sys,json,ast,copy,hashlib,traceback
from pathlib import Path
os.environ['MUJOCO_GL']='egl'
import numpy as np,cv2,h5py
B=Path('/mnt/nvme1/drh/D_VLA'); E=B/'experiments/region_pose'; O=E/'iter_035'
sys.path[:0]=[str(B/'src/v3'),str(B/'src')]
from stage2_robot_only_depth import RobotDepth
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(8388608),b''):h.update(v)
 return h.hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def nodehash(n):return hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
def check(p):
 d=json.loads(p.read_text())
 for f,h in d.items():
  q=Path(f);q=q if q.is_absolute() else B/q
  assert sha(q)==h,'old lock mismatch'
 return len(d)
def main():
 conf=json.loads((O/'config.json').read_text()); ident=conf['episode'];q=conf['query']
 oldcounts={n:check(E/n/'completed_locked.json') for n in ['iter_022','iter_033','iter_034']}
 ref=next(r for r in json.loads((E/'iter_003/evidence/base/protocol.json').read_text())['episodes'] if r['id']==ident)
 assert ref['split']=='train'
 motion=json.loads((E/'iter_003/evidence/motion'/ (ident+'.json')).read_text())
 assert len(motion['attempts'])==1 and motion['attempts'][0]['start_frame']==37 and q==36
 z33=np.load(E/'iter_033/cases'/ (ident+'.npz')); assert all(z33['full_voxels'][t]>=3 for t in conf['endpoints']);z33.close()
 case={'id':ident,'event':ident+'_attempt1_iter035','split':'train','query':q,'bridges':[{'frame':t} for t in conf['endpoints']],'selected_held_frames':conf['endpoints'],'early_support_start':55,'source_kind':'033_full_volume_diagnostic_not_fusion_or_truth'}
 save(O/'case.json',[case]); save(O/'refs.json',{ident:ref})
 trackpath=E/'iter_022/src/extend_cotracker_grid_v3.py'; tree=ast.parse(trackpath.read_text())
 paths=[trackpath,E/'iter_022/src/sampling_helpers.py',E/'iter_022/src/build.py',E/'iter_003/evidence/base/protocol.json',E/'iter_003/evidence/motion'/(ident+'.json'),E/'iter_003/evidence/motion'/(ident+'.npz'),Path(ref['component']),Path(ref['rgbd']),E/'iter_033/cases'/(ident+'.npz'),E/'iter_033/cases'/(ident+'.json'),E/'iter_010/weights/scaled_offline.pth',B/'src/v3/stage2_robot_only_depth.py',B/'src/v3/stage2_probe_motion_fingerpads.py']
 assert sha(paths[10])=='2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834'
 save(O/'input_locked.json',{str(p):sha(p) for p in paths})
 original_funcs={n.name:nodehash(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!='main'}
 class Adapt(ast.NodeTransformer):
  def visit_Assign(self,n):
   if n.lineno in {7,8,10,11,74,77,78,79}:
    replacements={7:"BASE=Path('/mnt/nvme1/drh/D_VLA/experiments/region_pose/iter_035')",8:"SPLIT='train'",10:"I=BASE/SPLIT",11:"OLD=BASE.parent/'iter_003'",74:"weights=Path('/mnt/nvme1/drh/D_VLA/experiments/region_pose/iter_010/weights/scaled_offline.pth')",77:"baseline={c['id']:c for c in json.loads((BASE/'case.json').read_text())}",78:"refs=json.loads((BASE/'refs.json').read_text())",79:"cases=json.loads((BASE/'case.json').read_text())"}
    return ast.copy_location(ast.parse(replacements[n.lineno]).body[0],n)
   return self.generic_visit(n)
 tree2=ast.fix_missing_locations(Adapt().visit(copy.deepcopy(tree)))
 assert original_funcs=={n.name:nodehash(n) for n in tree2.body if isinstance(n,ast.FunctionDef) and n.name!='main'}
 (O/'src/track.py').write_text(ast.unparse(tree2)+'\n')
 helper=(E/'iter_022/src/sampling_helpers.py').read_text()
 assert helper.count('experiments/region_pose/iter_022')==1
 (O/'src/sampling_helpers.py').write_text(helper.replace('experiments/region_pose/iter_022','experiments/region_pose/iter_035'))
 sources=[O/'设计方案.md',O/'config.json',O/'case.json',O/'refs.json',O/'src/prepare.py',O/'src/track.py',O/'src/sampling_helpers.py']
 save(O/'source_locked.json',{str(p):sha(p) for p in sources})
 # Same current RGB-D preprocessing, truncating before tracking/model construction.
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 loop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='case')
 def prune(n):
  n=copy.deepcopy(n)
  if hasattr(n,'body'):n.body=[prune(s) for s in n.body if s.lineno<=121]
  if isinstance(n,ast.Assign) and n.lineno==121:
   n.value.func=ast.Attribute(value=ast.Name(id='cv2',ctx=ast.Load()),attr='goodFeaturesToTrack',ctx=ast.Load());n.value.args=n.value.args[1:]
  return n
 selected=[prune(n) for n in loop.body if n.lineno in [101,104,106]]
 assert len(selected)==3
 code=compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(trackpath),'exec')
 env=dict(np=np,cv2=cv2,h5py=h5py,RobotDepth=RobotDepth,OLD=E/'iter_003',ident=ident,q=q,ref=ref)
 exec(code,env)
 mask=env['mask'];features=env['features'];old=np.empty((0,2),np.float32) if features is None else features.reshape(-1,2).astype(np.float32)
 buildtree=ast.parse((E/'iter_022/src/build.py').read_text()); block=next(n for n in buildtree.body if isinstance(n,ast.For) and n.lineno==56)
 grid=[copy.deepcopy(n) for n in block.body if 60<=n.lineno<=71];assert len(grid)==5
 gridcode=compile(ast.fix_missing_locations(ast.Module(body=grid,type_ignores=[])),'frozen022grid','exec')
 e=dict(np=np,mask=mask,old=old);exec(gridcode,e);uv=e['uv']
 e2=dict(np=np,mask=mask.copy(),old=old.copy());exec(gridcode,e2);assert np.array_equal(uv,e2['uv'])
 assert np.array_equal(uv[:len(old)],old)
 pix=np.rint(uv).astype(int);assert np.isfinite(uv).all() and len(uv)<=300 and (mask[pix[:,1],pix[:,0]]>0).all()
 # Independent current foreground/mask reconstruction from the saved initial pixel set.
 with np.load(ref['component']) as zz:initial=zz['pixel'].astype(np.int64)
 uu,vv=initial[:,0],initial[:,1];dq=env['dq'];d0=env['d0'];rd=env['rd']
 good=np.isfinite(dq[vv,uu])&(dq[vv,uu]>0)&(np.abs(dq[vv,uu]-d0[vv,uu])<.005)&(dq[vv,uu]<rd[vv,uu]-.002)
 mm=np.zeros(dq.shape,np.uint8);mm[vv[good],uu[good]]=255;mm=cv2.erode(mm,np.ones((3,3),np.uint8))
 assert np.array_equal(mask,mm),'independent mask mismatch'
 np.savez_compressed(O/'seeds'/(ident+'.npz'),mask=mask,original_uv=old,initial_uv=uv,source_is_grid=np.arange(len(uv))>=len(old))
 save(O/'seed_locked.json',{str(p):sha(p) for p in (O/'seeds').glob('*.npz')})
 stats={'old_locks':oldcounts,'query':q,'close':37,'endpoints':conf['endpoints'],'natural_points':len(old),'total_points':len(uv),'foreground_pixels_before_erosion':int(good.sum()),'after_erosion':int((mask>0).sum()),'seed_deterministic':True,'natural_prefix_equal':True,'independent_mask_equal':True,'quality_function_hashes':original_funcs,'grid_ast_hash':nodehash(ast.Module(body=grid,type_ignores=[])),'tracking_allowed':bool(len(uv)>=8),'training':False,'validation_read':False,'test_read':False}
 save(O/'audits/preflight.json',stats)
 slices={}
 with h5py.File(ref['rgbd'],'r') as f:
  g=f['agentview']
  for nm,t in [('depth_m',0),('depth_m',q),('rgb',q),('K',q),('camera_to_world',q)]:
   ar=g[nm][t];slices[f'{nm}/{t}']={'shape':list(ar.shape),'sha256':hashlib.sha256(ar.tobytes()).hexdigest()}
 save(O/'input_slices_locked.json',slices)
 for p in [O/'input_locked.json',O/'source_locked.json',O/'seed_locked.json']:check(p)
 save(O/'stage_state.json',{'stage':'preflight_passed' if len(uv)>=8 else 'insufficient_query_points'})
 print(json.dumps(stats,ensure_ascii=False))
try:main()
except Exception as e:
 (O/'prepare_traceback.txt').write_text(traceback.format_exc())
 save(O/'prepare_failure.json',{'type':type(e).__name__,'message':str(e)})
 print(json.dumps({'error_type':type(e).__name__,'message':str(e)}))
 raise SystemExit(1)
