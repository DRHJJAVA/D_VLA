from pathlib import Path
import ast,json,hashlib,types,traceback,sys
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA');E=R/'experiments/region_pose';I=E/'iter_036'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def main():
 z=np.load(I/'regions.npz');r=z['representative_indices'];out=[]
 a=ast.parse((E/'iter_024/src/audit_regions.py').read_text());funcs=[n for n in a.body if isinstance(n,ast.FunctionDef) and n.name in ['apply','rigid']]
 ns={'np':np};exec(compile(ast.Module(body=funcs,type_ignores=[]),'<024>','exec'),ns);reg=types.SimpleNamespace(**{k:ns[k] for k in ['apply','rigid']})
 tree=ast.parse((E/'iter_024/src/audit_loo.py').read_text())
 loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and n.lineno==32)
 nodes=[n for n in loop.body if 43<=n.lineno<=55]
 assert len(nodes)>=6
 for t in [55,57,60]:
  f=np.load(I/'loo_v2'/f'frame_{t}.npz');m=z['frame']==t;src=z['source_xyz'][m];base=z['mapped_query_xyz'][m]
  env={'np':np,'reg':reg,'a':f['inlier_query_xyz'],'b':f['inlier_endpoint_xyz'],'ids':f['inlier_track_id'],'H':f['H'],'Q':f['Q'],'src':src,'base':base,'query':z['query_xyz']}
  exec(compile(ast.Module(body=nodes,type_ignores=[]),'<024loo>','exec'),env)
  errors={'fit':float(np.max(abs(np.asarray(env['fits'])-f['fits']))),'query_max':float(np.max(abs(env['qmax']-f['query_max_shift']))),'region_max':float(np.max(abs(env['rmax']-f['region_max_shift'])))}
  assert max(errors.values())<1e-12
  assert np.array_equal(env['deleted'],f['deleted_track_id'])
  old=np.load(I/'loo'/f'frame_{t}.npz')
  assert all(np.array_equal(old[k],f[k],equal_nan=True) for k in f.files)
  out.append({'frame':t,'original024_core_equal':True,'max_diff':errors,'old_partial_loo_unchanged':True})
 supported=z['supported'];ids=np.flatnonzero(supported)
 groups={}
 for i in ids:groups.setdefault(tuple(np.floor(z['mapped_query_xyz'][i]/.001).astype(int)),[]).append(int(i))
 assert set(r.tolist())==set(v[0] for v in groups.values())
 from033=np.load(E/'iter_033/cases/white_yellow_mug_demo_6.npz')
 si=z['source_index_033']
 for k1,k2 in [('source_xyz','points_world'),('pixel','pixel'),('camera','camera')]:assert np.array_equal(z[k1],from033[k2][si])
 save(I/'audits/original024_independent_reconstruction.json',{'passed':True,'frames':out,'original_core_ast_hash':hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest(),'source_indices_exact':True,'independent_representative_indices_exact':True,'representative_camera_counts':{str(c):int((z['camera'][r]==c).sum()) for c in [0,1]},'supported_observation_camera_counts':{str(c):int((z['camera'][supported]==c).sum()) for c in [0,1]}})
if __name__=='__main__':
 try:main()
 except Exception as e:
  (I/'independent_failure_traceback.log').write_text(traceback.format_exc());save(I/'independent_failure.json',{'type':type(e).__name__,'message':str(e)});sys.exit(1)
