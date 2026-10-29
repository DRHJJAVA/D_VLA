from pathlib import Path
import sys,json,hashlib,importlib.util,numpy as np,h5py
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/region_pose';B=P/'iter_027'
sys.path.insert(0,str(B/'src'))
import track_new as m
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
for name in ['source_locked.json','input_locked.json']:
 for f,h in json.loads((B/name).read_text()).items():assert sha(R/f)==h
old=json.loads((P/'iter_014/train/summary.json').read_text())['rows']
cases={c['event']:c for c in json.loads((B/'cases.json').read_text())['rows']}
protocol={x['id']:x for x in json.loads((P/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
checks=[];slices={}
for event in sorted({x['event'] for x in old}):
 c=cases[event];ident=c['episode'];ref=np.load(c['target_source'])
 oldref=np.load(P/'iter_013/targets/train'/f"{event}_query{c['query_frame']}.npz")
 for key in ['query_xyz','query_pixel','query_rgb']:assert np.array_equal(ref[key],oldref[key]),(event,key)
 motion=np.load(P/'iter_003/evidence/motion'/f'{ident}.npz')
 with np.load(protocol[ident]['component']) as z:component=z['pixel'].astype(np.int64)
 with h5py.File(protocol[ident]['rgbd'],'r') as f:
  g=f['agentview'];robot=m.RobotDepth(g['depth_m'].shape[1])
  for rec in [x for x in old if x['event']==event]:
   q=rec['query'];assert rec['reference_query']==c['query_frame']
   a=np.load(P/'iter_014/train'/f'{event}_query{q}.npz')
   uv,xyz,rgb,*_=m.input_cloud(f,component,motion,q,robot)
   assert np.array_equal(uv,a['query_pixel'])
   assert np.array_equal(xyz,a['query_xyz'])
   assert np.array_equal(rgb[uv[:,1],uv[:,0]],a['query_rgb'])
   for key in ['rgb','depth_m','K','camera_to_world']:
    arr=np.asarray(g[key][q]);slices[f'{ident}:{q}:{key}']=hashlib.sha256(arr.tobytes()).hexdigest()
   checks.append({'event':event,'query':q,'reference':c['query_frame'],'reference_cloud_equal':True,'query_cloud_pixel_rgb_equal':True,'reuse_only_correspondence':True})
  robot.close()
dump(B/'audits/reuse_preflight.json',{'rows':checks,'all_pass':True,'old_queries':len(checks),'method_source_sha':sha(P/'iter_014/src/diagnose_early_queries.py')})
dump(B/'input_slices_locked.json',slices)
dump(B/'stage_state.json',{'stage':'reuse_preflight_passed','reused_queries':len(checks)})
print(json.dumps({'preflight':True,'queries':len(checks)}))
