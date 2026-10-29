import sys,json,hashlib,time,traceback
from pathlib import Path
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003');sys.path.insert(0,str(R/'src'))
from data import P0Dataset
ARCH=R/'p0_versions/before_visibility_v3/p0'
audit=json.loads((ARCH/'data_audit.json').read_text());ds=P0Dataset()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):
 a=np.asarray(a);return {'n':len(a),'median':float(np.median(a)) if len(a) else None,'p95':float(np.percentile(a,95)) if len(a) else None,'max':float(a.max()) if len(a) else None}
def sample_depth(depth,uv):
 H,W=depth.shape;x=np.clip(uv[:,0],0,W-1);y=np.clip(uv[:,1],0,H-1)
 x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,W-1);y1=np.minimum(y0+1,H-1)
 a=x-x0;b=y-y0
 return (1-a)*(1-b)*depth[y0,x0]+a*(1-b)*depth[y0,x1]+(1-a)*b*depth[y1,x0]+a*b*depth[y1,x1]
records=[];cycle_sets={'diagnostic':{'old':[],'new':[]},'boundary':{'old':[],'new':[]}}
counts={k:{'source_points':0,'source_kept':0,'legacy_secondary':0,'new_secondary':0,'old_bad':0,'old_bad_rejected':0,'new_bad':0,'states':{str(i):0 for i in range(5)}} for k in cycle_sets}
start=time.monotonic()
for s in audit['samples']:
 p=R.parent.parent/s['path']
 with np.load(ARCH/'samples'/p.name) as z:old={k:z[k].copy() for k in z.files}
 w=ds.load_window(s['episode_id'],s['current_frame'])
 unchanged={}
 for k in ['xyz_base','xyz_norm','point_valid','object_id','state','actions','actions_raw','action_valid','group_id','rgb','uv','projected_uv','camera_id','K','camera_to_world','history_valid']:
  unchanged[k]=bool(np.array_equal(old[k],w[k],equal_nan=True))
 assert all(unchanged.values()),'preserved_data_gate'
 assert np.array_equal(old['relation'][...,:-1],w['relation'][...,:-1]),'relation_geometry_changed'
 valid=w['point_valid'];cam=w['camera_id'];vis=w['view_visible']
 for c in [0,1]:assert vis[:,:,c][valid&(cam==c)].all(),'source_view_lost'
 np.savez_compressed(p,**w)
 s['sha256']=sha(p);s['no_future_frame_read']=True
 # Capture only current-frame geometry already read by adapter for independent audit.
 cap={}
 def profile(f,event,arg):
  if event=='return' and f.f_code.co_filename==str(R/'src/data.py') and f.f_code.co_name=='frame' and 'depth' in f.f_locals:
   for k in ['depth','K','C','xyz']:cap[k]=np.array(f.f_locals[k],copy=True)
 ds._cache.clear();sys.setprofile(profile)
 f=ds.frame(s['episode_id'],s['current_frame'],2048);sys.setprofile(None)
 subset='boundary' if 'boundary' in s['case'] else 'diagnostic'
 rec={'episode':s['episode_id'],'frame':s['current_frame'],'subset':subset,'unchanged':unchanged,'sample_sha256':s['sha256'],'directions':[]}
 ct=counts[subset];ct['source_points']+=int(f['point_valid'].sum())
 ct['source_kept']+=sum(int(((f['camera_id']==c)&f['view_visible'][:,c]&f['point_valid']).sum()) for c in [0,1])
 for sc in [0,1]:
  dc=1-sc;idx=np.flatnonzero(f['point_valid']&(f['camera_id']==sc));uv=f['projected_depth_uv'][idx,dc]
  zd=sample_depth(cap['depth'][dc],uv)
  hom=np.c_[uv+np.array([.375,.875]),np.ones(len(idx))]
  pd=(hom@np.linalg.inv(cap['K'][dc]).T)*zd[:,None]
  world=pd@cap['C'][dc,:3,:3].T+cap['C'][dc,:3,3]
  ps=(world-cap['C'][sc,:3,3])@cap['C'][sc,:3,:3]
  proj=ps@cap['K'][sc].T
  puv=proj[:,:2]/proj[:,2:3]-np.array([.375,.875])
  error=np.linalg.norm(puv-f['depth_uv'][idx],axis=1)
  legacy=f['view_visible_legacy'][idx,dc];current=f['view_visible'][idx,dc];state=f['view_state'][idx,dc]
  cycle_sets[subset]['old'].extend(error[legacy].tolist());cycle_sets[subset]['new'].extend(error[current].tolist())
  ct['legacy_secondary']+=int(legacy.sum());ct['new_secondary']+=int(current.sum())
  ct['old_bad']+=int((legacy&(error>2)).sum());ct['old_bad_rejected']+=int((legacy&(error>2)&~current).sum());ct['new_bad']+=int((current&(error>2)).sum())
  for i in range(5):ct['states'][str(i)]+=int((state==i).sum())
  rec['directions'].append({'source_camera':sc,'legacy':stats(error[legacy]),'new':stats(error[current]),'new_gt2px':int((current&(error>2)).sum())})
 records.append(rec)
report={'version':'visibility_v3','counts':counts,'cycles':{k:{m:stats(v) for m,v in group.items()} for k,group in cycle_sets.items()},'records':records,'all_original_sampled_points_preserved':True,'source_view_retention':1.0,'independent_reference':'bilinear target depth roundtrip retained unchanged; not used in new visibility selector','seconds':time.monotonic()-start,'inputs_sha256':{s['path']:s['sha256'] for s in audit['samples']}}
report['status']='passed' if all(v['new']['p95'] is not None and v['new']['p95']<=2 for v in report['cycles'].values()) else 'needs_review'
(R/'p0/visibility_v3_real.json').write_text(json.dumps(report,indent=2))
audit['input_protocol_version']=ds.input_protocol_version
audit['adapter_sha256']=sha(R/'src/data.py')
audit['visibility_config_sha256']=sha(R/'p0/visibility_v3_config.json')
audit['visibility_source_sha256']=sha(R/'src/visibility.py')
audit['previous_visibility_evidence_archived']='p0_versions/before_visibility_v3/p0'
audit['two_view_fusion']['visibility']='Source always retained. Secondary 4x4 inverse-depth continuous-surface fit; renderer-calibrated fixed metric error; explicit occluded/uncertain states.'
audit['visibility_v3_validation']='p0/visibility_v3_real.json and visibility_v3_synthetic.json'
(R/'p0/data_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['records','inputs_sha256']}))
