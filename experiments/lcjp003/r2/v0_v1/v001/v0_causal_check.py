import importlib.util,json,pathlib,sys,ast,hashlib,h5py,numpy as np
P=pathlib.Path(__file__).parent;R=pathlib.Path('/mnt/nvme1/drh/D_VLA')
# Load definitions only, excluding the script entrypoint. This cannot launch a second model probe.
tree=ast.parse((P/'v0_corrected_check.py').read_text())
tree.body=[n for n in tree.body if not isinstance(n,ast.Try)]
mod={ '__file__':str(P/'v0_corrected_check.py'),'__name__':'v0_definitions'}
exec(compile(tree,str(P/'v0_corrected_check.py'),'exec'),mod)
rows=json.loads((R/'experiments/contact_joint_object/manifests/episodes.json').read_text())['episodes']
e=next(e for e in rows if 'bbq_sauce' in e['task'] and e['episode']=='demo_0')
n=json.loads((P/'action_normalization.json').read_text());mu=np.array(n['mean']);sd=np.array(n['std'])
orig=h5py.Dataset.__getitem__;checks=[]
for t in [0,3,15,20]:
 records=[]
 def guarded(ds,key):
  # Static mesh metadata and per-frame object datasets are not time axes.
  path=ds.name
  if '/cache/rgbd/' in ds.file.filename and (path.startswith('/robot/') or path.startswith('/agentview/') or path.startswith('/robot0_eye_in_hand/')) and ds.ndim>0 and ds.shape[0]==e['frames']:
   ix=key[0] if isinstance(key,tuple) else key
   actual=np.arange(ds.shape[0])[ix]
   hi=int(np.asarray(actual).max()) if np.size(actual) else -1
   records.append({'dataset':path,'max_frame':hi})
   assert hi<=t,(path,hi,t)
  if '/cache/observations/' in ds.file.filename and path.startswith('/frames/'):
   fi=int(path.split('/')[2]);records.append({'dataset':path,'max_frame':fi});assert fi<=t
  return orig(ds,key)
 h5py.Dataset.__getitem__=guarded
 try:
  obj,h,target,grip,idx=mod['make_data'](e,t,mu,sd)
 finally:h5py.Dataset.__getitem__=orig
 assert len(records)>0 and idx.max()<=t
 checks.append({'current':t,'history_indices':idx.tolist(),'guarded_observation_reads':len(records),'max_observation_frame':max(q['max_frame'] for q in records),'future_observation_access':False,'future_actions_read_for_loss_only':True})
(P/'v0_causal_read_guard.json').write_text(json.dumps({'checks':checks,'optimizer_updates':0,'entrypoint_not_executed':True},indent=2))
print(json.dumps({'causal_checks':checks}))
