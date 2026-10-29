import pathlib,json,hashlib,h5py,numpy as np,time,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
B=pathlib.Path('/mnt/nvme1/drh/D_VLA');E=B/'experiments/contact_joint_object';R=E/'runs/v002/preflight/P1_interaction_extended_v1';A=B/'docs/lcjp002_interaction_extended_v1_assets'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
try:
 assert (R/'pipeline_complete.json').exists()
 p=json.loads((R/'protocol.json').read_text());data=json.loads((A/'results.json').read_text())
 assert [x['id'] for x in data['episodes']]==p['episodes']
 videos=json.loads((R/'video_integrity_check.json').read_text())
 report={'time':now(),'source_revision':p['source_revision'],'source_whole_file_hash_status':'official LFS metadata; whole files not downloaded','download_policy':'HTTP range copy of data/demo_0 only; subset hashes computed','parameters_equal_previous':json.loads((R/'association_config.json').read_text())==json.loads((E/'runs/v002/preflight/P1_interaction_v1/association_config.json').read_text()),'episodes':[],'images':0,'mask_instances':0,'masked_robot_pixels':0,'invalid_depth_pixels':0,'future_reads':0}
 assert report['parameters_equal_previous']
 for ep in data['episodes']:
  id=ep['id'];n=ep['frames_total'];assert [f['t'] for f in ep['frames']]==list(range(n))
  ids={};ranges=[];previous=None
  for f in ep['frames']:
   k=tuple(f['active_ids'])
   if k!=previous:ranges.append({'start':f['t'],'end':f['t'],'ids':list(k)});previous=k
   else:ranges[-1]['end']=f['t']
   for gid in k:ids[str(gid)]=ids.get(str(gid),0)+1
   for cam in ['agentview','robot0_eye_in_hand']:
    for mode in ['rgb','all','associated']:
     path=B/'docs'/f['views'][cam][mode];assert path.is_file() and path.stat().st_size>100;report['images']+=1
  assert ep['input_sha256']==sha(R/'rgbd'/(id+'.h5'))
  assert ep['segmentation_sha256']==sha(R/'segmentation'/(id+'.h5'))
  assert ep['association_source_sha256']==sha(E/'src_v002/interaction_extended_v1/associate_render.py')
  with h5py.File(R/'rgbd'/(id+'.h5'),'r') as h,h5py.File(R/'segmentation'/(id+'.h5'),'r') as s:
   assert len(s['frames'])==n
   for t in range(n):
    for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
     g=s[f'frames/{t}/{ci}'];assert g.attrs['causal_highest_rgb_frame_read']<=t
     rob=h[cam+'/robot_mask'][t].astype(bool);dep=h[cam+'/depth_m'][t]
     for obj in g['objects'].values():
      m=np.unpackbits(obj['mask'][:])[:256*256].reshape(256,256).astype(bool)
      assert int(m.sum())==obj.attrs['area']
      report['mask_instances']+=1;report['masked_robot_pixels']+=int((m&rob).sum());report['invalid_depth_pixels']+=int((m&(~np.isfinite(dep)|(dep<=0))).sum())
  report['episodes'].append({'id':id,'frames':n,'active_frames':ep['summary']['active_frames'],'unknown_frames':ep['summary']['unknown_frames'],'multi_active_frames':ep['summary']['multi_active_frames'],'id_frame_counts':ids,'active_ranges':ranges,'input_sha256':ep['input_sha256'],'segmentation_sha256':ep['segmentation_sha256']})
 assert report['masked_robot_pixels']==0 and report['invalid_depth_pixels']==0
 save(R/'integrity_check.json',report)
 print(json.dumps({k:v for k,v in report.items() if k!='episodes'}))
 print(json.dumps({'episodes':[{k:v for k,v in x.items() if k not in ['input_sha256','segmentation_sha256']} for x in report['episodes']]},ensure_ascii=True))
except Exception as e:
 save(R/'integrity_failure.json',{'time':now(),'type':type(e).__name__,'frames':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
 print(json.dumps({'verification_error':type(e).__name__}))
