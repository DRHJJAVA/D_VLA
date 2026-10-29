from pathlib import Path
import json,numpy as np,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
r=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_preference/multiview_temporal_fusion_v1')
docs=r.parents[2]/'docs'
rows=[];urls=set()
for i in range(3):
 d=json.loads((r/f'webdata_v2/case{i}.json').read_text());k=d['key']
 z=dict(np.load(r/'refined_cases'/f'{k}.npz'));g=d['fused'];ix=np.array(g['source_id'],int)
 assert np.array_equal(ix,z['supported_representatives'])
 err=float(np.max(np.abs(np.asarray(g['points'])-z['common_xyz'][ix])))
 assert err<1e-7
 for web,arr in [('rgb','rgb'),('pixel','pixel'),('frame','frame'),('camera','camera'),('new','new_vs_reference'),('wrist_extra','wrist_new_voxel'),('support_frames','distinct_frame_support')]:
  assert np.array_equal(np.array(g[web]),z[arr][ix]),web
 assert np.max(np.abs(np.array(g['shift_mm'])-z['loo_shift'][ix]*1000))<1e-4
 rr=np.array(d['ref']['original_index'],int)
 assert np.max(np.abs(np.array(d['ref']['points'])-z['reference_xyz'][rr]))<1e-7
 for j,s in enumerate(ix):
  wf=sorted(set(int(z['frame'][w]) for w in z['witness'][s] if w>=0))
  assert sorted(g['witness_frames'][j])==wf
 ns=[]
 for ft,v in d['views'].items():
  a=dict(np.load(r/'refined_cases'/f'{k}_view{ft}.npz'))
  good=a['near_without_front_conflict']
  assert np.array_equal(v['near'],good)
  assert np.array_equal(v['near_unknown'],a['near_unknown'])
  assert np.array_equal(v['depth_class'],a['depth_class'])
  assert np.array_equal(v['reference_near'],a['reference_near_finger'][rr])
  assert np.all(a['active_distance'][good]<=.003+1e-9)
  assert np.all(a['mount_distance'][good]>=a['active_distance'][good])
  assert not a['inside_collision'][good].any()
  assert not (a['depth_class'][:,good]==2).any()
  ns.append({'frame':int(ft),'display_orange':int(good.sum())})
 for url in d['images'].values():
  assert (docs/url.lstrip('/')).exists(),url
  urls.add(url)
 rows.append({'key':k,'point_serialization_max_m':err,'orange_checks':ns,'provenance_equal':True})
out={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),'cases':rows,'existing_image_urls':len(urls),'all_passed':True,'scope':'serialized points/provenance/geometry masks; not physical contact truth'}
(r/'audits/web_and_geometry.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False))
