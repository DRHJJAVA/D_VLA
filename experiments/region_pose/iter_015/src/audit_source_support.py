from pathlib import Path
import json,hashlib,numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA');B=Path(__file__).resolve().parents[1];P=B.parent
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def audit(split):
 source=P/'iter_014/train' if split=='train' else B/'val'
 summary=json.loads((source/'summary.json').read_text())
 refs={c['event']:c for c in json.loads((P/'iter_013/targets'/split/'summary.json').read_text())['rows']}
 rows=[]
 for c in summary['rows']:
  rec={k:c[k] for k in ['episode','event','query','reference_query','close_start','offset']}
  assert c['query']==c['close_start']-c['offset'] and c['query']<c['close_start']
  rec.update(eligible=False,correspondence_valid=c.get('correspondence_valid',False),reason=c.get('reason'))
  file=source/(c['event']+'_query'+str(c['query'])+'.npz')
  z=np.load(file);ref=np.load(refs[c['event']]['target_source'])
  if 'mapped_reference_xyz' not in z:
   rec['exclusion']='no_supported_mapping';rows.append(rec);continue
  points=z['query_xyz'];W=z['query_to_reference_world'];mapped=points@W[:3,:3].T+W[:3,3]
  assert np.max(np.abs(mapped-z['mapped_reference_xyz']))<1e-6
  k=z['diagnostic_known'].astype(bool);y=z['diagnostic_target'].astype(bool);assert not np.any(y&~k)
  dist,idx=cKDTree(ref['query_xyz']).query(mapped)
  # Exact same-frame identity retains original point indices.
  if c['query']==c['reference_query']:
   assert np.array_equal(points,ref['query_xyz']);idx=np.arange(len(points));dist=np.zeros(len(points))
  assert np.all(dist[k]<=.002+1e-7)
  assert np.all(ref['known'][idx[k]]) and np.array_equal(ref['target'][idx[k]].astype(bool),y[k])
  positive=np.unique(idx[y]);negative=np.unique(idx[k&~y])
  assert not np.intersect1d(positive,negative).size
  np.savez_compressed(B/'audits'/(split+'_'+c['event']+'_query'+str(c['query'])+'_source_indices.npz'),nearest_reference_index=idx,known=k,positive=y,unique_positive_reference_index=positive,unique_negative_reference_index=negative)
  rec.update(positive_pixels=int(y.sum()),negative_pixels=int((k&~y).sum()),unknown_pixels=int((~k).sum()),unique_positive_sources=len(positive),unique_negative_sources=len(negative),reference_positive_sources=int(ref['target'].sum()),reference_negative_sources=int((ref['known']&~ref['target'].astype(bool)).sum()),eligible=bool(c.get('correspondence_valid') and len(positive)>=3 and len(negative)>=16))
  if rec['eligible']:
   rec['pixel_metrics']={m:c[m] for m in ['early_geometry','reference_geometry_on_same_points','center']}
   rec['exclusion']=None
  else:rec['exclusion']='insufficient_unique_reference_sources'
  rows.append(rec)
 offsets=[]
 for offset in [15,10,5,1]:
  rr=[r for r in rows if r['offset']==offset];good=[r for r in rr if r['eligible']]
  offsets.append({'offset':offset,'attempted_events':len(rr),'eligible_events':len(good),'eligible_episode_ids':[r['episode'] for r in good],'conditional_mean_early_pixel_ap':float(np.mean([r['pixel_metrics']['early_geometry']['ap'] for r in good])) if good else None,'conditional_mean_reference_pixel_ap':float(np.mean([r['pixel_metrics']['reference_geometry_on_same_points']['ap'] for r in good])) if good else None,'not_whole_split_performance':True})
 result={'split':split,'rows':rows,'offsets':offsets,'original_episode_count':20 if split=='train' else 4,'represented_events':len(refs),'unrepresented_count':14 if split=='train' else 1,'missing_val_episode':'tomato_sauce_demo_4' if split=='val' else None,'unique_sources_are_not_independent_events':True,'scoring_training_performed':False}
 dump(B/'audits'/(split+'_source_support.json'),result)
 return result
