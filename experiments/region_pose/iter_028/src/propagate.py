from pathlib import Path
import sys,json,hashlib,ast,numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/region_pose';B=P/'iter_028'
sys.path.insert(0,str(B/'src'))
from tracking_math import apply,rigid
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
t=ast.parse((P/'iter_025/src/sampling.py').read_text());fn=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='fps')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'causal_fps','exec'))
for name in ['source_locked.json','input_locked.json','propagation_source_locked.json']:
 for f,h in json.loads((B/name).read_text()).items():assert sha(R/f)==h
cases={x['event']:x for x in json.loads((B/'cases.json').read_text())['rows']}
old=json.loads((P/'iter_015/val/summary.json').read_text())['rows']
new=[]
assert len(old)==12 and len(new)==0
rows=[];maxspread=0.;maxmapped=0.;reconstruction=[]
for rec,isold in [(x,True) for x in old]+[(x,False) for x in new]:
 event=rec['event'];q=rec['query'];refq=rec['reference_query'];c=cases[event]
 assert c['query_frame']==refq and q==rec['close_start']-rec['offset']
 key=f'{event}_query{q}'
 src=(P/'iter_015/val' if isold else B/'raw_new')/(key+'.npz')
 base={'episode':rec['episode'],'event':event,'query':q,'reference_query':refq,'offset':rec['offset'],'correspondence_reused':isold,'correspondence_valid':rec.get('correspondence_valid',q==refq),'source_file':str(src.relative_to(R))}
 if not src.exists():
  base.update(eligible=False,reason=rec.get('reason','no_correspondence_output'),positive_sources=0,negative_sources=0);rows.append(base);continue
 a=np.load(src);ref=np.load(c['target_source']);points=a['query_xyz'];n=len(points)
 if q==refq:
  assert np.array_equal(points,ref['query_xyz']) and np.array_equal(a['query_pixel'],ref['query_pixel'])
  alternatives=[];W=np.eye(4);mapped=points.copy();idx=np.arange(n);dist=np.zeros(n);spread=np.zeros(n)
  known=ref['known'].astype(bool).copy();y=ref['target'].astype(bool).copy();confidence=ref['confidence'].copy()
  allidx=idx[None];alldist=dist[None]
 else:
  if 'query_to_reference_world' not in a.files:
   base.update(eligible=False,reason=rec.get('reason','no_transform'),positive_sources=0,negative_sources=0);rows.append(base);continue
  W=a['query_to_reference_world'];mapped=apply(points,W);tree=cKDTree(ref['query_xyz'])
  dist,idx=tree.query(mapped);y=ref['target'][idx].astype(bool);known=ref['known'][idx].astype(bool)&(dist<=.002)
  ins=a['inlier'].astype(bool);srcxyz=a['query_match_xyz'][ins];dstxyz=a['reference_match_xyz'][ins]
  alternatives=[]
  for j in range(len(srcxyz)):
   aa=np.delete(srcxyz,j,axis=0);bb=np.delete(dstxyz,j,axis=0)
   scale=np.linalg.svd(aa-aa.mean(0),compute_uv=False)/np.sqrt(len(aa))
   if len(aa)>=7 and scale[1]>=.003:alternatives.append(rigid(aa,bb))
  spread=np.zeros(n);indices=[idx];distances=[dist];confidence=ref['confidence'][idx].copy()
  for alt in alternatives:
   pp=apply(points,alt);dd,ii=tree.query(pp);spread=np.maximum(spread,np.linalg.norm(pp-mapped,axis=1))
   known &= ref['known'][ii].astype(bool)&(ref['target'][ii].astype(bool)==y)&(dd<=.002)
   confidence=np.minimum(confidence,ref['confidence'][ii]);indices.append(ii);distances.append(dd)
  known &= spread<=.002
  if len(alternatives)<7:known[:]=False
  confidence=confidence*np.exp(-spread/.002)
  allidx=np.stack(indices);alldist=np.stack(distances)
  maxspread=max(maxspread,float(np.max(abs(spread-a['position_sensitivity_m']))))
  maxmapped=max(maxmapped,float(np.max(abs(mapped-a['mapped_reference_xyz']))))
  assert np.allclose(spread,a['position_sensitivity_m'],atol=1e-10,rtol=0)
  assert np.allclose(mapped,a['mapped_reference_xyz'],atol=1e-10,rtol=0)
 y &= known;confidence[~known]=0
 if not isold:
  assert np.array_equal(y,a['diagnostic_target'].astype(bool))
  assert np.array_equal(known,a['diagnostic_known'].astype(bool))
 sel=fps(points,min(2048,n));assert np.array_equal(sel,fps(points.copy(),min(2048,n)))
 pos=known&y;neg=known&~y
 ps=np.unique(idx[sel][pos[sel]]);ns=np.unique(idx[sel][neg[sel]])
 eligible=len(ps)>=3 and len(ns)>=16
 out={'query_xyz':points,'query_pixel':a['query_pixel'],'query_rgb':a['query_rgb'],'selected_index':sel,'reference_source_index':idx,'all_transform_reference_index':allidx,'all_transform_reference_distance_m':alldist,'target':y,'known':known,'confidence':confidence,'query_to_reference_world':W,'loo_query_to_reference_world':np.array(alternatives).reshape(-1,4,4),'position_sensitivity_m':spread}
 np.savez_compressed(B/'val'/(key+'.npz'),**out)
 base.update(eligible=eligible,reason='source_count_pass' if eligible else 'insufficient_distinct_reference_sources',full_points=n,positive_pixels=int(pos.sum()),negative_pixels=int(neg.sum()),unknown=int((~known).sum()),positive_sources_full=int(len(np.unique(idx[pos]))),negative_sources_full=int(len(np.unique(idx[neg]))),sample_points=len(sel),sample_positive_pixels=int(pos[sel].sum()),sample_negative_pixels=int(neg[sel].sum()),positive_sources=len(ps),negative_sources=len(ns),sample_unknown=int((~known[sel]).sum()),loo_fits=len(alternatives),same_frame_reuse=q==refq)
 dump(B/'val'/(key+'.json'),base);rows.append(base)
 reconstruction.append({'key':key,'original_mapping_and_sensitivity_equal':True,'new_raw_target_equal':True if not isold else None,'causal_fps_repeat_equal':True})
events=[]
for event in sorted(cases):
 rr=[x for x in rows if x['event']==event];assert len(rr)==4
 events.append({'event':event,'valid_queries':sum(x['eligible'] for x in rr),'by_offset':{str(x['offset']):{'eligible':x['eligible'],'positive_sources':x['positive_sources'],'negative_sources':x['negative_sources']} for x in rr}})
summary={'rows':rows,'events':events,'queries':len(rows),'eligible_queries':sum(x['eligible'] for x in rows),'events_all_four_eligible':sum(x['valid_queries']==4 for x in events),'events_at_least_one_eligible':sum(x['valid_queries']>0 for x in events),'original_validation_events':4,'candidate_validation_events':3,'reused_queries':12,'new_queries':0,'validation_read':True,'test_read':False,'scoring_training':False,'not_success_rate':True}
dump(B/'summary.json',summary)
dump(B/'audits/propagation_reconstruction.json',{'rows':reconstruction,'max_sensitivity_difference_m':maxspread,'max_mapping_difference_m':maxmapped})
dump(B/'val/locked.json',{str(f.relative_to(R)):sha(f) for f in (B/'val').iterdir() if f.suffix in ['.json','.npz'] and f.name!='locked.json'})
dump(B/'stage_state.json',{'stage':'propagation_complete','eligible_queries':summary['eligible_queries']})
print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','events']}))
