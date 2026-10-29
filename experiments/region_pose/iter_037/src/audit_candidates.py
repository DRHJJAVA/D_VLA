from pathlib import Path
import ast,json,hashlib,traceback,sys,time
import numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA');E=R/'experiments/region_pose';I=E/'iter_037'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def check(p):
 d=json.loads(p.read_text())
 for k,v in d.items():
  f=Path(k);f=f if f.is_absolute() else R/f
  assert sha(f)==v,'locked file mismatch'
 return len(d)
def stats(x):
 x=np.asarray(x);return {'n':int(x.size),'min':float(x.min()) if x.size else None,'median':float(np.median(x)) if x.size else None,'max':float(x.max()) if x.size else None}
def main():
 checks={'source':check(I/'source_locked.json'),'input':check(I/'input_locked.json'),'old036':check(E/'iter_036/completed_locked.json')}
 tree=ast.parse((E/'iter_025/src/build_targets.py').read_text())
 helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['apply','rigid','inside']]
 eventloop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and n.lineno==52)
 core=[n for n in eventloop.body if 64<=n.lineno<=116]
 assert len(core)>10
 st=ast.parse((E/'iter_025/src/sampling.py').read_text());fpsnode=next(n for n in st.body if isinstance(n,ast.FunctionDef) and n.name=='fps')
 orig13=ast.parse((E/'iter_013/src/build_targets.py').read_text())
 orig13func={n.name:n for n in orig13.body if isinstance(n,ast.FunctionDef)}
 helper_equal={n.name:ast.dump(n,include_attributes=False)==ast.dump(orig13func[n.name],include_attributes=False) for n in helpers}
 assert all(helper_equal.values())
 z0=np.load(E/'iter_036/regions.npz');z={k:z0[k] for k in z0.files};frames=[55,57,60];query=z['query_xyz']
 z['selected_frames']=np.array(frames);z['supported_candidate']=z['supported']
 z['query_visible_at_frames']=np.stack([z[f'query_visible_{t}'] for t in frames])
 for t in frames:
  z['finger_q_'+str(t)]=z[f'finger_{t}'];z['query_to_current_hand_'+str(t)]=z[f'Q_{t}'];z['hand_to_world_'+str(t)]=z[f'H_{t}']
 cr=np.load(E/'iter_035/train/correspondences/white_yellow_mug_demo_6_attempt1_iter035_query36.npz')
 ns={'np':np,'cKDTree':cKDTree,'z':z,'cr':cr}
 exec(compile(ast.Module(body=helpers+[fpsnode],type_ignores=[]),'<frozen025methods>','exec'),ns)
 # Sampling is executed before any target construction and receives only current XYZ.
 selected=ns['fps'](query,2048)
 assert np.array_equal(selected,ns['fps'](query,2048))
 exec(compile(ast.Module(body=core,type_ignores=[]),'<frozen025targetcore>','exec'),ns)
 positive=ns['positive'];negative=ns['negative'];confidence=ns['confidence'];known=ns['known']
 assert not np.any(positive&negative);assert np.all(confidence[~known]==0)
 data={'query_xyz':query,'query_pixel':z['query_pixel'],'positive':positive,'negative':negative,'known':known,'confidence':confidence,'causal_fps_indices':selected,'source_index_033':z['source_index_033'],'source_pixel':z['pixel'],'source_camera':z['camera'],'source_frame':z['frame']}
 independent_pos=[];independent_neg=[];independent_may=[];pw=[];nw=[];provenance=[]
 apply=ns['apply']
 for j,t in enumerate(frames):
  lo=np.load(E/'iter_036/loo_v2'/f'frame_{t}.npz');Q=z[f'Q_{t}'];q=z[f'finger_{t}']
  pp=np.stack([apply(query,u) for u in [Q,*lo['fits']]])
  spread=np.linalg.norm(pp-pp[0],axis=2).max(axis=0)
  def within(m):
   return ((np.abs(pp[:,:,0])<=.008+m)&(pp[:,:,2]>=.0854-m)&(pp[:,:,2]<=.1014+m)&(pp[:,:,1]>=q[1]+.0005-m)&(pp[:,:,1]<=q[0]-.0005+m))
  inside_all=within(.001).all(0);may=within(.004).any(0);stable=(spread<=.002)&(len(lo['fits'])>=7)
  ids=np.flatnonzero(z['supported']&(z['frame']==t))
  if len(ids):
   ds=np.linalg.norm(query[:,None,:]-z['mapped_query_xyz'][ids][None,:,:],axis=2)
   nearest_local=ds.argmin(1);dist=ds[np.arange(len(query)),nearest_local];nearest=ids[nearest_local]
  else:dist=np.full(len(query),np.inf);nearest=np.full(len(query),-1,int)
  pos=(dist<=.003)&inside_all&stable;neg=z[f'query_visible_{t}']&~may&stable
  independent_pos.append(pos);independent_neg.append(neg);independent_may.append(may)
  pw.append(pos*np.exp(-spread/.003)*np.exp(-dist/.003));nw.append(neg*np.exp(-spread/.003))
  assert np.array_equal(pos,ns['positives'][j]);assert np.array_equal(neg,ns['negatives'][j])
  data.update({f'positive_support_{t}':pos,f'negative_support_{t}':neg,f'may_inside4mm_{t}':may,f'robust_inside1mm_{t}':inside_all,f'stable_{t}':stable,f'spread_{t}':spread,f'distance_{t}':dist,f'nearest_region_index_{t}':nearest,f'visible_{t}':z[f'query_visible_{t}']})
  final_ids=np.flatnonzero(positive&pos);orig_source=nearest[final_ids]
  provenance.append({'frame':t,'supporting_final_positives':len(final_ids),'distinct_nearest_region_sources':len(np.unique(orig_source)),'nearest_camera_counts_queries':{str(ci):int((z['camera'][orig_source]==ci).sum()) for ci in [0,1]},'distinct_nearest_033_sources':np.unique(z['source_index_033'][orig_source]).tolist(),'positive_support_all_queries':int(pos.sum()),'negative_support_all_queries':int(neg.sum())})
 ps=np.stack(independent_pos).sum(0);nsum=np.stack(independent_neg).sum(0);ever=np.stack(independent_may).any(0)
 p2=ps>=2;n2=(nsum>=2)&~ever
 conflict=p2&n2;p2[conflict]=False;n2[conflict]=False
 c2=np.where(p2,np.stack(pw).sum(0)/np.maximum(ps,1),np.where(n2,np.stack(nw).sum(0)/np.maximum(nsum,1),0))
 assert np.array_equal(p2,positive);assert np.array_equal(n2,negative)
 error=float(np.max(abs(c2-confidence)));assert error<1e-7
 assert not np.any(negative&ever)
 # All nearest source indices retain exact observed point identity.
 source=np.load(E/'iter_033/cases/white_yellow_mug_demo_6.npz')
 for k,old in [('source_xyz','points_world'),('pixel','pixel'),('camera','camera')]:
  assert np.array_equal(z[k],source[old][z['source_index_033']])
 def counts(idx):
  po=idx[positive[idx]];ne=idx[negative[idx]]
  return {'points':len(idx),'positive':len(po),'negative':len(ne),'unknown':int((~known[idx]).sum()),'distinct_positive_query_indices':len(np.unique(po)),'distinct_positive_pixels':len(np.unique(z['query_pixel'][po],axis=0)),'positive_1mm_query_voxels':len(np.unique(np.floor(query[po]/.001).astype(np.int64),axis=0)),'positive_confidence':stats(confidence[po]),'quantity_threshold_pass':len(po)>=3 and len(ne)>=16}
 summary={'episode':'white_yellow_mug_demo_6','query':36,'frames':frames,'independent_episodes':1,'full':counts(np.arange(len(query))),'fps':counts(selected),'per_frame_provenance':provenance,'candidate_only':True,'added_to_training':False,'physical_identity_ground_truth':False,'test_validation_read':False,'new_network':False}
 np.savez_compressed(I/'candidates.npz',**data)
 save(I/'summary.json',summary)
 save(I/'audits/completion.json',{'passed':True,'before':checks,'after':{'source':check(I/'source_locked.json'),'input':check(I/'input_locked.json'),'old036':check(E/'iter_036/completed_locked.json')},'helpers_equal013':helper_equal,'original025_core_ast_sha256':hashlib.sha256(ast.dump(ast.Module(body=core,type_ignores=[]),include_attributes=False).encode()).hexdigest(),'original025_fps_ast_sha256':hashlib.sha256(ast.dump(fpsnode,include_attributes=False).encode()).hexdigest(),'independent_target_known_exact':True,'confidence_max_abs':error,'source_index_exact':True,'fps_current_xyz_only':True,'unknown_zero':True,'negative_never_inside4mm':True})
 save(I/'stage_state.json',{'stage':'computed','utc':time.time()})
if __name__=='__main__':
 try:main()
 except Exception as e:
  (I/'failure_traceback.log').write_text(traceback.format_exc());save(I/'failure.json',{'type':type(e).__name__,'message':str(e)});sys.exit(1)
