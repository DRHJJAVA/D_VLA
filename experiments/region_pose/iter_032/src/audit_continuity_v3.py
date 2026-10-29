from pathlib import Path
import ast,json,hashlib,datetime,traceback
import numpy as np
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';R=B/'iter_032'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def groups(mask):
 ids=np.flatnonzero(mask);out=[]
 for i in ids:
  i=int(i)
  if not out or i!=out[-1][-1]+1:out.append([i])
  else:out[-1].append(i)
 return out
def short(mask):
 g=groups(mask)
 return {'frames':np.flatnonzero(mask).tolist(),'count':int(np.sum(mask)),'longest':max(map(len,g),default=0),'runs':[[a[0],a[-1],len(a)] for a in g]}
def expr(tree,name):
 found=[a for a in ast.walk(tree) if isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in a.targets)]
 assert len(found)==1,name
 return compile(ast.Expression(found[0].value),str(R/'original_expression'), 'eval')
try:
 config=json.loads((R/'config.json').read_text());eps=config['episodes']
 inputs=json.loads((R/'input_locked.json').read_text())
 assert all(sha(P/k)==v for k,v in inputs.items())
 previous={}
 for num in ['002','003','020']:
  previous[num]=json.loads((B/('iter_'+num)/'completed_locked.json').read_text())
 checked=[]
 for k,v in inputs.items():
  p=P/k
  if 'completed_locked.json' in k:continue
  for num,lock in previous.items():
   candidates=[k,str(p.relative_to(B/('iter_'+num))) if p.is_relative_to(B/('iter_'+num)) else '']
   matches=[c for c in candidates if c in lock]
   if matches:
    assert all(lock[c]==v for c in matches)
    checked.append(k);break
 t=ast.parse((B/'iter_003/src/build_motion.py').read_text())
 run_node=next(a for a in t.body if isinstance(a,ast.FunctionDef) and a.name=='runs')
 scope={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[run_node],type_ignores=[])),'original_runs','exec'),scope)
 original_runs=scope['runs'];gap_code=expr(t,'gap');valid_code=expr(t,'valid');inside_code=expr(t,'inside');ph_code=expr(t,'ph')
 # Numerical continuity sanity checks including separated spans and exactly 5.
 for mask in [np.zeros(12,bool),np.ones(5,bool),np.array([1,1,1,1,0,1,1,1,1,1,0],bool)]:
  assert [a.tolist() for a in original_runs(mask)]==[a for a in groups(mask) if len(a)>=5]
 out=[]; details=[]
 for ep in eps:
  jf=B/'iter_003/evidence/motion'/(ep+'.json');nf=jf.with_suffix('.npz')
  v=json.loads(jf.read_text());assert v['split']=='train'
  z=np.load(nf,allow_pickle=False);H=z['hand_to_world'];q=z['measured_finger_q'];ov=z['visual_overlap'];pts=z['points_world'];offset=z['frame_offsets'];close=z['command_close']
  n=len(q);ix=np.arange(n);d=v['frame_diagnostics']
  assert n==len(d)==len(H)==len(close)==v['frames']
  assert np.array_equal([a['frame'] for a in d],ix)
  counts=np.array([a['finite_finger_region_observed_voxels'] for a in d])
  cloudn=np.array([a['hand_frame_object_points'] for a in d])
  gap=eval(gap_code,{'np':np},{'q':q})
  assert np.allclose(gap,q[:,0]-q[:,1]-.001,atol=1e-12,rtol=0,equal_nan=True)
  assert np.allclose(gap*1000,np.array([a['pad_gap_mm'] for a in d],dtype=float),atol=1e-9,rtol=0,equal_nan=True)
  assert np.allclose(ov,np.array([a['previous_frame_surface_overlap'] for a in d],dtype=float),atol=0,rtol=0,equal_nan=True)
  assert np.isfinite(H).all() and np.isfinite(q[1:]).all()
  assert np.all(np.flatnonzero(~np.isfinite(q).all(axis=1))==0)
  recon=[]
  for ti in range(n):
   p=pts[offset[ti]:offset[ti+1]]
   ph=(p-H[ti,:3,3])@H[ti,:3,:3]
   orig=eval(ph_code,{'np':np},{'p':p,'H':H,'t':ti})
   assert np.array_equal(ph,orig)
   inside=(np.abs(ph[:,0])<=.008)&(ph[:,2]>=.0854)&(ph[:,2]<=.1014)&(ph[:,1]<=q[ti,0]-.0005)&(ph[:,1]>=q[ti,1]+.0005)
   orig=eval(inside_code,{'np':np,'abs':abs},{'ph':ph,'q':q,'t':ti})
   assert np.array_equal(inside,orig)
   recon.append(len(set(map(tuple,np.rint(ph[inside]/.001).astype(int).tolist()))))
  assert np.array_equal(recon,counts)
  transitions=np.diff(np.r_[False,close,False].astype(int))
  starts=np.flatnonzero(transitions==1);ends=np.flatnonzero(transitions==-1)-1
  assert [(a['start_frame'],a['end_frame']) for a in v['attempts']]==list(zip(starts.tolist(),ends.tolist()))
  attempts=[]
  for a in v['attempts']:
   start=a['start_frame'];end=a['end_frame'];window=(ix>=start)&(ix<=end)
   lift=H[:,2,3]-H[start,2,3]
   gates={'region_ge3':counts>=3,'cloud_ge20':cloudn>=20,'overlap_ge0_5':ov>=.5,'lift_gt25mm':lift>.025,'gap_gt0':gap>0}
   valid=window.copy()
   for m in gates.values():valid &=m
   orig=eval(valid_code,{'np':np},{'counts':counts,'cloudn':cloudn,'ov':ov,'H':H,'start':start,'end':end,'gap':gap,'ix':ix})
   assert np.array_equal(orig,valid)
   spans=groups(valid);accepted=[i for g in spans if len(g)>=5 for i in g]
   og=[b.tolist() for b in original_runs(valid)]
   assert og==[g for g in spans if len(g)>=5]
   assert accepted==a['accepted_contact_frames']==a['moving_visual_support_frames']
   assert a['evidence_runs']==[[g[0],g[-1]] for g in spans if len(g)>=5]
   gate_stats={k:short(window&m) for k,m in gates.items()}
   region_frames=np.flatnonzero(window&gates['region_ge3'])
   region_detail=[{'frame':int(i),'count':int(counts[i]),'cloud':int(cloudn[i]),'overlap':float(ov[i]) if np.isfinite(ov[i]) else None,'lift_mm':float(lift[i]*1000),'gap_mm':float(gap[i]*1000),'rejected_gates':[k for k,m in gates.items() if not m[i]]} for i in region_frames]
   rr={'attempt':a['attempt'],'start':start,'end':end,'gates':gate_stats,'intersection':short(valid),'accepted':accepted,'region_candidate_details':region_detail,'nonpassing_counts_including_missing':{k:int(np.sum(window&~m)) for k,m in gates.items()},'overlap_missing_frames':np.flatnonzero(window&~np.isfinite(ov)).tolist(),'max_lift_mm':float(np.max(lift[window]*1000)),'overlap_known_below_threshold':int(np.sum(window&np.isfinite(ov)&(ov<.5)))}
   attempts.append(rr)
   for i in np.flatnonzero(window):
    details.append({'episode':ep,'attempt':a['attempt'],'frame':int(i),'motion_obs_index':int(i)-1 if i>0 else None,'region':int(counts[i]),'cloud':int(cloudn[i]),'overlap':float(ov[i]) if np.isfinite(ov[i]) else None,'gap_mm':float(gap[i]*1000),'lift_mm':float(lift[i]*1000),'gates':{k:bool(m[i]) for k,m in gates.items()},'all_pass':bool(valid[i])})
  row={'episode':ep,'frames':n,'saved_band_points':len(pts),'full_episode_region_ge3':short(counts>=3),'max_region':int(counts.max()),'attempts':attempts,'missing_fields':[],'unknown_measured_gap_frames':np.flatnonzero(~np.isfinite(q).all(axis=1)).tolist(),'surface_overlap_raw_reconstruction_available':False,'overlap_limitation':'NPZ仅存垫区来源点，未存每帧完整物体heldcloud；本轮核对已保存overlap及代码3mm来源，不声称重算原最近邻身份。'}
  out.append(row);save(R/(ep+'.json'),row)
 tree2=ast.parse((B/'iter_002/src/build_motion.py').read_text())
 overlap_meta=[]
 for a in ast.walk(tree2):
  if isinstance(a,ast.Compare) and a.lineno==67:
   overlap_meta.append({'line':a.lineno,'operators':[type(v).__name__ for v in a.ops],'constants':[v.value for v in ast.walk(a) if isinstance(v,ast.Constant)]})
 save(R/'frame_intersections.json',details)
 result={'unit':'4 training episodes; frames are not independent grasps','episodes':out,'new_labels':False,'new_training':False,'validation_read':False,'test_read':False,'overlap_source_ast':overlap_meta}
 save(R/'summary.json',result)
 assert all(sha(P/k)==v for k,v in inputs.items())
 sl=json.loads((R/'source_locked.json').read_text());assert all(sha(P/k)==v for k,v in sl.items())
 al=json.loads((R/'source_addendum_032_2.json').read_text());assert all(sha(P/k)==v for k,v in al.items())
 save(R/'audits/completion_audit.json',{'inputs_unchanged':len(inputs),'sources_unchanged':len(sl),'previous_lock_matches':checked,'region_reconstruction_equal':True,'gap_json_npz_equal':True,'overlap_json_npz_equal':True,'attempt_close_boundaries_equal':True,'original_valid_ast_equal':True,'independent_continuity_equal':True,'all_original_accepted_reproduced':True,'frame_rows':len(details),'test_read':False,'stage2_accepted':False})
 a3=json.loads((R/'source_addendum_032_3.json').read_text());assert all(sha(P/k)==v for k,v in a3.items())
 save(R/'stage_state.json',{'stage':'audited_pending_report'})
 print(json.dumps({'audited':True,'rows':len(details),'episodes':[{'id':a['episode'],'region_ge3_count':a['full_episode_region_ge3']['count'],'region_longest':a['full_episode_region_ge3']['longest'],'attempts':[{'range':[b['start'],b['end']],'gate_counts':{k:[z['count'],z['longest']] for k,z in b['gates'].items()},'intersection':b['intersection'],'region_candidates':b['region_candidate_details']} for b in a['attempts']]} for a in out]},ensure_ascii=True))
except Exception as e:
 save(R/'failure_v3.json',{'type':type(e).__name__,'message':str(e)})
 (R/'traceback_v3_remote.log').write_text(traceback.format_exc())
 print(json.dumps({'audited':False,'error_type':type(e).__name__,'message':str(e)},ensure_ascii=True))
 raise SystemExit(1)
