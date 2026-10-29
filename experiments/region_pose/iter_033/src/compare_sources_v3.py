from pathlib import Path
import os,sys,ast,json,hashlib,datetime,traceback
os.environ['MUJOCO_GL']='egl'
import numpy as np,h5py
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';R=B/'iter_033'
sys.path.insert(0,str(P/'src/v3'))
from stage2_probe_motion_fingerpads import cloud,tc
from stage2_robot_only_depth import RobotDepth
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def expression(tree,name):
 arr=[a for a in ast.walk(tree) if isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in a.targets)]
 assert len(arr)==1,name
 return compile(ast.Expression(arr[0].value),'original_'+name,'eval')
def runs(mask):
 ids=np.flatnonzero(mask);gg=[]
 for i in ids:
  i=int(i)
  if not gg or i!=gg[-1][-1]+1:gg.append([i])
  else:gg[-1].append(i)
 return gg
def stat(mask):
 g=runs(mask)
 return {'frames':np.flatnonzero(mask).tolist(),'count':int(np.sum(mask)),'longest':max(map(len,g),default=0),'runs':[[a[0],a[-1],len(a)] for a in g]}
try:
 locks=json.loads((R/'input_locked.json').read_text());assert all(sha(Path(k))==v for k,v in locks.items())
 src=json.loads((R/'source_locked.json').read_text());src.update(json.loads((R/'source_addendum_033_1.json').read_text()));src.update(json.loads((R/'source_addendum_033_2.json').read_text()));assert all(sha(Path(k))==v for k,v in src.items())
 t2=ast.parse((B/'iter_002/src/build_motion.py').read_text());t3=ast.parse((B/'iter_003/src/build_motion.py').read_text());t12=ast.parse((B/'iter_012/src/audit_regions.py').read_text())
 e={k:expression(t2,k) for k in ['hand','rd','nonrobot','ph','spatial','delta','near','good']}
 oldloop=next(a for a in ast.walk(t2) if isinstance(a,ast.For) and a.lineno==53)
 pairiter=eval(compile(ast.Expression(oldloop.iter),'side_pairs','eval'))
 fn=next(a for a in t12.body if isinstance(a,ast.FunctionDef) and a.name=='in_region')
 ns={'np':np,'abs':abs};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'original_region','exec'),ns)
 in_region=ns['in_region'];okcode=expression(next(a for a in t12.body if isinstance(a,ast.FunctionDef) and a.name=='main'),'ok')
 cases=json.loads((R/'cases.json').read_text());summ=[];read_slices={}
 for row in cases:
  ep=row['id'];save(R/'stage_state.json',{'stage':'processing','episode':ep})
  old=json.loads((B/'iter_003/evidence/motion'/(ep+'.json')).read_text())
  assert old['split']=='train'
  z=np.load(B/'iter_003/evidence/motion'/(ep+'.npz'),allow_pickle=False)
  H=z['hand_to_world'];q=z['measured_finger_q'];ov=z['visual_overlap'];n=len(H);counts=np.zeros(n,int);narrow=np.zeros(n,int);strictcounts=np.zeros(n,int)
  points=[];pixels=[];cams=[];offsets=[0];narrow_flags=[];frame_records=[];all_narrow_equal=True;representatives=[]
  with h5py.File(row['rgbd'],'r') as f:
   actualhand=eval(e['hand'],{'np':np,'tc':tc},{'f':f})
   assert np.array_equal(actualhand,H),'hand calibration mismatch'
   size=f['agentview/rgb'].shape[1];robot=RobotDepth(size)
   try:
    for t in range(n):
     if t==0:
      offsets.append(0);frame_records.append({'frame':0,'status':'unknown_measured_q','full_voxels':None,'narrow_voxels':None});continue
     pf=[];uf=[];cf=[];nf=[];strictp=[];oldp=[];oldu=[];oldc=[];olds=[];camera_stats=[]
     for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
      c=f[cam];d=c['depth_m'][t];K=c['K'][t];T=c['camera_to_world'][t]
      scope={'q':q,'t':t,'T':T,'hand':H,'K':K,'robot':robot,'np':np}
      rd=eval(e['rd'],{'np':np},scope)
      p=cloud(d,K,T);ph=eval(e['ph'],{'np':np},{'p':p,'hand':H,'t':t})
      nonrobot=eval(e['nonrobot'],{}, {'d':d,'rd':rd})
      spatial=eval(e['spatial'],{'abs':abs},{'ph':ph})
      union=np.zeros(len(p),bool)
      for side,k in pairiter:
       delta=eval(e['delta'],{}, {'side':side,'k':k,'ph':ph,'q':q,'t':t})
       near=eval(e['near'],{}, {'delta':delta,'spatial':spatial})
       good=eval(e['good'],{}, {'near':near,'nonrobot':nonrobot})
       ids=np.flatnonzero(good);union[ids]=True
       oldp.append(p[ids]);oldu.append(np.c_[ids%d.shape[1],ids//d.shape[1]]);oldc.append(np.full(len(ids),ci));olds.append(np.full(len(ids),k))
      # Exact 012 full finite-region function and depth/robot filter.
      ok=eval(okcode,{'np':np,'in_region':in_region},{'hand':ph,'finger':q,'t':t,'dep':d,'rd':rd})
      independent=in_region(ph,q[t]) & np.isfinite(d.reshape(-1)) & (d.reshape(-1)>0) & (d.reshape(-1)<rd.reshape(-1)-.002)
      assert np.array_equal(ok,independent),'full region validity mismatch'
      strictp.append(p[np.flatnonzero(ok)])
      ok=in_region(ph,q[t]) & np.isfinite(d.reshape(-1)) & (d.reshape(-1)>0) & nonrobot
      assert not np.any(union & ~ok),'narrow not subset of finite valid source'
      ids=np.flatnonzero(ok)
      pf.append(p[ids]);uf.append(np.c_[ids%d.shape[1],ids//d.shape[1]]);cf.append(np.full(len(ids),ci));nf.append(union[ids])
      camera_stats.append({'camera':cam,'full_points':len(ids),'narrow_points':int(union.sum()),'full_region_before_robot':int((in_region(ph,q[t])&np.isfinite(d.reshape(-1))&(d.reshape(-1)>0)).sum())})
      for name,a in [('depth_m',d),('K',K),('camera_to_world',T)]:
       read_slices[ep+'/'+cam+'/'+str(t)+'/'+name]=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
     op=np.concatenate(oldp);ou=np.concatenate(oldu);oc=np.concatenate(oldc);oside=np.concatenate(olds)
     lo,hi=z['frame_offsets'][t:t+2];expected=z['points_world'][lo:hi]
     equal=np.array_equal(op,expected) and np.array_equal(ou,z['point_pixel'][lo:hi]) and np.array_equal(oc,z['point_camera'][lo:hi]) and np.array_equal(oside,z['point_side'][lo:hi])
     if not equal:
      save(R/'audits'/('mismatch_'+ep+'.json'),{'frame':t,'reconstructed_points':len(op),'saved_points':len(expected),'xyz_equal':bool(np.array_equal(op,expected)),'pixel_equal':bool(np.array_equal(ou,z['point_pixel'][lo:hi]))})
      raise ValueError('old narrow source reconstruction mismatch: '+ep+' frame '+str(t))
     oph=(op-H[t,:3,3])@H[t,:3,:3]
     narrow[t]=len(np.unique(np.rint(oph[in_region(oph,q[t])]/.001).astype(int),axis=0))
     assert narrow[t]==old['frame_diagnostics'][t]['finite_finger_region_observed_voxels'],'old count mismatch'
     pp=np.concatenate(pf);uv=np.concatenate(uf);cc=np.concatenate(cf);nn=np.concatenate(nf)
     phh=(pp-H[t,:3,3])@H[t,:3,:3]
     vox=np.rint(phh/.001).astype(int)
     _,keep=np.unique(vox,axis=0,return_index=True)
     counts[t]=len(keep)
     sph=(np.concatenate(strictp)-H[t,:3,3])@H[t,:3,:3]
     strictcounts[t]=len(np.unique(np.rint(sph/.001).astype(int),axis=0))
     representatives.extend((offsets[-1]+keep).tolist())
     # Save all real depth observations and representative indices separately.
     points.append(pp);pixels.append(uv);cams.append(cc);narrow_flags.append(nn);offsets.append(offsets[-1]+len(pp))
     frame_records.append({'frame':t,'full_voxels':int(counts[t]),'full_012_strict_voxels':int(strictcounts[t]),'narrow_voxels':int(narrow[t]),'full_observations':len(pp),'camera_stats':camera_stats,'old_points_equal':True})
   finally:robot.close()
  cloudn=np.array([a['hand_frame_object_points'] for a in old['frame_diagnostics']]);gap=q[:,0]-q[:,1]-.001;ix=np.arange(n);attempts=[]
  for a in old['attempts']:
   start=a['start_frame'];end=a['end_frame'];motion=(ix>=start)&(ix<=end)&(cloudn>=20)&(ov>=.5)&((H[:,2,3]-H[start,2,3])>.025)&(gap>0)
   oldvalid=motion&(narrow>=3);fullvalid=motion&(counts>=3)
   accepted=[i for g in runs(oldvalid) if len(g)>=5 for i in g]
   assert accepted==a['accepted_contact_frames'],'old accepted mismatch'
   newaccepted=[i for g in runs(fullvalid) if len(g)>=5 for i in g]
   attempts.append({'attempt':a['attempt'],'start':start,'end':end,'old_intersection':stat(oldvalid),'full_intersection':stat(fullvalid),'old_accepted':accepted,'candidate_accepted':newaccepted,'claim':'diagnostic candidate only; not new label or grasp success'})
  allpoints=np.concatenate(points) if points else np.empty((0,3));allpix=np.concatenate(pixels) if pixels else np.empty((0,2),int);allcams=np.concatenate(cams) if cams else np.empty(0,int);allflags=np.concatenate(narrow_flags) if narrow_flags else np.empty(0,bool)
  np.savez_compressed(R/'cases'/(ep+'.npz'),points_world=allpoints,pixel=allpix,camera=allcams,frame_offsets=np.array(offsets),from_narrow=allflags,representative_indices=np.array(representatives,dtype=np.int64),narrow_voxels=narrow,full_voxels=counts,full_012_strict_voxels=strictcounts,hand_to_world=H,measured_finger_q=q)
  rec={'episode':ep,'split':'train','frames':n,'frame_records':frame_records,'old_narrow_all_frames_equal':True,'full_observation_count':len(allpoints),'full_region_ge3':stat(counts>=3),'attempts':attempts,'rgbd':row['rgbd']}
  save(R/'cases'/(ep+'.json'),rec);summ.append(rec)
  print(json.dumps({'episode':ep,'old_reproduced':True,'observations':len(allpoints),'attempts':[{'attempt':a['attempt'],'old_accepted':len(a['old_accepted']),'candidate_accepted':len(a['candidate_accepted']),'full_runs':a['full_intersection']['runs']} for a in attempts]}),flush=True)
 save(R/'read_slices_locked_after_run.json',read_slices)
 assert all(sha(Path(k))==v for k,v in locks.items())
 assert all(sha(Path(k))==v for k,v in src.items())
 save(R/'summary.json',{'episodes':summ,'unit':'5 training episodes, 6 attempts; frames not independent grasps','new_labels':False,'new_training':False,'test_read':False,'validation_read':False})
 save(R/'audits/completion_audit.json',{'inputs_unchanged':len(locks),'source_unchanged':len(src),'all_original_narrow_points_pixels_cameras_sides_counts_equal':True,'all_original_accepted_equal':True,'finite_geometry_012_ast_equal':True,'primary_robot_filter':'002 common to narrow and full','secondary_012_strict_filter_independently_equal':True,'hand_from_camera_calibration_equal':True,'slices':len(read_slices),'slices_hash_timing':'after reading; full file hashes frozen before run','new_labels':False,'new_training':False,'validation_read':False,'test_read':False})
 save(R/'stage_state.json',{'stage':'computed_pending_audit_report'})
except Exception as exc:
 save(R/'failure_v3.json',{'type':type(exc).__name__,'message':str(exc)})
 (R/'traceback_v3_remote.log').write_text(traceback.format_exc())
 print(json.dumps({'failed':True,'type':type(exc).__name__,'message':str(exc)}),flush=True)
 raise SystemExit(1)
