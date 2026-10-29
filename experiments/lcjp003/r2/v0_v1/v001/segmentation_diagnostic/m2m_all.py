import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import pathlib,sys,json,time,datetime,hashlib,traceback,fcntl
R=pathlib.Path('/mnt/nvme1/drh/D_VLA');W=R/'experiments/lcjp003/r2/v0_v1/v001';D=W/'segmentation_diagnostic/m2m_all';D.mkdir(exist_ok=True);A=R/'docs/LCJP003_R2_V01_assets/coverage32';B=A/'m2m';B.mkdir(exist_ok=True)
sys.path[:0]=[str(R/'experiments/contact_joint_object/dependencies/sam2'),str(R/'experiments/contact_joint_object/dependencies/python')]
def dump(p,obj):
 p=pathlib.Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':')));t.replace(p)
def sha(x):return hashlib.sha256(x).hexdigest()
def filehash(p):return sha(pathlib.Path(p).read_bytes())
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
try:
 guard=(D/'process.lock').open('w');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import numpy as np,torch,h5py,cv2
 from PIL import Image,ImageDraw
 from sam2.build_sam import build_sam2
 from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
 from torch.nn.attention import sdpa_kernel,SDPBackend
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
 original_lock=json.loads((W/'source_lock.json').read_text())
 for f,h in original_lock.items():assert filehash(R/f)==h,'original_source_hash_mismatch:'+f
 coverage_lock=json.loads((W/'segmentation_coverage_lock.json').read_text())
 assert filehash(W/'v1_sampling_lock.json')==coverage_lock['sampling_lock_sha256']
 assert filehash(W/'segmentation_coverage.py')==coverage_lock['source_sha256']
 tracked=list(original_lock)+[str((W/'segmentation_coverage.py').relative_to(R)),'experiments/contact_joint_object/dependencies/sam2/sam2/automatic_mask_generator.py','experiments/contact_joint_object/dependencies/sam2/sam2/build_sam.py','experiments/contact_joint_object/dependencies/sam2/sam2/sam2_image_predictor.py','experiments/contact_joint_object/dependencies/sam21_tiny.pt']
 hashes={f:filehash(R/f) for f in tracked}
 lock={'time':now(),'source_hashes':hashes,'sampling_sha256':filehash(W/'v1_sampling_lock.json'),'original_source_lock_sha256':filehash(W/'source_lock.json'),'coverage_lock_sha256':filehash(W/'segmentation_coverage_lock.json'),'script_sha256':filehash(__file__),'all_original_source_match':True,'heldout_read':False,'optimizer_updates':0}
 dump(D/'build_lock.json',lock)
 cases=json.loads((W/'v1_sampling_lock.json').read_text())['cases'];basepage=json.loads((A/'comparison.json').read_text());lookup={c['index']:c for c in basepage['cases']}
 assert len(cases)==100 and all('demo_40' not in x['id'] and 'demo_41' not in x['id'] and 'demo_42' not in x['id'] and 'demo_43' not in x['id'] and 'demo_44' not in x['id'] for x in cases)
 anchors=[]
 p=W/'visual_review_checkpoints_cases000_049.json'
 if p.exists():
  for x in json.loads(p.read_text())['items']:anchors.append({'case':x['case_index'],'camera':x['camera'],'point_xy':x['point_xy'],'original_status':x['old_status']})
 p=W/'review_anchor_cases050_099.json'
 if p.exists():
  z=json.loads(p.read_text())
  rows=z if isinstance(z,list) else z.get('items',z.get('cases',z.get('anchors',[])))
  for x in rows:anchors.append({'case':x.get('case',x.get('case_index')),'camera':x['camera'],'point_xy':x['point_xy'],'original_status':x.get('original_status',x.get('old_status'))})
 priority=[(x['case'],x['camera']) for x in anchors]
 order=[]
 for pair in priority+[(i,v) for i in range(100) for v in range(2)]:
  if pair not in order:order.append(pair)
 assert len(order)==200
 dump(D/'review_anchor_lock.json',{'inspection_only':True,'not_training_labels':True,'generation_not_conditioned_on_anchor':True,'items':anchors})
 model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(R/'experiments/contact_joint_object/dependencies/sam21_tiny.pt'),device='cuda')
 controls=dict(points_per_side=32,points_per_batch=64,pred_iou_thresh=.8,stability_score_thresh=.95,stability_score_offset=1.,mask_threshold=0.,box_nms_thresh=.7,crop_n_layers=0,crop_nms_thresh=.7,crop_overlap_ratio=512/1500,crop_n_points_downscale_factor=1,min_mask_region_area=0,use_m2m=True,multimask_output=True)
 gen=SAM2AutomaticMaskGenerator(model,**controls)
 colors=[(245,68,92),(247,160,37),(180,220,51),(31,213,146),(38,177,239),(138,106,240),(234,89,207)]
 records=[];pagecases={};baseline_hashes={};start_all=time.perf_counter()
 def proposal(m,index,meta,source,source_index,duplicate_of=None):
  yy,xx=np.where(m);contours,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  flat=m.ravel().astype(np.int8);df=np.diff(np.r_[0,flat,0]);s=np.where(df==1)[0];e=np.where(df==-1)[0]
  return {'index':index,'original_proposal_index':meta.get('original_index',meta.get('original_proposal_index')),'area':int(m.sum()),'bbox':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],'contours':[x[:,0,:].tolist() for x in contours if len(x)>1],'rle':[[int(a),int(b-a)] for a,b in zip(s,e)],'predicted_iou':meta.get('predicted_iou'),'stability_score':meta.get('stability_score'),'crop_box':meta.get('crop_box'),'source':source,'source_index':source_index,'duplicate_of':duplicate_of}
 def overlay(rgb,masks,path):
  img=rgb.copy()
  for j,m in enumerate(masks):
   cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(img,cs,-1,colors[j%len(colors)],1)
   yy,xx=np.where(m)
   if len(xx):cv2.putText(img,str(j),(int(np.median(xx)),int(np.median(yy))),cv2.FONT_HERSHEY_SIMPLEX,.34,colors[j%len(colors)],1,cv2.LINE_AA)
  Image.fromarray(img).save(path,quality=95)
 def flush(done):
  status='complete_pending_visual_review' if done else 'running'
  times=[x['seconds'] for x in records]
  summary={'status':status,'completed_views':len(records),'total_views':200,'optimizer_updates':0,'elapsed_seconds':time.perf_counter()-start_all,'mean_seconds_per_view':float(np.mean(times)) if times else None,'old_candidates_preserved':True,'semantic_recognition_not_required':True,'coverage_not_yet_passed':True}
  dump(D/'results.json',{'summary':summary,'controls':controls,'records':records,'heldout_read':False,'official_source_modified':False})
  dump(A/'m2m_comparison.json',{'summary':summary,'cases':[pagecases[i] for i in sorted(pagecases)]})
  dump(D/'state.json',{**summary,'updated_at':now()})
 for ci,vi in order:
  c=cases[ci];camera=['agentview','robot0_eye_in_hand'][vi];bc=lookup[ci];bv=next(x for x in bc['views'] if x['camera']==vi)
  with h5py.File(R/'experiments/contact_joint_object/cache/rgbd'/f"{c['id']}.h5",'r') as h:
   rgb=np.array(h[camera+'/rgb'][c['frame']]);robot=np.array(h[camera+'/robot_mask'][c['frame']]).astype(bool)
  displayed=np.array(Image.open(R/'docs'/bv['raw']).convert('RGB'));assert np.array_equal(rgb,displayed),'raw_mismatch'
  oldp=W/f'coverage_case{ci}_view{vi}_grid32.npz';baseline_hashes[str(oldp.relative_to(R))]=filehash(oldp)
  old=np.load(oldp)['masks'].astype(bool);bm=bv['grids']['32']['proposals'];assert len(old)==len(bm)
  for i,m in enumerate(old):assert int(m.sum())==bm[i]['area']
  torch.cuda.synchronize();t=time.perf_counter()
  with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):anns=gen.generate(rgb)
  torch.cuda.synchronize();secs=time.perf_counter()-t
  masks=[];meta=[]
  for i,a in enumerate(anns):
   m=np.asarray(a['segmentation'],bool)&~robot
   if not m.any():continue
   yy,xx=np.where(m);meta.append({'index':len(masks),'original_index':i,'area':int(m.sum()),'raw_area':int(a['area']),'bbox':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],'predicted_iou':float(a['predicted_iou']),'stability_score':float(a['stability_score']),'crop_box':[float(x) for x in a['crop_box']]});masks.append(m)
  new=np.stack(masks) if masks else np.zeros((0,256,256),bool)
  union=np.concatenate([old,new],axis=0);assert np.array_equal(union[:len(old)],old)
  assert not (union&robot).any()
  stem=f'case{ci:03d}_view{vi}';newpath=D/(stem+'_m2m.npz');unionpath=D/(stem+'_union.npz')
  np.savez_compressed(newpath,masks=new);np.savez_compressed(unionpath,masks=union)
  nprops=[proposal(m,i,meta[i],'m2m',i) for i,m in enumerate(new)]
  uprops=[];seen={}
  for i,m in enumerate(union):
   source='baseline32' if i<len(old) else 'm2m';si=i if i<len(old) else i-len(old);info=bm[si] if source=='baseline32' else meta[si];h=sha(m.tobytes());dup=seen.get(h);seen.setdefault(h,i);uprops.append(proposal(m,i,info,source,si,dup))
  ni=B/(stem+'_m2m.jpg');ui=B/(stem+'_union.jpg');overlay(rgb,new,ni);overlay(rgb,union,ui)
  if ci not in pagecases:pagecases[ci]={k:v for k,v in bc.items() if k!='views'};pagecases[ci]['views']=[]
  pagecases[ci]['views'].append({'camera':vi,'raw':bv['raw'],'robot_mask_pixels':int(robot.sum()),'grids':{'m2m':{'seconds':secs,'overlay':str(ni.relative_to(R/'docs')),'proposals':nprops},'union':{'seconds':secs,'overlay':str(ui.relative_to(R/'docs')),'proposals':uprops}}})
  rec={'case':ci,'camera':vi,'frame':c['frame'],'target':c['target'],'episode_id':c['id'],'raw_rgb_sha256':sha(rgb.tobytes()),'page_raw_sha256':sha(displayed.tobytes()),'raw_exact_equal':True,'robot_mask_sha256':sha(robot.tobytes()),'baseline_masks_sha256':sha(old.tobytes()),'m2m_masks_sha256':sha(new.tobytes()),'union_masks_sha256':sha(union.tobytes()),'baseline_retained_exact':True,'baseline_count':len(old),'m2m_count':len(new),'union_count':len(union),'exact_duplicate_count':sum(x['duplicate_of'] is not None for x in uprops),'seconds':secs,'m2m_npz':str(newpath.relative_to(R)),'union_npz':str(unionpath.relative_to(R)),'metadata':meta,'union_sources':[{'index':x['index'],'source':x['source'],'source_index':x['source_index'],'duplicate_of':x['duplicate_of']} for x in uprops],'review_anchor_checks':[]}
  for anchor in anchors:
   if (anchor['case'],anchor['camera'])==(ci,vi):
    px,py=anchor['point_xy'];indices=[i for i,m in enumerate(new) if m[py,px]]
    rec['review_anchor_checks'].append({**anchor,'anchor_robot':bool(robot[py,px]),'m2m_indices':indices,'m2m_metadata':[meta[i] for i in indices]})
    panel=Image.new('RGB',(320*(2+len(indices)),380),(18,24,34));dr=ImageDraw.Draw(panel)
    checks=[('Raw RGB',None,None),('Baseline union',old.any(axis=0),None)]+[(f'm2m candidate {i}',new[i],meta[i]) for i in indices]
    for col,(label,m,info) in enumerate(checks):
     im=rgb.copy()
     if m is not None:im[m]=(.6*im[m]+.4*np.array([43,235,129])).astype(np.uint8)
     panel.paste(Image.fromarray(im).resize((312,312)),(320*col+4,24));dr.text((320*col+7,7),f'{stem} | {label}',fill='white')
     if info:dr.text((320*col+7,340),f'area={info["area"]} IoU={info["predicted_iou"]:.4f}\nstability={info["stability_score"]:.4f} bbox={info["bbox"]}',fill=(210,223,236))
    anchorimg=B/(stem+'_anchor_candidates.jpg');panel.save(anchorimg,quality=95);rec['review_anchor_checks'][-1]['image']=str(anchorimg.relative_to(R))
  records.append(rec)
  if len(records)%5==0:flush(False)
 for f,h in hashes.items():assert filehash(R/f)==h,'source_changed_during_run:'+f
 for f,h in baseline_hashes.items():assert filehash(R/f)==h,'baseline_changed_during_run:'+f
 dump(D/'integrity.json',{'source_start_end_equal':True,'source_files':len(hashes),'baseline_start_end_equal':True,'baseline_files':len(baseline_hashes),'all_200_raw_exact_equal':all(x['raw_exact_equal'] for x in records),'all_baseline_masks_preserved':all(x['baseline_retained_exact'] for x in records),'baseline_file_hashes':baseline_hashes,'optimizer_updates':0})
 flush(True);dump(D/'complete.json',{'completed_at':now(),'completed_views':len(records),'optimizer_updates':0,'visual_review_pending':True})
except Exception as e:
 dump(D/'failure.json',{'exception_type':type(e).__name__,'message':str(e)[:300],'frames':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
 raise SystemExit(1)
