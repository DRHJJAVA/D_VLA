import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import pathlib,sys,json,time,datetime,hashlib,traceback
R=pathlib.Path('/mnt/nvme1/drh/D_VLA')
W=R/'experiments/lcjp003/r2/v0_v1/v001'
D=W/'segmentation_diagnostic'
A=R/'docs/LCJP003_R2_V01_assets/coverage32'
sys.path[:0]=[str(R/'experiments/contact_joint_object/dependencies/sam2'),str(R/'experiments/contact_joint_object/dependencies/python')]
def dump(name,obj):
 p=D/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)
def sha(x):return hashlib.sha256(x).hexdigest()
try:
 import numpy as np,torch,h5py,cv2
 from PIL import Image,ImageDraw
 from sam2.build_sam import build_sam2
 from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
 from torch.nn.attention import sdpa_kernel,SDPBackend
 torch.set_num_threads(4)
 torch.backends.cudnn.allow_tf32=True
 torch.backends.cuda.matmul.allow_tf32=False
 cases=json.loads((W/'v1_sampling_lock.json').read_text())['cases']
 comp=json.loads((A/'comparison.json').read_text())
 originals=json.loads((W/'segmentation_coverage.json').read_text())['records']
 model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(R/'experiments/contact_joint_object/dependencies/sam21_tiny.pt'),device='cuda')
 controls=dict(points_per_side=32,points_per_batch=64,pred_iou_thresh=.8,stability_score_thresh=.95,stability_score_offset=1.,mask_threshold=0.,box_nms_thresh=.7,crop_nms_thresh=.7,crop_overlap_ratio=512/1500,crop_n_points_downscale_factor=1,min_mask_region_area=0,multimask_output=True)
 reports=[]
 selected=[(2,1),(6,1),(9,1),(73,1),(1,0)]
 colors=[(245,68,92),(247,160,37),(180,220,51),(31,213,146),(38,177,239),(138,106,240),(234,89,207)]
 combined=Image.new('RGB',(1152,420*len(selected)),(18,24,34))
 for row,(ci,vi) in enumerate(selected):
  c=cases[ci];camera=['agentview','robot0_eye_in_hand'][vi]
  with h5py.File(R/'experiments/contact_joint_object/cache/rgbd'/f"{c['id']}.h5",'r') as h:
   rgb=np.array(h[camera+'/rgb'][c['frame']]);robot=np.array(h[camera+'/robot_mask'][c['frame']]).astype(bool)
  ce=next(x for x in comp['cases'] if x['index']==ci)
  ve=next(x for x in ce['views'] if x['camera']==vi)
  page_raw=np.array(Image.open(R/'docs'/ve['raw']).convert('RGB'))
  assert np.array_equal(page_raw,rgb),'raw_image_mismatch'
  baseline=np.load(W/f'coverage_case{ci}_view{vi}_grid32.npz')['masks'].astype(bool)
  results={}
  for crop in [0,1]:
   gen=SAM2AutomaticMaskGenerator(model,crop_n_layers=0,use_m2m=bool(crop),**controls)
   torch.cuda.synchronize();start=time.perf_counter()
   with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):anns=gen.generate(rgb)
   torch.cuda.synchronize();seconds=time.perf_counter()-start
   masks=[];metadata=[]
   for original_idx,a in enumerate(anns):
    m=np.asarray(a['segmentation'],dtype=bool)&~robot
    if not m.any():continue
    yy,xx=np.where(m)
    metadata.append({'index':len(masks),'original_index':original_idx,'area':int(m.sum()),'raw_area':int(a['area']),'predicted_iou':float(a['predicted_iou']),'stability_score':float(a['stability_score']),'bbox':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],'crop_box':[float(x) for x in a['crop_box']]})
    masks.append(m)
   arr=np.stack(masks) if masks else np.zeros((0,256,256),bool)
   if crop==0:assert arr.shape==baseline.shape and np.array_equal(arr,baseline),'baseline_recompute_mismatch'
   name=f'm2m_case{ci}_view{vi}_m2m{crop}'
   np.savez_compressed(D/(name+'.npz'),masks=arr)
   results[crop]={'crop_layers':0,'use_m2m':bool(crop),'raw_proposals':len(anns),'nonempty_after_robot':len(arr),'seconds':seconds,'npz':str((D/(name+'.npz')).relative_to(R)),'masks_sha256':sha(arr.tobytes()),'metadata':metadata}
   if vi==1:
    ax,ay=(128,190) if ci==73 else (128,185)
    results[crop]['diagnostic_anchor']={'x':ax,'y':ay,'robot':bool(robot[ay,ax]),'candidate_indices':[i for i,m in enumerate(arr) if m[ay,ax]]}
   if crop==1:
    rle=[]
    for m in arr:
     flat=m.ravel().astype(np.int8);diff=np.diff(np.r_[0,flat,0]);starts=np.where(diff==1)[0];ends=np.where(diff==-1)[0]
     rle.append([[int(x),int(y-x)] for x,y in zip(starts,ends)])
    results[crop]['rle']=rle
   results[crop]['_array']=arr
  panels=[]
  for col,label in enumerate(['Raw RGB','32 grid baseline crop0','32 grid m2m True']):
   src=rgb.copy()
   if col:
    arr=results[col-1]['_array']
    for j,m in enumerate(arr):
     contours,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
     cv2.drawContours(src,contours,-1,colors[j%len(colors)],1)
     yy,xx=np.where(m)
     if len(xx):cv2.putText(src,str(j),(int(np.median(xx)),int(np.median(yy))),cv2.FONT_HERSHEY_SIMPLEX,.34,colors[j%len(colors)],1,cv2.LINE_AA)
   panel=Image.new('RGB',(384,420),(18,24,34))
   panel.paste(Image.fromarray(src).resize((380,380)),(2,24))
   draw=ImageDraw.Draw(panel);draw.text((8,7),f'Case {ci} view {vi} | {label}',fill='white')
   note=f'{c["target"]} frame {c["frame"]}' if col==0 else f'N={results[col-1]["nonempty_after_robot"]}  {results[col-1]["seconds"]:.3f}s'
   draw.text((8,406),note,fill=(207,220,233))
   combined.paste(panel,(384*col,420*row));panels.append(panel)
  one=Image.new('RGB',(1152,420),(18,24,34))
  for col,panel in enumerate(panels):one.paste(panel,(384*col,0))
  gallery=f'm2m_case{ci}_view{vi}_comparison.jpg';one.save(A/gallery,quality=95)
  for r in results.values():r.pop('_array')
  original=next(x for x in originals if x['case']==ci and x['camera']==vi and x['grid']==32)
  report={'case':ci,'camera':vi,'frame':c['frame'],'episode_id':c['id'],'target':c['target'],'raw_rgb_sha256':sha(rgb.tobytes()),'page_raw_rgb_sha256':sha(page_raw.tobytes()),'raw_exact_equal':True,'robot_mask_sha256':sha(robot.tobytes()),'baseline_recompute_exact_equal':True,'original_baseline_seconds':original['seconds'],'runs':results,'comparison_image':str((A/gallery).relative_to(R))}
  reports.append(report);dump('m2m_progress.json',{'completed_views':len(reports),'total_views':len(selected),'case':ci,'camera':vi,'optimizer_updates':0})
 combined.save(A/'m2m_five_views_comparison.jpg',quality=95)
 result={'status':'complete_pending_visual_review','single_variable':'use_m2m:False->True','controls':controls,'cases':reports,'optimizer_updates':0,'heldout_read':False,'old_cache_overwritten':False,'official_source_modified':False,'comparison_image':'docs/LCJP003_R2_V01_assets/coverage32/m2m_five_views_comparison.jpg','timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
 dump('m2m_result.json',result)
except Exception as e:
 dump('m2m_failure.json',{'exception_type':type(e).__name__,'message':str(e)[:240],'frames':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
 raise SystemExit(1)
