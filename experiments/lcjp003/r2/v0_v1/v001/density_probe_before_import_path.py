import os,sys,pathlib,json,time,traceback
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='2'
R=pathlib.Path('/mnt/nvme1/drh/D_VLA');P=pathlib.Path(__file__).parent;A=R/'docs/LCJP003_R2_V01_assets'
sys.path.insert(0,str(R/'experiments/contact_joint_object/dependencies/sam2'))
import torch,numpy as np,h5py,cv2
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from torch.nn.attention import sdpa_kernel,SDPBackend
def main():
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
 model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(R/'experiments/contact_joint_object/dependencies/sam21_tiny.pt'),device='cuda')
 d=json.loads((P/'v1_result.json').read_text());records=[]
 for ci in [10,60]:
  c=d['cases'][ci]
  with h5py.File(R/'experiments/contact_joint_object/cache/rgbd'/f"{c['id']}.h5",'r') as f:
   for v,cn in enumerate(['agentview','robot0_eye_in_hand']):
    rgb=f[cn]['rgb'][0];robot=f[cn]['robot_mask'][0].astype(bool)
    for density in [8,32]:
     gen=SAM2AutomaticMaskGenerator(model,points_per_side=density,points_per_batch=64)
     ts=time.perf_counter()
     with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH): masks=gen.generate(rgb)
     dt=time.perf_counter()-ts
     canvas=rgb.copy();allm=[];meta=[]
     for i,m in enumerate(masks):
      mask=m['segmentation'].astype(bool)&~robot
      if not mask.any():continue
      allm.append(mask)
      contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      color=((i*71+60)%200+40,(i*137+30)%200+40,(i*49+90)%200+40);cv2.drawContours(canvas,contours,-1,color,1)
      meta.append({'index':i,'area':int(mask.sum()),'predicted_iou':float(m['predicted_iou']),'stability_score':float(m['stability_score'])})
     out=f'density_case{ci}_view{v}_grid{density}'
     np.savez_compressed(P/(out+'.npz'),masks=np.stack(allm) if allm else np.empty((0,256,256),bool))
     im=Image.fromarray(canvas).resize((768,768));im.save(A/(out+'.jpg'),quality=95)
     records.append({'case':ci,'camera':v,'grid':density,'raw_proposals':len(masks),'nonempty_after_robot_mask':len(allm),'seconds':dt,'image':out+'.jpg','metadata':meta})
     (P/'density_state.json').write_text(json.dumps({'completed':len(records),'total':8}))
 result={'status':'completed_pending_visual_review','controls':'same official SAM2.1 tiny checkpoint, thresholds/defaults, math SDPA scoped to SAM; only points_per_side changes 8 to32; no semantic/contact rule','cases':[10,60],'records':records,'optimizer_updates':0,'cache_overwritten':False}
 (P/'density_probe.json').write_text(json.dumps(result,indent=2));(A/'density_probe.json').write_text(json.dumps(result))
except_marker=None
try:main()
except Exception as e:(P/'density_failure.json').write_text(json.dumps({'type':type(e).__name__,'summary':str(e).splitlines()[0][:200],'frames':[{'file':pathlib.Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]}))
