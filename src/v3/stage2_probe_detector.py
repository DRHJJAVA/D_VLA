"""Exploratory pretrained visual detector check. No output is used for model training."""
from pathlib import Path
import os,json
os.environ['TRANSFORMERS_NO_TF']='1'
import numpy as np,h5py,torch
from PIL import Image,ImageDraw
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
R=Path('/mnt/nvme1/drh/D_VLA');name='IDEA-Research/grounding-dino-tiny'
processor=AutoProcessor.from_pretrained(name,local_files_only=True)
model=AutoModelForZeroShotObjectDetection.from_pretrained(name,local_files_only=True).cuda().eval()
cases=[('alphabet_soup','a soup can. a robot gripper.'),('tomato_sauce','a tomato sauce bottle. a robot gripper.'),('porcelain_mug','a white mug. a robot gripper.'),('white_yellow_mug','a yellow mug. a robot gripper.')]
canvas=Image.new('RGB',(448*4,448));report=[]
for i,(asset,prompt) in enumerate(cases):
 path=R/'data/v2/stage1_rgbd'/asset/'demo_0.h5'
 with h5py.File(path,'r') as f:rgb=f['agentview/rgb'][0]
 im=Image.fromarray(rgb);inputs=processor(images=im,text=prompt,return_tensors='pt').to('cuda')
 with torch.no_grad():out=model(**inputs)
 result=processor.post_process_grounded_object_detection(out,inputs.input_ids,box_threshold=.18,text_threshold=.18,target_sizes=[(rgb.shape[0],rgb.shape[1])])[0]
 detections=[]
 draw=ImageDraw.Draw(im)
 for score,label,box in zip(result['scores'],result['labels'],result['boxes']):
  b=[float(x) for x in box];q=float(score)
  if q<.2:continue
  detections.append(dict(label=label,score=q,box=b))
  color='yellow' if 'gripper' not in label else 'cyan'
  draw.rectangle(b,outline=color,width=3)
  draw.text((b[0],max(0,b[1]-14)),f'{label} {q:.2f}',fill=color,stroke_width=1,stroke_fill='black')
 canvas.paste(im,(448*i,0))
 report.append(dict(asset=asset,prompt=prompt,detections=detections))
 print(json.dumps(report[-1]),flush=True)
out=R/'docs/图示/第二步_自动物体定位探针.jpg';canvas.save(out,quality=90)
(R/'audits/v3/stage2_detector_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(out,flush=True)
