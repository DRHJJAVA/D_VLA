"""Automatic object proposals from supplied fixed-camera RGB video.
Generic pretrained detector supplies candidates; sustained visual change selects a candidate.
No simulator state, contact, action, or old region label is read for proposal creation."""
from pathlib import Path
import argparse,json,os,time
os.environ['TRANSFORMERS_NO_TF']='1'
import h5py,numpy as np,torch,cv2
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
R=Path('/mnt/nvme1/drh/D_VLA')
PROMPT='a food carton. a bottle. a tin can. a food box. a jar. a mug. a cup. a robot gripper.'
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
 overlap=max(0,x2-x1)*max(0,y2-y1)
 area=lambda q:max(0,q[2]-q[0])*max(0,q[3]-q[1])
 return overlap/(area(a)+area(b)-overlap+1e-9)
def candidates(processor,model,rgb):
 inp=processor(images=Image.fromarray(rgb),text=PROMPT,return_tensors='pt').to('cuda')
 with torch.no_grad():out=model(**inp)
 z=processor.post_process_grounded_object_detection(out,inp.input_ids,box_threshold=.20,text_threshold=.13,target_sizes=[rgb.shape[:2]])[0]
 h,w=rgb.shape[:2];raw=[]
 for score,label,box in zip(z['scores'],z['labels'],z['boxes']):
  if 'gripper' in label:continue
  b=box.int().cpu().tolist();b=[max(0,b[0]),max(0,b[1]),min(w,b[2]),min(h,b[3])]
  area=max(0,b[2]-b[0])*max(0,b[3]-b[1])
  if area<60 or area>.13*h*w:continue
  raw.append(dict(box=b,detector_score=float(score),phrase=label))
 raw.sort(key=lambda x:x['detector_score'],reverse=True)
 keep=[]
 for z in raw:
  if max((iou(z['box'],q['box']) for q in keep),default=0)<.55:keep.append(z)
  if len(keep)==25:break
 return keep
def measure(video,cand):
 x1,y1,x2,y2=cand['box'];g=video['agentview/rgb'];n=len(g)
 # First five supplied frames establish a visual reference, regardless of action phase.
 reference=np.median(g[:min(5,n),y1:y2,x1:x2].astype(np.float32),axis=0)
 change=np.empty(n,dtype=np.float32)
 for t in range(n):
  patch=g[t,y1:y2,x1:x2].astype(np.float32)
  change[t]=np.mean(np.abs(patch-reference))/255
 kernel=min(7,max(1,n//20));smooth=np.convolve(change,np.ones(kernel)/kernel,mode='same')
 # A sustained change anywhere in the supplied recording is valid; no final-state requirement.
 sustained=float(np.quantile(smooth,.92))
 cand['sustained_change']=sustained
 cand['selection_score']=sustained*float(cand['detector_score'])
 return cand
def run(entry,processor,model):
 path=Path(entry['rgbd_path']);out=R/'data/v3/proposals/motion_grounding'/entry['asset_group']/(path.stem+'.json')
 if out.exists():
  j=json.loads(out.read_text());assert j['source_rgbd']==str(path) and j['status']=='complete';return j
 out.parent.mkdir(parents=True,exist_ok=True);start=time.time()
 with h5py.File(path,'r') as f:
  rgb0=f['agentview/rgb'][0]
  found=candidates(processor,model,rgb0)
  measured=[measure(f,c) for c in found]
 measured.sort(key=lambda x:x['selection_score'],reverse=True)
 result=dict(status='complete',id=entry['id'],source_rgbd=str(path),frames=entry['frames'],prompt=PROMPT,selection='pretrained generic object proposals times sustained RGB change across supplied frames',candidate_count=len(measured),selected=measured[0] if measured else None,candidates=measured,uses_simulator_truth=False,uses_region_labels=False,seconds=round(time.time()-start,1))
 tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));tmp.replace(out)
 print(json.dumps(dict(id=entry['id'],count=len(measured),selected=result['selected'],seconds=result['seconds'])),flush=True)
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 torch.set_num_threads(4)
 rows=[json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]
 if a.limit:rows=rows[:a.limit]
 name='IDEA-Research/grounding-dino-tiny'
 processor=AutoProcessor.from_pretrained(name,local_files_only=True)
 model=AutoModelForZeroShotObjectDetection.from_pretrained(name,local_files_only=True).cuda().eval()
 for entry in rows:run(entry,processor,model)
if __name__=='__main__':main()
