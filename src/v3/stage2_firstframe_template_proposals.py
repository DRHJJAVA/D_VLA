"""Name-free first-frame template proposals from TRAIN object crops.
Query future frames, test data, actions and simulator truth are never read in proposal generation."""
from pathlib import Path
import json
import cv2,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
E={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
OUT=R/'data/v3/proposals/first_frame_templates'
def overlap(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
 inter=max(0,x2-x1)*max(0,y2-y1)
 area=lambda q:(q[2]-q[0])*(q[3]-q[1])
 return inter/(area(a)+area(b)-inter+1e-8)
def read_first(id):
 with h5py.File(E[id]['rgbd_path']) as f:return f['agentview/rgb'][0]
def templates():
 result=[]
 for id in S['train']:
  e=E[id]
  box=json.loads((R/'data/v3/proposals/motion_rgbd_rerank'/e['asset_group']/(e['source_episode']+'.json')).read_text())['selected']['box']
  x1,y1,x2,y2=box
  rgb=read_first(id)
  patch=rgb[y1:y2,x1:x2]
  if min(patch.shape[:2])<12:continue
  result.append((id,patch))
 return result
def propose(id,train):
 query=read_first(id)
 candidates=[]
 for donor,template in train:
  if id==donor:continue
  h,w=template.shape[:2]
  for scale in (.85,1.,1.15):
   H=max(12,int(round(h*scale)));W=max(12,int(round(w*scale)))
   if H>=query.shape[0] or W>=query.shape[1]:continue
   templ=cv2.resize(template,(W,H),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
   response=cv2.matchTemplate(query,templ,cv2.TM_CCOEFF_NORMED)
   for _ in range(2):
    _,score,_,at=cv2.minMaxLoc(response)
    x,y=at
    if y>40:candidates.append(dict(box=[x,y,x+W,y+H],score=float(score),donor=donor))
    cv2.rectangle(response,(max(0,x-W//2),max(0,y-H//2)),(min(response.shape[1]-1,x+W//2),min(response.shape[0]-1,y+H//2)),-1,-1)
 candidates.sort(key=lambda q:q['score'],reverse=True)
 dedup=[]
 for c in candidates:
  if max((overlap(c['box'],d['box']) for d in dedup),default=0)<.5:dedup.append(c)
  if len(dedup)>=30:break
 return dict(id=id,frame_index=0,candidates=dedup,method='RGB first-frame template correlation against training object crops',no_query_future=True,no_names=True,no_simulator_truth=True)
def main():
 train=templates();audit=[]
 for split in ('train','val'):
  for id in S[split]:
   result=propose(id,train)
   path=OUT/split/(id+'.json');path.parent.mkdir(parents=True,exist_ok=True)
   if path.exists():raise FileExistsError(path)
   path.write_text(json.dumps(result,indent=2))
   # The old full-sequence box is opened only for retrospective validation audit.
   e=E[id]
   old=json.loads((R/'data/v3/proposals/motion_rgbd_rerank'/e['asset_group']/(e['source_episode']+'.json')).read_text())['selected']['box']
   vals=[overlap(c['box'],old) for c in result['candidates']]
   row=dict(id=id,split=split,count=len(vals),best_iou=max(vals,default=0),top5_iou=max(vals[:5],default=0),best_rank=int(np.argmax(vals))+1 if vals else None,top1_iou=vals[0] if vals else 0)
   audit.append(row);print(json.dumps(row),flush=True)
 summary={}
 for split in ('train','val'):
  rr=[x for x in audit if x['split']==split]
  summary[split]=dict(total=len(rr),best_iou_mean=float(np.mean([x['best_iou'] for x in rr])),top5_iou_ge_0_5=sum(x['top5_iou']>=.5 for x in rr),any_iou_ge_0_5=sum(x['best_iou']>=.5 for x in rr),top1_iou_ge_0_5=sum(x['top1_iou']>=.5 for x in rr))
 target=R/'audits/v3/stage2_firstframe_template_audit.json'
 target.write_text(json.dumps(dict(reference='old full-sequence box for evaluation only; not true segmentation',summary=summary,rows=audit),indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
