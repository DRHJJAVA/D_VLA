"""First-frame-only, name-free RGB-D object proposals from tabletop height components.
No sequence future, object names, robot actions, segmentation or simulator truth in proposal generation."""
from pathlib import Path
import json
import cv2,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
E={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
OUT=R/'data/v3/proposals/first_frame_depth'
def propose(e):
 with h5py.File(e['rgbd_path']) as f:
  depth=f['agentview/depth_m'][0];K=f['agentview/K'][0];T=f['agentview/camera_to_world'][0]
 yy,xx=np.indices(depth.shape)
 x=(xx-K[0,2])*depth/K[0,0];y=(yy-K[1,2])*depth/K[1,1]
 world=np.stack((x,y,depth),-1)@T[:3,:3].T+T[:3,3]
 height=world[:,:,2]
 lo=float(np.quantile(height,.02));hi=float(np.quantile(height,.98))
 hist,edges=np.histogram(height[::3,::3],bins=np.arange(lo-.002,hi+.006,.004))
 peak=int(np.argmax(hist));table=float((edges[peak]+edges[peak+1])/2)
 mask=((height>table+.014)&(height<table+.24)&(depth>.05)).astype('uint8')*255
 mask[:40]=0
 mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
 proposals=[]
 for idx in range(1,n):
  x,y,w,h,area=map(int,stats[idx])
  if area<25 or area>7500 or w<7 or h<7 or w>180 or h>180:continue
  if x<3 or y<3 or x+w>=depth.shape[1]-3 or y+h>=depth.shape[0]-3:continue
  component=(labels[y:y+h,x:x+w]==idx)
  median_height=float(np.median(height[y:y+h,x:x+w][component])-table)
  proposals.append(dict(box=[x,y,x+w,y+h],pixels=area,height_m=median_height))
 proposals.sort(key=lambda q:q['pixels'],reverse=True)
 return dict(id=e['id'],source_rgbd=e['rgbd_path'],frame_index=0,method='first-frame depth height from dominant horizontal plane',table_height_m=table,candidates=proposals[:30],no_future=True,no_names=True,no_simulator_truth=True)
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
 inter=max(0,x2-x1)*max(0,y2-y1)
 area=lambda q:(q[2]-q[0])*(q[3]-q[1])
 return inter/(area(a)+area(b)-inter+1e-9)
def main():
 audit=[]
 for split in ('train','val'):
  for id in S[split]:
   e=E[id];p=OUT/split/(id+'.json');p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():result=json.loads(p.read_text())
   else:
    result=propose(e);p.write_text(json.dumps(result,indent=2))
   # Only this audit function may look at the earlier full-sequence candidate for comparison.
   old=json.loads((R/'data/v3/proposals/motion_rgbd_rerank'/e['asset_group']/(e['source_episode']+'.json')).read_text())['selected']['box']
   vals=[iou(c['box'],old) for c in result['candidates']]
   audit.append(dict(id=id,split=split,plane=result['table_height_m'],count=len(vals),best_iou=max(vals,default=0),best_index=int(np.argmax(vals)) if vals else None,top5_iou=max(vals[:5],default=0)))
  print(json.dumps(dict(split=split,episodes=len(S[split]),mean_best_iou=float(np.mean([x['best_iou'] for x in audit if x['split']==split])),center_in_best=sum(x['best_iou']>.3 for x in audit if x['split']==split))),flush=True)
 target=R/'audits/v3/stage2_firstframe_depth_proposals_audit.json'
 target.write_text(json.dumps(dict(reference='old full-sequence box used in evaluation only; not independent ground truth',rows=audit),indent=2))
if __name__=='__main__':main()
