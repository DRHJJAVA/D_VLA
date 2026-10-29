"""Name-free moving-object proposals from the original fixed-camera RGB sequence.
Reads all supplied frames. No object names, text prompts, actions, simulator states, or labels."""
from pathlib import Path
import json,cv2,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
def overlap(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
 ab=max(0,x2-x1)*max(0,y2-y1);area=lambda x:(x[2]-x[0])*(x[3]-x[1])
 return ab/(area(a)+area(b)-ab+1e-9)
def proposals(entry):
 path=Path(entry['rgbd_path']);out=R/'data/v3/proposals/motion_only'/entry['asset_group']/(path.stem+'.json')
 if out.exists():
  z=json.loads(out.read_text());assert z['source_rgbd']==str(path) and z['status']=='complete';return z
 with h5py.File(path,'r') as f:
  g=f['agentview/rgb'];n=len(g);h,w=g.shape[1:3]
  ref=np.median(g[:min(5,n)].astype(np.float32),axis=0)
  width=max(5,n//10)
  ends=sorted(set([min(n,round(n*x)) for x in (.2,.4,.6,.8,1.0)]))
  all_candidates=[]
  for end in ends:
   begin=max(0,end-width)
   later=np.median(g[begin:end].astype(np.float32),axis=0)
   diff=np.abs(ref-later).mean(2)/255
   mask=(diff>.09).astype('uint8')*255
   mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),'uint8'))
   mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),'uint8'))
   count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
   for x,y,bw,bh,area in stats[1:]:
    if area<15 or area>.08*h*w:continue
    box=[int(x),int(y),int(x+bw),int(y+bh)]
    if y<.05*h:continue
    change=float(diff[y:y+bh,x:x+bw].mean())
    edge=(x<2 or y<2 or x+bw>w-2 or y+bh>h-2)
    score=change*np.sqrt(area)*(.5 if edge else 1.)
    all_candidates.append(dict(box=box,score=float(score),change=change,area=int(area),edge=bool(edge),window=[begin,end]))
 all_candidates.sort(key=lambda z:z['score'],reverse=True)
 dedup=[]
 for c in all_candidates:
  if max((overlap(c['box'],q['box']) for q in dedup),default=0)<.55:dedup.append(c)
  if len(dedup)==20:break
 result=dict(status='complete',id=entry['id'],source_rgbd=str(path),frames=entry['frames'],method='name-free fixed-camera sustained visual change over multiple temporal windows',uses_object_name=False,uses_text_prompt=False,uses_simulator_truth=False,uses_region_labels=False,candidate_count=len(dedup),selected=dedup[0] if dedup else None,candidates=dedup)
 out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));tmp.replace(out)
 print(json.dumps(dict(id=entry['id'],selected=result['selected'],candidate_count=len(dedup))),flush=True)
 return result
def main():
 rows=[json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]
 for x in rows:proposals(x)
if __name__=='__main__':main()
