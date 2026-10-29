"""Visual-only object instance tracker baseline on TRAIN episodes, no contact/action truth."""
from pathlib import Path
import json,time
import numpy as np,h5py,cv2
R=Path('/mnt/nvme1/drh/D_VLA')
COMP=R/'data/v3/proposals/foreground3d_components/train'
OUT=R/'audits/v3/stage2_mil_instance_tracker_probe.json'
GALLERY=R/'docs/图示'
CASES={
 'tomato_sauce_demo_3':[0,30,45,61,75,88,102,110,125,150],
 'porcelain_mug_demo_4':[0,25,36,48,52,57,70,90],
 'white_yellow_mug_demo_2':[0,30,45,51,61,75,100,130],
}
rows=[]
for id,frames in CASES.items():
 with np.load(COMP/(id+'.npz')) as z:
  path=Path(str(z['rgbd_path']));box=z['component_box'].astype(int)
 initial=(int(box[0]),int(box[1]),int(box[2]-box[0]),int(box[3]-box[1]))
 with h5py.File(path) as f:
  rgb=f['agentview/rgb'];n=len(rgb)
  tracker=cv2.TrackerMIL_create()
  tracker.init(cv2.cvtColor(rgb[0],cv2.COLOR_RGB2BGR),initial)
  track=[dict(frame=0,ok=True,box=list(initial))]
  for i in range(1,n):
   ok,b=tracker.update(cv2.cvtColor(rgb[i],cv2.COLOR_RGB2BGR))
   track.append(dict(frame=i,ok=bool(ok),box=list(map(int,b))))
  w=448;h=448;cols=5;canvas=np.zeros((2*h,cols*w,3),np.uint8)
  for j,t in enumerate(frames):
   if t>=n:continue
   im=cv2.cvtColor(rgb[t],cv2.COLOR_RGB2BGR)
   x,y,bw,bh=track[t]['box']
   cv2.rectangle(im,(x,y),(x+bw,y+bh),(0,255,0),2)
   ix,iy,iw,ih=initial
   cv2.rectangle(im,(ix,iy),(ix+iw,iy+ih),(0,165,255),1)
   cv2.rectangle(im,(0,0),(448,38),(0,0,0),-1)
   cv2.putText(im,id+' t'+str(t)+' '+str(track[t]['ok']),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.50,(255,255,255),1,cv2.LINE_AA)
   canvas[(j//cols)*h:(j//cols+1)*h,(j%cols)*w:(j%cols+1)*w]=im
  dest=GALLERY/('第二步_实例跟踪MIL_'+id+'_内部核对.jpg')
  if dest.exists():raise RuntimeError(str(dest))
  cv2.imwrite(str(dest),canvas,[cv2.IMWRITE_JPEG_QUALITY,91])
  rows.append(dict(id=id,frames=n,initial_box=list(initial),tracked=track,gallery=str(dest)))
  print(json.dumps(dict(id=id,frames=n,at=[track[t] for t in frames if t<n],gallery=str(dest))),flush=True)
record=dict(status='visual_only_mil_tracker_baseline',source='train 448 RGB-D frame RGB, firstframe automatic component box',no_manual_labels=True,no_actions=True,no_contact_truth=True,rows=rows,caveat='tracker success flag is not object-identity accuracy; inspect imagery and depth, abstain on drift')
if OUT.exists():raise RuntimeError(str(OUT))
OUT.write_text(json.dumps(record,indent=2))
print('complete',len(rows))
