"""Render locked strict single-object visual-frequency baseline for user review.
This is NOT a trained grasp-scoring model. No local sync; all output remote.
"""
from pathlib import Path
import hashlib,json,math
import cv2,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
S=R/'data/v3/samples/object_only_dino_v1/poseval16'
P=R/'outputs/v3_stage2/object_only_frequency_v1/poseval16'
LOCK=R/'outputs/v3_stage2/object_only_frequency_v1/predictions_locked.json'
MAN=R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json'
OUT=R/'docs/图示/第二步_当前单物体条件评分'
IDS=['alphabet_soup_demo_13','tomato_sauce_demo_13','porcelain_mug_demo_13','white_yellow_mug_demo_13']
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
lock=json.loads(LOCK.read_text())
assert lock['status']=='strict_object_only_frequency_predictions_locked'
assert lock['no_query_future'] and lock['no_simulator_truth']
paths={x['id']:Path(x['rgbd_path']) for x in json.loads(MAN.read_text())['rows']}
OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for id in IDS:
 pred=P/(id+'.npz')
 assert sha(pred)==lock['prediction_sha256']['poseval16/'+id]
 with np.load(S/(id+'.npz')) as f:pixel=f['pixel'].copy();box=f['object_crop_box'].copy()
 with np.load(pred) as f:score=f['score'].copy();assertframe=int(f['query_frame']);future=bool(f['has_future'])
 assert assertframe==0 and not future and len(score)==len(pixel)
 with h5py.File(paths[id]) as f:rgb=f['agentview/rgb'][0];dep=f['agentview/depth_m'][0]
 x1,y1,x2,y2=map(int,box);w=x2-x1;h=y2-y1
 y=pixel[:,1]-y1;x=pixel[:,0]-x1
 neutral=np.array([124,116,104],dtype=np.uint8)
 input_rgb=np.broadcast_to(neutral,(h,w,3)).copy()
 input_rgb[y,x]=rgb[pixel[:,1],pixel[:,0]]
 d=dep[pixel[:,1],pixel[:,0]]
 assert np.all((d>.05)&np.isfinite(d))
 near,far=np.quantile(d,[.03,.97]);scale=max(.002,far-near)
 gray=(np.clip((far-d)/scale,0,1)*225+20).astype(np.uint8)
 depth_rgb=np.full((h,w,3),128,dtype=np.uint8)
 depth_rgb[y,x]=gray[:,None]
 k=max(1,int(math.ceil(len(score)*.2)))
 selected=np.argsort(score,kind='stable')[-k:]
 output=input_rgb.copy()
 output[y[selected],x[selected]]=[255,218,25]
 def tile(a):
  return cv2.resize(a,(320,320),interpolation=cv2.INTER_NEAREST)
 board=np.full((340,960,3),255,np.uint8)
 for j,im in enumerate((input_rgb,depth_rgb,output)):
  board[20:340,j*320:(j+1)*320]=tile(im)
 titles=['RGB object input','Depth object input','Top 20% visual score']
 for j,t in enumerate(titles):
  cv2.putText(board,t,(j*320+6,15),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1,cv2.LINE_AA)
 dest=OUT/(id+'.png')
 if dest.exists():raise RuntimeError('output exists '+str(dest))
 cv2.imwrite(str(dest),cv2.cvtColor(board,cv2.COLOR_RGB2BGR))
 rows.append(dict(id=id,points=len(pixel),selected=k,prediction_sha256=sha(pred),image=str(dest.relative_to(R/'docs'))))
print(json.dumps(rows,ensure_ascii=False),flush=True)
manifest=OUT/'manifest.json'
assert not manifest.exists()
manifest.write_text(json.dumps(dict(status='rendered_locked_strict_object_baseline',method='direct visual-frequency transfer from train RGB-D, not trained grasp model',rows=rows),ensure_ascii=False,indent=2))
