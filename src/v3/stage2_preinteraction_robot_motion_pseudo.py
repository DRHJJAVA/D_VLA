"""Build self-supervised moving-manipulator pseudo masks from preinteraction RGB.
TRAIN masks may teach an auxiliary robot separator; VAL masks are audit only.
Complete source RGB-D episodes stay read-only. No actions, names, simulator truth,
contact annotations, human labels, or reserved held-out episodes are opened.
"""
from pathlib import Path
import hashlib,json,math
import cv2,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
SRC=R/'data/v3/samples/visual_interaction_uniform448_3dcc'
COMP=R/'data/v3/proposals/foreground3d_components'
OUT=R/'data/v3/samples/preinteraction_robot_motion_pseudo_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pseudo(rgb0,rgb1):
 g0=cv2.cvtColor(rgb0,cv2.COLOR_RGB2GRAY)
 g1=cv2.cvtColor(rgb1,cv2.COLOR_RGB2GRAY)
 flow=cv2.calcOpticalFlowFarneback(g0,g1,None,.5,3,21,3,5,1.2,0)
 mag=np.linalg.norm(flow,axis=2)
 scaled=np.uint8(np.clip(mag*20,0,255))
 thr,mask=cv2.threshold(scaled,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
 n,lab,stats,_=cv2.connectedComponentsWithStats(mask)
 candidates=[j for j in range(1,n) if stats[j,cv2.CC_STAT_AREA]>=1800]
 if not candidates:return None,float(thr),float(mag.max())
 largest=max(candidates,key=lambda j:stats[j,cv2.CC_STAT_AREA])
 primary=np.uint8(lab==largest)
 core=cv2.erode(primary,np.ones((5,5),np.uint8),iterations=1)
 expanded=cv2.dilate(primary,np.ones((31,31),np.uint8),iterations=1)
 negative=np.uint8(expanded==0)
 return (primary,core,negative),float(thr),float(mag.max())
def main():
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   with np.load(SRC/split/(id+'.npz')) as z:
    event=int(z['onsets'][0]);object_pixel=z['pixel'].copy()
   with np.load(COMP/split/(id+'.npz')) as z:rgbd=Path(str(z['rgbd_path']))
   last=max(9,event-22)
   times=sorted(set(int(round(x)) for x in np.linspace(8,last,8)))
   rgbd_sha256=sha(rgbd)
   with h5py.File(rgbd) as f:
    rgb=f['agentview/rgb']
    assert times[-1]+3<len(rgb)
    for t in times:
     frame=rgb[t];next_frame=rgb[t+3]
     masks,thr,maxmag=pseudo(frame,next_frame)
     if masks is None:
      row=dict(id=id,split=split,frame=t,status='no_large_motion_component')
      rows.append(row);print(json.dumps(row),flush=True);continue
     primary,core,negative=masks
     area=int(primary.sum());fraction=area/primary.size
     overlap=float(primary[object_pixel[:,1],object_pixel[:,0]].mean())
     status='usable' if .01<=fraction<=.5 and overlap<.03 and core.sum()>1000 else 'low_confidence'
     row=dict(id=id,split=split,frame=t,status=status,area=area,
      area_fraction=fraction,first_object_overlap=overlap,
      otsu_scaled_flow_threshold=thr,max_flow_pixels=maxmag)
     rows.append(row);print(json.dumps(row),flush=True)
     if status!='usable':continue
     dest=OUT/split/(id+'_'+str(t).zfill(3)+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
     if dest.exists():
      with np.load(dest) as old:
       assert np.array_equal(old['rgb'],frame)
       assert np.array_equal(old['positive_core'],core)
      continue
     temp=dest.with_suffix('.partial.npz')
     if temp.exists():raise RuntimeError(str(temp))
     np.savez_compressed(temp,rgb=frame,positive_core=core,negative_far=negative,
      moving_component=primary,frame=np.asarray(t),episode_id=np.asarray(id),
      complete_sequence_event=np.asarray(event),rgbd_source_sha256=np.asarray(rgbd_sha256))
     temp.replace(dest)
 counts={split:dict(total=sum(x['split']==split for x in rows),
  usable=sum(x['split']==split and x['status']=='usable' for x in rows),
  low_confidence=sum(x['split']==split and x['status']=='low_confidence' for x in rows),
  no_large_motion_component=sum(x['split']==split and x['status']=='no_large_motion_component' for x in rows))
  for split in ('train','val')}
 record=dict(status='preinteraction_robot_motion_pseudo_v1_complete',
  source='complete RGB-D train/val sequences; preinteraction frames only',
  no_actions=True,no_names=True,no_simulator_truth=True,no_manual_labels=True,
  pseudo_label='largest Otsu optical-flow moving component; eroded high-confidence positive core and far-background negatives',
  caveat='moving robot pixels are partial; not verified robot-tip or contact labels',
  counts=counts,rows=rows)
 p=OUT/'manifest.json'
 if p.exists():
  old=json.loads(p.read_text())
  assert old['counts']==counts
 else:p.write_text(json.dumps(record,indent=2))
 print(json.dumps(counts),flush=True)
if __name__=='__main__':main()
