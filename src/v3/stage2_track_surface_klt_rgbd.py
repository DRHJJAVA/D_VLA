"""Full-sequence initial-object surface tracking diagnostic, train/old-val only.
Forward/backward KLT + first appearance check, followed by independent RGB-D rigidity.
No action/contact/object truth; no reinitialization or grasp labels.
"""
from pathlib import Path
import json,hashlib,argparse,time
import numpy as np,cv2,h5py
R=Path('/mnt/nvme1/drh/D_VLA')
OUT=R/'audits/v3/surface_klt_rgbd_pilot_v1'
SELECTION={'train':['tomato_sauce_demo_3','porcelain_mug_demo_4','white_yellow_mug_demo_2'],
           'val':['alphabet_soup_demo_1','porcelain_mug_demo_2','tomato_sauce_demo_4','white_yellow_mug_demo_1']}
def write_json(p,v):
 s=json.dumps(v,ensure_ascii=False,indent=2)
 if p.exists():assert p.read_text()==s,str(p)
 else:p.write_text(s)
def protocol():
 rows=[]
 for split,ids in SELECTION.items():
  for id in ids:
   p=R/'data/v3/proposals/foreground3d_components'/split/(id+'.npz')
   with np.load(p) as z:rgbd=str(z['rgbd_path'])
   with h5py.File(rgbd,'r') as f:n=len(f['agentview/rgb'])
   rows.append(dict(split=split,id=id,component=str(p),rgbd=rgbd,frames=n))
 return dict(episodes=rows,corner_max=300,corner_quality=.01,corner_min_distance=3,mask_erosion=3,
 lk_window=21,lk_levels=3,fb_error_px=.75,patch_size=11,initial_patch_ncc=.55,initial_patch_std_min=3.,
 depth_range_m=.01,rigid_threshold_m=.008,rigid_ransac_iterations=200,
 minimum_rigid_points=8,minimum_inlier_fraction=.6,minimum_initial_span_fraction=.08,
 second_axis_ratio_min=.03,wrist_depth_tolerance_m=.015,seed=2026,
 no_reinitialization=True,full_sequences=True,no_names_as_input=True,no_actions=True,no_simulator_truth=True,
 no_manual_labels=True,no_grasp_labels=True,no_reserved=True,
 caveat='initial surface subset only; dropout means unsupported, not absence of contact; geometric consistency is not point identity truth')
def cloud(uv,dep,K,T):
 p=np.rint(uv).astype(int);p[:,0]=np.clip(p[:,0],0,dep.shape[1]-1);p[:,1]=np.clip(p[:,1],0,dep.shape[0]-1)
 z=dep[p[:,1],p[:,0]]
 a=np.stack(((uv[:,0]-K[0,2])*z/K[0,0],(uv[:,1]-K[1,2])*z/K[1,1],z),1)
 return a@T[:3,:3].T+T[:3,3],np.isfinite(z)&(z>.05)
def patch(gray,p):
 return cv2.getRectSubPix(gray,(11,11),tuple(map(float,p))).astype(np.float32).reshape(-1)
def normalized(p):
 q=p-p.mean();s=np.linalg.norm(q);return q/max(s,1e-6)
def fit(a,b):
 ac=a.mean(0);bc=b.mean(0);u,s,v=np.linalg.svd((a-ac).T@(b-bc))
 d=np.eye(3);d[-1,-1]=np.linalg.det(v.T@u.T);rot=v.T@d@u.T
 return rot,bc-rot@ac
def rigid(a,b,seed):
 if len(a)<8:return None
 rng=np.random.default_rng(seed);best=None
 for i in range(200):
  ix=rng.choice(len(a),3,replace=False)
  if np.linalg.norm(np.cross(a[ix[1]]-a[ix[0]],a[ix[2]]-a[ix[0]]))<1e-6:continue
  rot,tr=fit(a[ix],b[ix]);e=np.linalg.norm(a@rot.T+tr-b,axis=1);ok=e<.008
  if best is None or ok.sum()>best.sum():best=ok
 if best is None or best.sum()<8:return None
 rot,tr=fit(a[best],b[best]);e=np.linalg.norm(a@rot.T+tr-b,axis=1);ok=e<.008
 if ok.sum()<8:return None
 rot,tr=fit(a[ok],b[ok]);e=np.linalg.norm(a@rot.T+tr-b,axis=1)
 return rot,tr,ok,e
def wrist_check(points,w,t):
 K=w['K'][t];T=w['camera_to_world'][t];dep=w['depth_m'][t]
 cam=(points-T[:3,3])@T[:3,:3];z=cam[:,2];safe=np.maximum(z,.001)
 uv=np.stack((K[0,0]*cam[:,0]/safe+K[0,2],K[1,1]*cam[:,1]/safe+K[1,2]),1)
 ii=(z>.05)&(uv[:,0]>=0)&(uv[:,0]<dep.shape[1]-1)&(uv[:,1]>=0)&(uv[:,1]<dep.shape[0]-1)
 delta=np.full(len(z),np.inf);jj=np.flatnonzero(ii);q=np.rint(uv[jj]).astype(int)
 delta[jj]=dep[q[:,1],q[:,0]]-z[jj]
 return ii,delta
def run(row,sha):
 op=OUT/(row['id']+'.json');npz=OUT/(row['id']+'.npz')
 if op.exists():
  old=json.loads(op.read_text());assert old['protocol_sha256']==sha and npz.exists();print('reuse',row['id'],flush=True);return
 with np.load(row['component']) as z:pix=z['pixel']
 with h5py.File(row['rgbd'],'r') as f:
  a=f['agentview'];w=f['robot0_eye_in_hand'];n=len(a['rgb'])
  first=a['rgb'][0];g0=cv2.cvtColor(first,cv2.COLOR_RGB2GRAY);dep=a['depth_m'][0]
  m=np.zeros(dep.shape,np.uint8);m[pix[:,1],pix[:,0]]=255
  m=cv2.erode(m,np.ones((3,3),np.uint8))
  dr=cv2.dilate(dep,np.ones((3,3),np.uint8))-cv2.erode(dep,np.ones((3,3),np.uint8))
  m[~np.isfinite(dep)|(dep<=.05)|(dr>=.01)]=0
  q=cv2.goodFeaturesToTrack(g0,maxCorners=300,qualityLevel=.01,minDistance=3,mask=m,blockSize=3)
  p0=q.reshape(-1,2) if q is not None else np.empty((0,2),np.float32)
  keep=np.asarray([patch(g0,p).std()>=3 for p in p0],bool);p0=p0[keep]
  templates=np.stack([normalized(patch(g0,p)) for p in p0]) if len(p0) else np.empty((0,121))
  x0,_=cloud(p0,dep,a['K'][0],a['camera_to_world'][0])
  count=len(p0);tracks=np.full((n,count,2),np.nan,np.float32);tracks[0]=p0
  rigid_flags=np.zeros((n,count),bool);nccs=np.full((n,count),np.nan,np.float32);nccs[0]=1.
  ids=np.arange(count);previous=p0.copy();prev=g0;rows=[];poses=[]
  area0=float(np.prod(np.ptp(p0,axis=0))) if count else 0
  selected=set([0,n-1]+[round((n-1)*x) for x in [.25,.4,.55,.75]])
  # Also retain already-inspected train pilot times, only for visual comparison.
  known={'tomato_sauce_demo_3':[45,75,110,150],'porcelain_mug_demo_4':[36,57,70,90],'white_yellow_mug_demo_2':[45,61,75,100]}
  selected.update(x for x in known.get(row['id'],[]) if x<n)
  canvases=[]
  for t in range(n):
   rgb=a['rgb'][t];g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);depth=a['depth_m'][t];reject_fb=reject_ncc=0
   if t and len(ids):
    nxt,st,_=cv2.calcOpticalFlowPyrLK(prev,g,previous[:,None].astype(np.float32),None,winSize=(21,21),maxLevel=3,
      criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    back,bs,_=cv2.calcOpticalFlowPyrLK(g,prev,nxt,None,winSize=(21,21),maxLevel=3,
      criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    curr=nxt[:,0];fb=np.linalg.norm(back[:,0]-previous,axis=1)
    good=(st[:,0]>0)&(bs[:,0]>0)&np.isfinite(curr).all(1)&(fb<.75)
    good&=(curr[:,0]>=6)&(curr[:,0]<g.shape[1]-6)&(curr[:,1]>=6)&(curr[:,1]<g.shape[0]-6)
    reject_fb=int((~good).sum())
    corr=np.full(len(ids),-1.)
    for j in np.flatnonzero(good):corr[j]=float(normalized(patch(g,curr[j]))@templates[ids[j]])
    reject_ncc=int((good&(corr<.55)).sum());good&=corr>=.55
    ids=ids[good];previous=curr[good];tracks[t,ids]=previous;nccs[t,ids]=corr[good]
   current=previous
   if len(ids):
    x,valid=cloud(current,depth,a['K'][t],a['camera_to_world'][t])
    pp=np.rint(current).astype(int);dr=cv2.dilate(depth,np.ones((3,3),np.uint8))-cv2.erode(depth,np.ones((3,3),np.uint8))
    valid&=dr[pp[:,1],pp[:,0]]<.01
   else:x=np.empty((0,3));valid=np.empty(0,bool)
   goodids=ids[valid];xx=x[valid];res=rigid(x0[goodids],xx,2026+t)
   v=dict(frame=t,tracks_alive=len(ids),initial_tracks=count,depth_supported=int(valid.sum()),
      rejected_fb=reject_fb,rejected_initial_appearance=reject_ncc,rigid_inliers=0,pose_candidate=False,
      initial_span_fraction=0.,second_axis_ratio=0.,residual_mm=None,wrist_depth_match_fraction=None)
   if len(xx):
    inside,delta=wrist_check(xx,w,t);v['wrist_depth_match_fraction']=float((inside&(np.abs(delta)<.015)).mean())
   pose=np.full((4,4),np.nan)
   if res is not None:
    rot,tr,ok,e=res;sel=goodids[ok];rigid_flags[t,sel]=True
    span=float(np.prod(np.ptp(p0[sel],axis=0))/max(area0,1e-9))
    sv=np.linalg.svd(x0[sel]-x0[sel].mean(0),compute_uv=False);ratio=float(sv[1]/max(sv[0],1e-9))
    accepted=bool(ok.sum()>=8 and ok.mean()>=.6 and span>=.08 and ratio>=.03)
    pose=np.eye(4);pose[:3,:3]=rot;pose[:3,3]=tr
    v.update(rigid_inliers=int(ok.sum()),rigid_inlier_fraction=float(ok.mean()),pose_candidate=accepted,
       initial_span_fraction=span,second_axis_ratio=ratio,residual_mm=float(np.median(e[ok])*1000))
   rows.append(v);poses.append(pose)
   if t in selected:
    left=cv2.cvtColor(first,cv2.COLOR_RGB2BGR);right=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    for j in ids:
     co=tuple(map(int,np.rint(np.asarray(cv2.applyColorMap(np.asarray([[int(j*255/max(count-1,1))]],np.uint8),cv2.COLORMAP_TURBO)[0,0]))))
     cv2.circle(left,tuple(np.rint(p0[j]).astype(int)),3,co,-1,cv2.LINE_AA)
     cv2.circle(right,tuple(np.rint(tracks[t,j]).astype(int)),3,co,-1,cv2.LINE_AA)
    canvas=np.concatenate((left,right),1)
    cv2.rectangle(canvas,(0,0),(canvas.shape[1],36),(0,0,0),-1)
    title=row['id']+' t'+str(t)+' tracks '+str(len(ids))+'/'+str(count)+' rigid '+str(v['rigid_inliers'])+(' candidate' if v['pose_candidate'] else ' ABSTAIN')
    cv2.putText(canvas,title,(7,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    image=OUT/(row['id']+'_t'+str(t)+'.jpg')
    assert not image.exists();cv2.imwrite(str(image),canvas,[cv2.IMWRITE_JPEG_QUALITY,89])
   prev=g
  assert not npz.exists()
  np.savez_compressed(npz,first_pixels=p0,first_world=x0,tracks=tracks,initial_appearance_ncc=nccs,rigid_inlier=rigid_flags,pose=np.stack(poses),protocol_sha256=sha)
  summary=dict(id=row['id'],split=row['split'],protocol_sha256=sha,frames=n,initial_tracks=count,
   initial_mask_pixels=int((m>0).sum()),last_frame_alive=int(rows[-1]['tracks_alive']),
   frames_with_pose_candidate=sum(v['pose_candidate'] for v in rows),rows=rows,
   caveat='candidate rigid tracking of firstframe visible texture only; unknown identity after occlusion; no new surface or recovery; no grasp output')
  write_json(op,summary);print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
def main():
 cv2.setNumThreads(4);OUT.mkdir(parents=True,exist_ok=True)
 p=protocol();write_json(OUT/'protocol.json',p);sha=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
 print('protocol_locked',sha,flush=True)
 for row in p['episodes']:run(row,sha)
 print('complete',len(p['episodes']),flush=True)
if __name__=='__main__':main()
