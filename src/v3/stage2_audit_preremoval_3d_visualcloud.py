"""Audit image-derived first-frame point cloud against held-out simulator contacts (evaluation only)."""
from pathlib import Path
import json, math
import h5py
import numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
entries=[json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z]
splits=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
splitof={i:k for k,ids in splits.items() for i in ids}
def project(world,K,T,depth,tol=.02):
 cam=(world-T[:3,3])@T[:3,:3]
 z=cam[:,2];safe=np.where(z>0,z,1)
 u=np.rint(K[0,0]*cam[:,0]/safe+K[0,2]).astype(int)
 v=np.rint(K[1,1]*cam[:,1]/safe+K[1,2]).astype(int)
 valid=(z>0)&(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
 ids=np.flatnonzero(valid);visible=np.zeros(len(world),bool)
 visible[ids]=abs(depth[v[ids],u[ids]]-z[ids])<tol
 return u,v,visible
def calc(e):
 aid=e['asset_group'];ep=e['source_episode']
 box=json.loads((R/'data/v3/proposals/motion_rgbd_rerank'/aid/(ep+'.json')).read_text())['selected']['box']
 state=json.loads((R/'data/v2/stage1'/aid/(ep+'_states_contacts.json')).read_text())
 r0=state['rows'][0];p0=np.asarray(r0['object_position']);rot0=np.asarray(r0['object_rotation'])
 contactlocal=[]
 for row in state['rows']:
  p=np.asarray(row['object_position']);rot=np.asarray(row['object_rotation'])
  for c in row['contacts']:contactlocal.append((np.asarray(c['world'])-p)@rot)
 if not contactlocal:return dict(id=e['id'],split=splitof[e['id']],reason='no reference contacts')
 contacts=np.asarray(contactlocal)@rot0.T+p0
 with h5py.File(e['rgbd_path']) as f:
  depthseq=f['agentview/depth_m'][:]
  first=depthseq[0];Kseq=f['agentview/K'][:];Tseq=f['agentview/camera_to_world'][:]
 x1,y1,x2,y2=box
 ring=first[max(0,y1-8):min(first.shape[0],y2+8),max(0,x1-8):min(first.shape[1],x2+8)].copy()
 ry=y1-max(0,y1-8);rx=x1-max(0,x1-8)
 ring[ry:ry+y2-y1,rx:rx+x2-x1]=np.nan
 background=float(np.nanmedian(ring))
 d=first[y1:y2,x1:x2];mask=(d>0.05)&(d<background-.015)
 yy,xx=np.nonzero(mask)
 if len(xx)<30:return dict(id=e['id'],split=splitof[e['id']],reason='too few visual foreground points',count=len(xx))
 px=x1+xx;py=y1+yy;z=d[yy,xx];K=Kseq[0];T=Tseq[0]
 cam=np.stack(((px-K[0,2])*z/K[0,0],(py-K[1,2])*z/K[1,1],z),1)
 pts=cam@T[:3,:3].T+T[:3,3]
 cu,cv,visible=project(contacts,K,T,first)
 visible&=(cu>=x1)&(cu<x2)&(cv>=y1)&(cv<y2)
 cp=contacts[visible]
 if not len(cp):return dict(id=e['id'],split=splitof[e['id']],reason='no visible contacts',object_points=len(pts))
 nearest,ix=cKDTree(pts).query(cp)
 ix=ix[nearest<.015]
 if not len(ix):return dict(id=e['id'],split=splitof[e['id']],reason='no near-cloud contacts',object_points=len(pts),visible_contacts=len(cp))
 foreground=mask
 removed=((d[None,:,:]>d[0:1,:,:]+.025)&foreground[None]).sum((1,2))/foreground.sum() if False else ((depthseq[:,y1:y2,x1:x2]>d[None,:,:]+.025)&foreground[None]).sum((1,2))/foreground.sum()
 candidates=[i for i in range(1,len(removed)-2) if np.all(removed[i:i+3]>.2)]
 if not candidates:return dict(id=e['id'],split=splitof[e['id']],reason='no sustained removal onset',object_points=len(pts))
 onset=candidates[0]
 min_dist=np.full(len(pts),np.inf)
 xlo=max(0,x1-30);ylo=max(0,y1-30);xhi=min(first.shape[1],x2+30);yhi=min(first.shape[0],y2+30)
 for frame in range(max(0,onset-8),onset):
  now=depthseq[frame,ylo:yhi:2,xlo:xhi:2]
  old=depthseq[0,ylo:yhi:2,xlo:xhi:2]
  sy,sx=np.nonzero((now<old-.02)&(now>.05))
  if not len(sx):continue
  pp=xlo+2*sx;qq=ylo+2*sy;z2=now[sy,sx];kt=Kseq[frame];tt=Tseq[frame]
  cam2=np.stack(((pp-kt[0,2])*z2/kt[0,0],(qq-kt[1,2])*z2/kt[1,1],z2),1)
  moving=cam2@tt[:3,:3].T+tt[:3,3]
  min_dist=np.minimum(min_dist,cKDTree(moving).query(pts)[0])
 if not np.isfinite(min_dist).any():return dict(id=e['id'],split=splitof[e['id']],reason='no moving 3d surface')
 # Three prespecified baselines: image center, cloud centroid, and nearest camera.
 score=-min_dist
 center=(pts-pts.mean(0))**2
 centroid_score=-np.linalg.norm(center,axis=1)
 image_score=-np.linalg.norm(np.stack((px,py),1)-np.asarray([(x1+x2)/2,(y1+y2)/2]),axis=1)
 budget=max(1,math.ceil(.2*len(pts)))
 def cov(s):
  top=np.argpartition(s,-budget)[-budget:]
  return float(np.isin(ix,top).mean())
 return dict(id=e['id'],split=splitof[e['id']],object_points=len(pts),visible_contacts=len(cp),evaluable_contacts=len(ix),removal_onset=int(onset),top20_coverage=cov(score),center3d_coverage=cov(centroid_score),center2d_coverage=cov(image_score),random_budget=budget/len(pts))
def main():
 rows=[]
 for e in entries:
  if splitof[e['id']]=='test':continue
  r=calc(e);rows.append(r);print(json.dumps(r),flush=True)
 by={}
 for split in ('train','val'):
  rr=[x for x in rows if x['split']==split and 'top20_coverage' in x]
  by[split]={k:float(np.mean([x[k] for x in rr])) for k in ('top20_coverage','center3d_coverage','center2d_coverage','random_budget')} if rr else {}
  by[split]['evaluable']=len(rr)
 out=dict(status='evaluation_only',points='first-frame RGB-D depth foreground, no simulator mesh',contact_reference='old unreviewed simulator records for evaluation only',partition=by,rows=rows)
 target=R/'audits/v3/stage2_preremoval_3d_visualcloud_audit.json'
 target.write_text(json.dumps(out,indent=2,ensure_ascii=False))
 print(json.dumps(by),flush=True)
if __name__=='__main__':main()
