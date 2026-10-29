"""Evaluation only: does a frozen wrist-camera approach cue rank visible object-surface contacts?
Simulator geometry and contacts are read only here, never by proposal or learner."""
from pathlib import Path
import json,math
import h5py,numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
R=Path('/mnt/nvme1/drh/D_VLA')
entries={x['id']:x for x in [json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z]}
splits=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
splitof={id:k for k,ids in splits.items() for id in ids}
def project(world,K,T,depth,tol=.015):
 cam=(world-T[:3,3])@T[:3,:3];z=cam[:,2];safe=np.where(z>0,z,1)
 u=np.rint(K[0,0]*cam[:,0]/safe+K[0,2]).astype(int)
 v=np.rint(K[1,1]*cam[:,1]/safe+K[1,2]).astype(int)
 valid=(z>0)&(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
 ids=np.flatnonzero(valid);visible=np.zeros(len(world),bool)
 visible[ids]=abs(depth[v[ids],u[ids]]-z[ids])<tol
 return u,v,visible
def calc(e):
 asset=e['asset_group'];ep=e['source_episode']
 q=json.loads((R/'data/v3/proposals/motion_rgbd_rerank'/asset/(ep+'.json')).read_text())
 box=q['selected']['box']
 state=json.loads((R/'data/v2/stage1'/asset/(ep+'_states_contacts.json')).read_text())
 r0=state['rows'][0];p0=np.asarray(r0['object_position']);rot0=np.asarray(r0['object_rotation'])
 with np.load(R/'data/v2/stage1'/asset/'surface.npz') as f:local=f['points_object'][::10]
 world=local@rot0.T+p0
 contactlocal=[]
 for row in state['rows']:
  p=np.asarray(row['object_position']);rot=np.asarray(row['object_rotation'])
  for c in row['contacts']:contactlocal.append((np.asarray(c['world'])-p)@rot)
 if not contactlocal:return dict(id=e['id'],split=splitof[e['id']],reason='no reference contacts')
 contactworld=np.asarray(contactlocal)@rot0.T+p0
 with h5py.File(e['rgbd_path']) as f:
  depthseq=f['agentview/depth_m'][:];depth=depthseq[0];K=f['agentview/K'][0];T=f['agentview/camera_to_world'][0]
  wrist=f['robot0_eye_in_hand/camera_to_world'][:,:3,3]
  u,v,visible=project(world,K,T,depth)
  visible &= (u>=box[0])&(u<box[2])&(v>=box[1])&(v<box[3])
  cu,cv,cvisible=project(contactworld,K,T,depth,.02)
  cvisible &= (cu>=box[0])&(cu<box[2])&(cv>=box[1])&(cv<box[3])
 pts=world[visible]
 if len(pts)<20:return dict(id=e['id'],split=splitof[e['id']],reason='few visible object points',count=len(pts))
 cp=contactworld[cvisible]
 if len(cp)==0:return dict(id=e['id'],split=splitof[e['id']],reason='no visible first-pose reference contacts',object_points=len(pts))
 nearest,ix=cKDTree(pts).query(cp)
 ix=ix[nearest<.015]
 if len(ix)==0:return dict(id=e['id'],split=splitof[e['id']],reason='no near-surface contacts',object_points=len(pts),visible_contacts=len(cp))
 # Image-only event cue: object-depth removal onset, then closer-depth occlusion just before it.
 x1,y1,x2,y2=box
 crop=depthseq[:,y1:y2,x1:x2]
 ring=depth[max(0,y1-8):min(depth.shape[0],y2+8),max(0,x1-8):min(depth.shape[1],x2+8)].copy()
 ry=y1-max(0,y1-8);rx=x1-max(0,x1-8);ring[ry:ry+y2-y1,rx:rx+x2-x1]=np.nan
 background=float(np.nanmedian(ring))
 foreground=depth[y1:y2,x1:x2]<background-.015
 if foreground.sum()<20:return dict(id=e['id'],split=splitof[e['id']],reason='no RGB-D foreground')
 removed=((crop>crop[0:1]+.025)&foreground[None]).sum((1,2))/foreground.sum()
 candidates=[i for i in range(1,len(removed)-2) if np.all(removed[i:i+3]>.2)]
 if not candidates:return dict(id=e['id'],split=splitof[e['id']],reason='no sustained removal onset',object_points=len(pts))
 onset=candidates[0]
 begin=max(0,onset-15)
 refdepth=depth[v[visible],u[visible]]
 # Broaden to nearby visible object surface; contacts can lie next to a finger silhouette.
 maps=[]
 for frame in range(begin,onset):
  near=(crop[frame]<crop[0]-.02)&foreground
  distance=distance_transform_edt(~near)
  yy=v[visible]-y1;xx=u[visible]-x1
  maps.append(np.exp(-distance[yy,xx]**2/(2*6.0**2)))
 score=np.mean(maps,axis=0)
 budget=max(1,math.ceil(.2*len(pts)));top=np.argpartition(score,-budget)[-budget:]
 coverage=float(np.isin(ix,top).mean())
 return dict(id=e['id'],split=splitof[e['id']],object_points=len(pts),reference_contacts=len(contactlocal),visible_contacts=len(cp),evaluable_contacts=len(ix),top20_coverage=coverage,random_budget=budget/len(pts),score_mean=float(score.mean()),score_max=float(score.max()),removal_onset=int(onset),pre_window=[int(begin),int(onset)])
def main():
 results=[]
 for e in entries.values():
  if splitof[e['id']]=='test':continue
  r=calc(e);results.append(r);print(json.dumps(r),flush=True)
 by={}
 for split in ('train','val','test'):
  rr=[x for x in results if x['split']==split and 'top20_coverage' in x]
  by[split]=dict(total=len(splits[split]),evaluable=len(rr),mean_coverage=float(np.mean([x['top20_coverage'] for x in rr])) if rr else None,mean_random_budget=float(np.mean([x['random_budget'] for x in rr])) if rr else None)
 out=dict(status='evaluation_only',cue='RGB-D distance within 6 pixels of closer-depth finger silhouette before removal onset',budget='top 20% visible reference object mesh points',contacts='old unreviewed simulator records projected to first pose; never training',partition=by,rows=results)
 target=R/'audits/v3/stage2_preremoval_proximity_audit.json'
 target.write_text(json.dumps(out,indent=2,ensure_ascii=False))
 print(json.dumps(by),flush=True)
if __name__=='__main__':main()
