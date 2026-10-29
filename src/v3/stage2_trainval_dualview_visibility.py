"""RGB-D only visibility audit around automatic object removal events.
Uses no simulator truth, robot actions, names, manual labels or held-out episodes.
Quantifies whether first-frame object surface and near-contact geometry are
visually observable from agentview and wrist camera before motion.
"""
from pathlib import Path
import json
import h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
SRC=R/'data/v3/samples/visual_interaction_uniform448_3dcc'
COMP=R/'data/v3/proposals/foreground3d_components'
OUT=R/'audits/v3/stage2_trainval_dualview_visibility_prelim.json'
def proj(xyz,K,T,depth):
 cam=(xyz-T[:3,3])@T[:3,:3]
 z=cam[:,2]
 safe=np.where(z>.01,z,1)
 u=np.rint(K[0,0]*cam[:,0]/safe+K[0,2]).astype(int)
 v=np.rint(K[1,1]*cam[:,1]/safe+K[1,2]).astype(int)
 inside=(z>.05)&(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
 delta=np.full(len(xyz),np.nan,dtype=np.float32)
 ii=np.flatnonzero(inside)
 delta[ii]=depth[v[ii],u[ii]]-z[ii]
 return inside,delta
def main():
 assert not OUT.exists()
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   with np.load(SRC/split/(id+'.npz')) as z:
    xyz=z['xyz'].copy();onsets=z['onsets'].copy()
   with np.load(COMP/split/(id+'.npz')) as z:path=Path(str(z['rgbd_path']))
   with h5py.File(path) as f:
    per=[]
    for event in onsets:
     t=int(event)-1
     a=f['agentview/depth_m'][t]
     ak=f['agentview/K'][t];at=f['agentview/camera_to_world'][t]
     w=f['robot0_eye_in_hand/depth_m'][t]
     wk=f['robot0_eye_in_hand/K'][t];wt=f['robot0_eye_in_hand/camera_to_world'][t]
     ai,ad=proj(xyz,ak,at,a)
     wi,wd=proj(xyz,wk,wt,w)
     av=ai&(np.abs(ad)<.015)
     wv=wi&(np.abs(wd)<.015)
     ao=ai&(ad<-.02)
     ar=ai&(ad>.025)
     per.append(dict(event=int(event),frame=t,points=len(xyz),
      agent_in_frame=float(ai.mean()),agent_visible=float(av.mean()),
      agent_occluded_by_nearer_surface=float(ao.mean()),
      agent_removed_or_moved=float(ar.mean()),
      wrist_in_frame=float(wi.mean()),wrist_visible=float(wv.mean()),
      either_view_visible=float((av|wv).mean()),
      wrist_recovers_agent_occlusion=float((ao&wv).mean())))
   row=dict(id=id,split=split,events=per)
   rows.append(row);print(json.dumps(row),flush=True)
 summary={}
 for split in ('train','val'):
  ee=[v for r in rows if r['split']==split for v in r['events']]
  summary[split]=dict(episodes=sum(r['split']==split for r in rows),events=len(ee))
  for key in ('agent_visible','agent_occluded_by_nearer_surface','agent_removed_or_moved',
   'wrist_in_frame','wrist_visible','either_view_visible','wrist_recovers_agent_occlusion'):
   summary[split][key+'_mean']=float(np.mean([x[key] for x in ee]))
 report=dict(status='trainval_dualview_rgbd_visibility_preliminary',
  source='complete RGB-D demonstration sequences',
  reference='first-frame visible object surface projected into each view at frame immediately before stable removal',
  no_simulator_truth=True,no_robot_actions=True,no_names=True,
  caution='wrist camera can see new unseen object sides that are not represented by first-frame agentview points; these fractions are not contact observability or grasp performance',
  summary=summary,rows=rows)
 OUT.write_text(json.dumps(report,indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
