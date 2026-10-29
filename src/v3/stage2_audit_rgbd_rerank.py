"""Evaluation-only proposal audit against saved simulator geometry; never feeds learner."""
from pathlib import Path
import json
import h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
rows=[json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]
out=[]
for x in rows:
 asset=x['asset_group'];ep=x['source_episode']
 proposal=R/'data/v3/proposals/motion_rgbd_rerank'/asset/(ep+'.json')
 assert proposal.exists(),proposal
 q=json.loads(proposal.read_text());box=q['selected']['box'] if q['selected'] else None
 st=json.loads((R/'data/v2/stage1'/asset/(ep+'_states_contacts.json')).read_text())['rows'][0]
 with np.load(R/'data/v2/stage1'/asset/'surface.npz') as f:local=f['points_object'][::10]
 world=local@np.asarray(st['object_rotation']).T+np.asarray(st['object_position'])
 with h5py.File(x['rgbd_path'],'r') as f:
  g=f['agentview'];K=g['K'][0];T=g['camera_to_world'][0];depth=g['depth_m'][0]
 camera=(world-T[:3,3])@T[:3,:3];z=camera[:,2];safe=np.where(z>0,z,1)
 u=np.rint(K[0,0]*camera[:,0]/safe+K[0,2]).astype(int);v=np.rint(K[1,1]*camera[:,1]/safe+K[1,2]).astype(int)
 good=(z>0)&(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0]);ids=np.flatnonzero(good)
 ids=ids[abs(depth[v[ids],u[ids]]-z[ids])<.015]
 assert len(ids)>100,(x['id'],len(ids))
 gt=[int(u[ids].min()),int(v[ids].min()),int(u[ids].max()+1),int(v[ids].max()+1)]
 if box:
  intersection=max(0,min(box[2],gt[2])-max(box[0],gt[0]))*max(0,min(box[3],gt[3])-max(box[1],gt[1]))
  area=lambda b:(b[2]-b[0])*(b[3]-b[1])
  iou=intersection/(area(box)+area(gt)-intersection)
  center_inside=box[0]<=.5*(gt[0]+gt[2])<=box[2] and box[1]<=.5*(gt[1]+gt[3])<=box[3]
 else:iou=0;center_inside=False
 out.append(dict(id=x['id'],asset=asset,proposal_box=box,reference_box=gt,iou=float(iou),center_inside=bool(center_inside),candidate_count=q['original_candidate_count']))
 print(json.dumps(out[-1]),flush=True)
summary=dict(status='audited',episodes=len(out),iou_ge_0_5=sum(z['iou']>=.5 for z in out),reference_center_inside=sum(z['center_inside'] for z in out),median_iou=float(np.median([z['iou'] for z in out])),independent_reference='simulator mesh/pose used only here, not proposal creation or training',rows=out)
(R/'audits/v3/stage2_rgbd_rerank_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
