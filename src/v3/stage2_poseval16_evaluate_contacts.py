"""Evaluate locked first-frame frequency ranking against unreviewed old simulator contacts.
Simulator data are independent evaluation only. Report end-to-end visible-contact
coverage, including failed localization/surface sampling, and conditional coverage.
"""
from pathlib import Path
import hashlib,json,math,sys,collections
import h5py,numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_audit_preremoval_3d_visualcloud import project
MAN=R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json'
PRED=R/'outputs/v3_stage2/visual_frequency_poseval16'
LOCK=PRED/'predictions_locked.json'
CONTACT=R/'data/v3/evaluation/poseval16_contacts'
OUT=R/'audits/v3/stage2_poseval16_contact_diagnostic.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def calculate(e,lock):
 id=e['id'];path=PRED/'rank_1'/(id+'.npz')
 assert sha(path)==lock['prediction_sha256']['rank_1/'+id]
 with np.load(path) as z:
  xyz=z['xyz'].copy();pixel=z['pixel'].copy()
  box=z['source_box'].copy();score=z['score'].copy()
  assert int(z['query_frame'])==0 and not bool(z['has_future'])
 state=json.loads((CONTACT/e['asset_group']/(e['source_episode']+'_states_contacts.json')).read_text())
 assert state['evaluation_only'] and state['old_contacts_unreviewed']
 assert state['prediction_lock_sha256']==sha(LOCK)
 rows=state['rows'];r0=rows[0]
 p0=np.asarray(r0['object_position']);rot0=np.asarray(r0['object_rotation'])
 local=[]
 for row in rows:
  p=np.asarray(row['object_position']);rot=np.asarray(row['object_rotation'])
  for c in row['contacts']:local.append((np.asarray(c['world'])-p)@rot)
 base=dict(id=id,asset_group=e['asset_group'],object_points=len(xyz),reference_contacts_total=len(local))
 if not local:return dict(base,status='no_simulator_contacts')
 contact=np.asarray(local)@rot0.T+p0
 with h5py.File(e['rgbd_path']) as f:
  depth=f['agentview/depth_m'][0];K=f['agentview/K'][0]
  T=f['agentview/camera_to_world'][0]
 u,v,visible=project(contact,K,T,depth,.02)
 base['depth_visible_contacts']=int(visible.sum())
 if not visible.any():return dict(base,status='no_depth_visible_contacts')
 x1,y1,x2,y2=box
 inside=visible&(u>=x1)&(u<x2)&(v>=y1)&(v<y2)
 base['visible_in_candidate_box']=int(inside.sum())
 cp=contact[inside]
 ix=np.empty(0,np.int64)
 if len(cp):
  near,at=cKDTree(xyz).query(cp)
  ix=at[near<.015]
 base['near_cloud_contacts']=len(ix)
 base['surface_recall']=len(ix)/int(visible.sum())
 baselines={'transfer':score,
  'height_high':xyz[:,2],
  'center3d':-np.linalg.norm(xyz-xyz.mean(0),axis=1),
  'center2d':-np.linalg.norm(pixel-np.asarray([(x1+x2)/2,(y1+y2)/2]),axis=1)}
 for frac in (.1,.2,.3):
  k=max(1,math.ceil(frac*len(xyz)))
  row=dict(random_conditional=k/len(xyz),
   random_end_to_end=(k/len(xyz))*base['surface_recall'])
  for name,s in baselines.items():
   top=np.argpartition(s,-k)[-k:]
   hits=int(np.isin(ix,top).sum())
   row[name+'_hits']=hits
   row[name+'_conditional']=hits/len(ix) if len(ix) else None
   row[name+'_end_to_end']=hits/int(visible.sum())
  base['budget_'+str(int(100*frac))]=row
 base['status']='evaluable' if len(ix) else 'visible_but_surface_missing'
 return base
def summarize(rows,frac):
 key='budget_'+str(frac)
 valid=[x[key] for x in rows if key in x]
 conditional=[x[key] for x in rows if x.get('near_cloud_contacts',0)>0]
 fields=['transfer','height_high','center3d','center2d']
 out=dict(episodes_with_depth_visible=len(valid),episodes_with_near_cloud=len(conditional))
 for f in fields:
  out[f+'_end_to_end_mean']=float(np.mean([x[f+'_end_to_end'] for x in valid])) if valid else None
  out[f+'_conditional_mean']=float(np.mean([x[f+'_conditional'] for x in conditional])) if conditional else None
 out['random_end_to_end_mean']=float(np.mean([x['random_end_to_end'] for x in valid])) if valid else None
 out['random_conditional_mean']=float(np.mean([x['random_conditional'] for x in conditional])) if conditional else None
 return out
def main():
 assert not OUT.exists()
 lock=json.loads(LOCK.read_text())
 assert lock['status']=='poseval16_multicandidate_visual_frequency_locked'
 assert lock['query_manifest_sha256']==sha(MAN)
 assert json.loads((CONTACT/'manifest.json').read_text())['status']=='poseval16_contacts_extracted_for_evaluation_only'
 entries=json.loads(MAN.read_text())['rows'];assert len(entries)==16
 rows=[]
 for e in entries:
  row=calculate(e,lock);rows.append(row);print(json.dumps(row),flush=True)
 summary={str(frac):summarize(rows,frac) for frac in (10,20,30)}
 summary['total_episodes']=len(rows)
 summary['depth_visible_episodes']=sum(x.get('depth_visible_contacts',0)>0 for x in rows)
 summary['near_cloud_episodes']=sum(x.get('near_cloud_contacts',0)>0 for x in rows)
 groups={}
 for g in sorted({x['asset_group'] for x in rows}):
  subset=[x for x in rows if x['asset_group']==g]
  groups[g]={str(frac):summarize(subset,frac) for frac in (10,20,30)}
 report=dict(status='pose_development_old_contact_diagnostic_only',
  reference='unreviewed old simulator contacts; visible contact proximity only; not gripper-wrapped surface',
  predictions_locked_sha256=sha(LOCK),no_simulator_truth_in_training=True,
  summary=summary,by_asset=groups,rows=rows)
 OUT.write_text(json.dumps(report,indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
