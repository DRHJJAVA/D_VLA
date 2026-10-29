"""Exploratory old-contact diagnostic after stable-cue pose predictions were locked.
Previously viewed episodes; old simulator contacts are evaluation-only and unreviewed.
Measures only depth-visible contact-point proximity, not wrapped grasp surface.
"""
from pathlib import Path
import hashlib,json,math,collections,sys
import h5py,numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_audit_preremoval_3d_visualcloud import project
MAN=R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json'
NEW=R/'outputs/v3_stage2/stable_cue_object_only_poseval16'
OLD=R/'outputs/v3_stage2/object_only_frequency_v1'
CONTACT=R/'data/v3/evaluation/poseval16_contacts'
OLD_LOCK=R/'outputs/v3_stage2/visual_frequency_poseval16/predictions_locked.json'
OUT=R/'audits/v3/stage2_stable_cue_object_only_poseval16_contact_exploratory.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(score,ix,fraction):
 k=max(1,math.ceil(fraction*len(score)))
 top=np.argpartition(score,-k)[-k:]
 return float(np.isin(ix,top).mean()) if len(ix) else None
def main():
 assert not OUT.exists()
 nlock=json.loads((NEW/'predictions_locked.json').read_text())
 olock=json.loads((OLD/'predictions_locked.json').read_text())
 assert nlock['status']=='stable_cue_object_only_poseval16_locked'
 assert nlock['no_simulator_truth'] and nlock['no_query_future']
 entries=json.loads(MAN.read_text())['rows'];assert len(entries)==16
 rows=[]
 for e in entries:
  id=e['id'];npred=NEW/(id+'.npz');opred=OLD/'poseval16'/(id+'.npz')
  assert sha(npred)==nlock['prediction_sha256'][id]
  assert sha(opred)==olock['prediction_sha256']['poseval16/'+id]
  with np.load(npred) as z:
   xyz=z['xyz'].copy();pixel=z['pixel'].copy();ns=z['score'].copy()
   assert int(z['query_frame'])==0 and not bool(z['has_query_future'])
  with np.load(opred) as z:
   os=z['score'].copy()
   assert np.array_equal(pixel,z['pixel'])
  state=json.loads((CONTACT/e['asset_group']/(e['source_episode']+'_states_contacts.json')).read_text())
  assert state['evaluation_only'] and state['old_contacts_unreviewed']
  assert state['prediction_lock_sha256']==sha(OLD_LOCK)
  srows=state['rows'];r0=srows[0]
  p0=np.asarray(r0['object_position']);rot0=np.asarray(r0['object_rotation'])
  local=[]
  for sr in srows:
   p=np.asarray(sr['object_position']);rot=np.asarray(sr['object_rotation'])
   for c in sr['contacts']:local.append((np.asarray(c['world'])-p)@rot)
  row=dict(id=id,asset_group=e['asset_group'],points=len(xyz),contacts=len(local))
  if not local:row['status']='no_contacts';rows.append(row);continue
  contact=np.asarray(local)@rot0.T+p0
  with h5py.File(e['rgbd_path']) as f:
   depth=f['agentview/depth_m'][0];K=f['agentview/K'][0];T=f['agentview/camera_to_world'][0]
  u,v,visible=project(contact,K,T,depth,.02)
  row['depth_visible']=int(visible.sum())
  if not visible.any():row['status']='not_depth_visible';rows.append(row);continue
  near,ix=cKDTree(xyz).query(contact[visible])
  ix=ix[near<.015]
  row['near_cloud']=len(ix)
  row['surface_recall']=len(ix)/int(visible.sum())
  for n in (10,20,30):
   key='budget_'+str(n);f=n/100
   row[key]=dict(stable_cue_transfer=metric(ns,ix,f),
    prior_broad_cue_transfer=metric(os,ix,f),
    random=math.ceil(f*len(xyz))/len(xyz))
  row['status']='evaluable' if len(ix) else 'surface_missing'
  rows.append(row);print(json.dumps(row),flush=True)
 valid=[x for x in rows if x['status']=='evaluable']
 def agg(rr):
  return {str(n):{k:float(np.mean([x['budget_'+str(n)][k] for x in rr])) if rr else None
   for k in ('stable_cue_transfer','prior_broad_cue_transfer','random')} for n in (10,20,30)}
 groups={g:agg([x for x in valid if x['asset_group']==g]) for g in sorted(set(x['asset_group'] for x in rows))}
 report=dict(status='previously_viewed_poseval16_exploratory_contact_check',
  reference='unreviewed old simulator contact points, visible proximity only',
  no_simulator_truth_in_training=True,no_query_future=True,
  old_contact_data_evaluation_only=True,not_new_blind_test=True,
  prediction_lock_sha256=sha(NEW/'predictions_locked.json'),
  episodes=16,evaluable=len(valid),summary=agg(valid),by_asset=groups,rows=rows)
 OUT.write_text(json.dumps(report,indent=2))
 print(json.dumps(dict(evaluable=len(valid),summary=report['summary'],by_asset=groups)),flush=True)
if __name__=='__main__':main()
