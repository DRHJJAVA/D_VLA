"""Independent old simulator contact extraction on pose-development episodes.
Runs only after all first-frame visual-frequency predictions are hash-locked.
Evaluation only; unreviewed contacts never enter training or inference.
"""
from pathlib import Path
import hashlib,json,sys
import h5py
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v2'))
import stage1_collect as stage1
MAN=R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json'
LOCK=R/'outputs/v3_stage2/visual_frequency_poseval16/predictions_locked.json'
OUT=R/'data/v3/evaluation/poseval16_contacts'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lock=json.loads(LOCK.read_text())
 assert lock['status']=='poseval16_multicandidate_visual_frequency_locked'
 assert lock['query_manifest_sha256']==sha(MAN)
 entries=json.loads(MAN.read_text())['rows'];assert len(entries)==16
 rows=[]
 for group in sorted({x['asset_group'] for x in entries}):
  subset=[e for e in entries if e['asset_group']==group]
  assert len(subset)==4
  with h5py.File(subset[0]['source_path'],'r') as f:
   env=stage1.OffScreenRenderEnv(bddl_file_name=str(stage1.resolve(f['data'])),use_camera_obs=False)
   try:
    for e in subset:
     dest=OUT/group/(e['source_episode']+'_states_contacts.json')
     if dest.exists():
      old=json.loads(dest.read_text())
      assert old['episode']==e['source_episode'] and len(old['rows'])==e['source_frames']
      assert old['prediction_lock_sha256']==sha(LOCK)
      row=dict(id=e['id'],status='reused',frames=len(old['rows']))
     else:
      ep=f['data'][e['source_episode']]
      stage1.restore(env,ep)
      rr,candidates,error=stage1.inspect_episode(env,ep,group)
      assert len(rr)==e['source_frames'] and error<1e-5
      stage1.save(dest,dict(episode=e['source_episode'],rows=rr,candidates=candidates,
       evaluation_only=True,old_contacts_unreviewed=True,
       prediction_lock_sha256=sha(LOCK),roundtrip_max_error=error))
      row=dict(id=e['id'],status='complete',frames=len(rr),
       contact_frames=sum(bool(x['contacts']) for x in rr),roundtrip_max_error=error)
     rows.append(row);print(json.dumps(row),flush=True)
   finally:env.close()
 manifest=OUT/'manifest.json'
 data=dict(status='poseval16_contacts_extracted_for_evaluation_only',
  old_contacts_unreviewed=True,source_manifest_sha256=sha(MAN),
  prediction_lock_sha256=sha(LOCK),rows=rows)
 if manifest.exists():
  old=json.loads(manifest.read_text())
  assert old['source_manifest_sha256']==data['source_manifest_sha256']
 else:manifest.write_text(json.dumps(data,indent=2))
 print(json.dumps(dict(status='complete',episodes=len(rows))),flush=True)
if __name__=='__main__':main()
