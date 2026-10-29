"""Evaluation-only simulator contact extraction for frozen fresh-test predictions.
Must run only after predictions_locked.json exists. Never used by training or inference."""
from pathlib import Path
import json,sys
import h5py
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v2'))
import stage1_collect as stage1
MANIFEST=R/'data/v3/manifests/stage2_fresh_blind_8.json'
LOCK=R/'outputs/v3_stage2/fresh_blind_8/predictions_locked.json'
OUT=R/'data/v3/evaluation/fresh_blind_contacts'
def main():
    assert LOCK.exists(),'Prediction lock required before opening simulator states'
    lock=json.loads(LOCK.read_text())
    assert lock['status']=='predictions_locked_before_evaluation'
    rows=json.loads(MANIFEST.read_text())['rows']
    for group in sorted({x['asset_group'] for x in rows}):
        subset=[x for x in rows if x['asset_group']==group]
        with h5py.File(subset[0]['source_path'],'r') as f:
            env=stage1.OffScreenRenderEnv(bddl_file_name=str(stage1.resolve(f['data'])),use_camera_obs=False)
            try:
                for e in subset:
                    dest=OUT/group/(e['source_episode']+'_states_contacts.json')
                    if dest.exists():
                        old=json.loads(dest.read_text())
                        assert old['episode']==e['source_episode'] and len(old['rows'])==e['source_frames']
                        print(json.dumps(dict(id=e['id'],status='reused')),flush=True)
                        continue
                    ep=f['data'][e['source_episode']]
                    stage1.restore(env,ep)
                    rr,candidates,error=stage1.inspect_episode(env,ep,group)
                    assert len(rr)==e['source_frames'] and error<1e-5
                    stage1.save(dest,dict(episode=e['source_episode'],rows=rr,candidates=candidates,evaluation_only=True,old_contacts_unreviewed=True,prediction_lock=str(LOCK),roundtrip_max_error=error))
                    print(json.dumps(dict(id=e['id'],status='complete',frames=len(rr),contact_frames=sum(bool(r['contacts']) for r in rr),roundtrip_max_error=error)),flush=True)
            finally:env.close()
if __name__=='__main__':main()
