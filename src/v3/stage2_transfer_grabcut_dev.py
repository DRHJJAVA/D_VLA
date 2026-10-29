"""Nonparametric train-demo visual-frequency transfer to development first-frame object clouds.
No simulator truth, names, actions, or query future are read. Freeze scores before audit."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_build_frequency_targets import S,load,transfer
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
DATA=R/'data/v3/samples/grabcut_firstframe_dev'
OUT=R/'outputs/v3_stage2/grabcut_dev_8/transfer'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    entries=json.loads(M.read_text())['rows']
    donors=[load('train',id,True) for id in S['train']]
    assert len(donors)==20
    OUT.mkdir(parents=True,exist_ok=True)
    info=[];hashes={}
    for e in entries:
        id=e['id'];path=DATA/'val'/(id+'.npz')
        with np.load(path) as f:q={k:f[k].copy() for k in ('xyz','dino_feat','pixel','source_box')}
        q['id']=id
        vec=q['dino_feat'].mean(0);q['embedding']=vec/np.linalg.norm(vec)
        score,sources=transfer(q,donors)
        assert np.isfinite(score).all() and len(score)==len(q['xyz'])
        target=OUT/(id+'.npz')
        if target.exists():
            with np.load(target) as old:assert np.array_equal(old['score'],score)
        else:np.savez_compressed(target,score=score,pixel=q['pixel'],xyz=q['xyz'],source_box=q['source_box'],query_frame=np.asarray(0))
        hashes[id]=sha(target)
        row=dict(id=id,points=len(score),score_mean=float(score.mean()),score_max=float(score.max()),donors=sources)
        info.append(row);print(json.dumps(row),flush=True)
    lock=OUT/'transfer_locked.json';assert not lock.exists()
    lock.write_text(json.dumps(dict(status='train_visual_transfer_locked_before_contact_audit',manifest_sha256=sha(M),sample_manifest_sha256=sha(DATA/'manifest.json'),prediction_sha256=hashes,rows=info,created_unix=time.time()),indent=2))
    print(json.dumps(dict(status='locked',episodes=len(info))),flush=True)
if __name__=='__main__':main()
