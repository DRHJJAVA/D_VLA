"""Score name-free first-frame pose-validation proposals with frozen TRAIN DINO crops.
All query evidence comes from frame zero. Retain ranked alternatives rather than forcing a target.
No simulator state, contacts, object IDs, text, or future query frames enter scoring.
"""
from pathlib import Path
import json,hashlib,sys
import h5py,numpy as np,torch
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_extract_dino import model
from stage2_dino_template_dev import embed
MAN=R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json'
PROP=R/'data/v3/proposals/first_frame_masked_templates/poseval16'
DONOR=R/'data/v3/features/dino_template_rerank/train_embeddings.npz'
OUT=R/'data/v3/proposals/firstframe_dino_candidates/poseval16'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    entries=json.loads(MAN.read_text())['rows']
    assert len(entries)==16 and len({e['id'] for e in entries})==16
    with np.load(DONOR) as f:
        ids=f['ids'].tolist();donors=f['embedding'].copy()
    assert len(ids)==len(donors)==20 and donors.shape[1]==768
    torch.set_num_threads(4)
    net=None;rows=[]
    for e in entries:
        id=e['id'];p=PROP/(id+'.json');q=json.loads(p.read_text())
        assert q['no_query_future'] and q['frame_index']==0 and q['candidates']
        dest=OUT/(id+'.json')
        if dest.exists():
            old=json.loads(dest.read_text())
            assert old['proposal_sha256']==sha(p) and old['query_frame']==0
            assert len(old['candidates'])==len(q['candidates'])
            rows.append(dict(id=id,status='reused',candidates=len(q['candidates'])))
            continue
        if net is None:net=model()
        with h5py.File(e['rgbd_path']) as f:rgb=f['agentview/rgb'][0]
        vec=np.stack([embed(net,rgb,c['box']) for c in q['candidates']])
        cls=vec[:,:384]@donors[:,:384].T
        avg=vec[:,384:]@donors[:,384:].T
        candidates=[]
        for i,c in enumerate(q['candidates']):
            candidates.append(dict(rank_by_template=i+1,box=c['box'],template_score=c['score'],
                dino_cls_max=float(cls[i].max()),dino_avg_max=float(avg[i].max()),
                dino_cls_top5=float(np.sort(cls[i])[-5:].mean()),
                dino_avg_top5=float(np.sort(avg[i])[-5:].mean())))
        order=np.argsort(-np.asarray([x['dino_cls_max'] for x in candidates]))
        result=dict(id=id,query_frame=0,no_query_future=True,no_simulator_truth=True,
            no_names=True,proposal_sha256=sha(p),donor_sha256=sha(DONOR),
            method='masked TRAIN RGB template proposals; frozen DINO CLS rerank',
            ranked_candidate_indices=order.tolist(),candidates=candidates)
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_suffix('.partial')
        if tmp.exists():raise RuntimeError('inspect partial: '+str(tmp))
        tmp.write_text(json.dumps(result,indent=2))
        tmp.replace(dest)
        row=dict(id=id,status='scored',candidates=len(candidates),
            top1_box=candidates[int(order[0])]['box'],top1_dino_cls=float(candidates[int(order[0])]['dino_cls_max']))
        rows.append(row);print(json.dumps(row),flush=True)
    manifest=OUT/'manifest.json'
    record=dict(status='poseval16_firstframe_dino_candidates_complete',query_frame=0,
        no_query_future=True,no_simulator_truth=True,episodes=len(rows),
        pose_validation_manifest_sha256=sha(MAN),rows=rows)
    if manifest.exists():
        old=json.loads(manifest.read_text())
        assert old['pose_validation_manifest_sha256']==record['pose_validation_manifest_sha256']
    else:manifest.write_text(json.dumps(record,indent=2))
    print(json.dumps(dict(status='complete',episodes=len(rows))),flush=True)
if __name__=='__main__':main()
