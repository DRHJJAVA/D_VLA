"""Freeze fresh-test model predictions before opening any query future or evaluation truth."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np,torch
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src'))
sys.path.insert(0,str(R/'src/v3'))
from pointnet2_grasp import PointNet2Grasp
from stage2_train_frequency_student import tensor

MANIFEST=R/'data/v3/manifests/stage2_fresh_blind_8.json'
DATA=R/'data/v3/samples/firstframe_template_fresh'
OUT=R/'outputs/v3_stage2/fresh_blind_8'
SEEDS=(2026,2027,2028)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
    return h.hexdigest()
def main():
    torch.set_num_threads(4)
    manifest=json.loads(MANIFEST.read_text())
    assert manifest['status']=='frozen_before_render_or_prediction'
    ids=[e['id'] for e in manifest['rows']]
    assert len(ids)==8 and len(set(ids))==8
    lock=OUT/'predictions_locked.json'
    if lock.exists():
        old=json.loads(lock.read_text())
        assert old['manifest_sha256']==sha(MANIFEST)
        for id in ids:
            path=OUT/'predictions'/(id+'.npz')
            assert old['prediction_sha256'][id]==sha(path)
        print(json.dumps(dict(status='already_locked',episodes=8)),flush=True)
        return
    models=[];checkpoints={}
    for seed in SEEDS:
        ckpath=R/'outputs/v3_stage2/frequency_student'/('seed_'+str(seed))/'best.pt'
        checkpoints[str(seed)]=sha(ckpath)
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        net=PointNet2Grasp().eval()
        net.load_state_dict(ck['state'])
        models.append(net)
    pred_dir=OUT/'predictions';pred_dir.mkdir(parents=True,exist_ok=True)
    pred_hashes={}
    for id in ids:
        sample_path=DATA/'val'/(id+'.npz')
        graph_path=DATA/'graphs/val'/(id+'.npz')
        with np.load(sample_path) as f:s={k:f[k].copy() for k in ('xyz','dino_feat','pixel','source_box')}
        with np.load(graph_path) as f:g={k:f[k].copy() for k in f.files}
        s['target']=np.zeros(len(s['xyz']),dtype=np.float32)
        output=pred_dir/(id+'.npz')
        sample_hash=sha(sample_path);graph_hash=sha(graph_path)
        if output.exists():
            with np.load(output) as old:
                assert str(old['sample_sha256'])==sample_hash
                assert str(old['graph_sha256'])==graph_hash
                assert str(old['checkpoints_json'])==json.dumps(checkpoints,sort_keys=True)
                assert len(old['score'])==len(s['xyz'])
            pred_hashes[id]=sha(output)
            print(json.dumps(dict(id=id,status='reused',points=len(s['xyz']))),flush=True)
            continue
        xyz,feat,_,gg=tensor((id,s,g),'cpu')
        with torch.inference_mode():
            logits=np.stack([net(xyz,feat,gg).detach().numpy() for net in models],axis=0)
        assert logits.shape==(3,len(s['xyz'])) and np.isfinite(logits).all()
        score=logits.mean(axis=0).astype(np.float32)
        tmp=output.with_suffix('.partial')
        with tmp.open('wb') as f:
            np.savez_compressed(f,score=score,member_logits=logits.astype(np.float32),xyz=s['xyz'],pixel=s['pixel'],source_box=s['source_box'],sample_sha256=sample_hash,graph_sha256=graph_hash,checkpoints_json=json.dumps(checkpoints,sort_keys=True),query_frame=np.asarray(0),has_future=np.asarray(False))
        tmp.replace(output)
        pred_hashes[id]=sha(output)
        print(json.dumps(dict(id=id,status='predicted',points=len(score),score_min=float(score.min()),score_max=float(score.max()))),flush=True)
    record=dict(status='predictions_locked_before_evaluation',manifest_sha256=sha(MANIFEST),sample_manifest_sha256=sha(DATA/'manifest.json'),checkpoint_sha256=checkpoints,prediction_sha256=pred_hashes,episodes=ids,method='equal raw-logit average; first RGB-D frame, first-frame TRAIN-template ROI, frozen DINOv2, fixed 3 checkpoints',budgets=manifest['budgets'],no_query_future=True,no_simulator_truth=True,locked_unix=time.time())
    temp=lock.with_suffix('.partial')
    temp.write_text(json.dumps(record,indent=2))
    temp.replace(lock)
    print(json.dumps(dict(status=record['status'],episodes=len(ids),lock=str(lock))),flush=True)
if __name__=='__main__':main()
