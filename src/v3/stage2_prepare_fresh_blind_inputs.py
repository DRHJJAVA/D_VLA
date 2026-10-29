"""Fresh same-asset blind query inputs. Reads query RGB-D at t=0 only; no outcomes or simulator truth."""
from pathlib import Path
import hashlib,json,sys,time
import h5py,torch
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
import stage2_firstframe_template_proposals as proposal
import stage2_build_firstframe_template_val as sample_builder

MANIFEST=R/'data/v3/manifests/stage2_fresh_blind_8.json'
PROPOSALS=R/'data/v3/proposals/first_frame_templates/fresh_blind'
SAMPLES=R/'data/v3/samples/firstframe_template_fresh'
def main():
    manifest=json.loads(MANIFEST.read_text())
    assert manifest['status']=='frozen_before_render_or_prediction'
    rows=manifest['rows']
    assert len(rows)==8 and len({x['id'] for x in rows})==8
    for e in rows:
        with h5py.File(e['rgbd_path']) as f:
            assert f['agentview/rgb'].shape[0]==e['source_frames']
            assert f['agentview/depth_m'].shape[0]==e['source_frames']
    proposal.E.update({e['id']:e for e in rows})
    train_templates=proposal.templates()
    assert len(train_templates)>=15
    PROPOSALS.mkdir(parents=True,exist_ok=True)
    for e in rows:
        dest=PROPOSALS/(e['id']+'.json')
        if dest.exists():
            q=json.loads(dest.read_text())
            assert q['id']==e['id'] and q['no_query_future']
            print(json.dumps(dict(id=e['id'],proposal='reused',candidates=len(q['candidates']))),flush=True)
            continue
        q=proposal.propose(e['id'],train_templates)
        assert q['candidates'] and q['no_query_future']
        dest.write_text(json.dumps(q,indent=2))
        print(json.dumps(dict(id=e['id'],proposal='complete',candidates=len(q['candidates']),top_score=q['candidates'][0]['score'])),flush=True)
    sample_builder.E.update({e['id']:e for e in rows})
    sample_builder.OUT=SAMPLES
    torch.set_num_threads(4)
    net=None
    built=[]
    for e in rows:
        id=e['id'];path=SAMPLES/'val'/(id+'.npz');graph=SAMPLES/'graphs/val'/(id+'.npz')
        if path.exists() and graph.exists():
            built.append(dict(id=id,status='reused'))
            continue
        if path.exists() or graph.exists():
            raise RuntimeError(f'partial sample/graph for {id}')
        if net is None:net=sample_builder.model()
        row=sample_builder.build(id,net,PROPOSALS/(id+'.json'))
        built.append(row)
        print(json.dumps(row),flush=True)
    meta=dict(status='first_frame_inputs_complete',manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),query_frames=[0],train_templates=len(train_templates),rows=built,created_unix=time.time())
    target=SAMPLES/'manifest.json'
    if target.exists():
        old=json.loads(target.read_text())
        assert old['manifest_sha256']==meta['manifest_sha256']
    else:target.write_text(json.dumps(meta,indent=2))
    print(json.dumps(dict(complete=True,episodes=len(built),manifest_sha256=meta['manifest_sha256'])),flush=True)
if __name__=='__main__':main()
