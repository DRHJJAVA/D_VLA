"""Development-only DINO reranking of first-frame RGB template candidates.
Candidate features use only query frame 0; full-sequence automatic ROI is opened for audit after scoring."""
from pathlib import Path
import argparse,json,sys
import cv2,h5py,numpy as np,torch
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_extract_dino import model,preproc
from stage2_firstframe_template_proposals import overlap
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
ORIG={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
OUT=R/'data/v3/features/dino_template_rerank'
AUDIT=R/'audits/v3/stage2_dino_template_dev_8.json'
def embed(net,rgb,box):
    x1,y1,x2,y2=map(int,box)
    patch=rgb[y1:y2,x1:x2]
    assert min(patch.shape[:2])>=10
    patch=cv2.resize(patch,(448,448),interpolation=cv2.INTER_LINEAR)
    with torch.inference_mode():
        tokens=net.forward_features(preproc(patch[None]))
        cls=tokens[0,0].float()
        avg=tokens[0,1:].float().mean(0)
        cls=cls/cls.norm();avg=avg/avg.norm()
    return np.concatenate((cls.cpu().numpy(),avg.cpu().numpy())).astype(np.float32)
def read_rgb(path):
    with h5py.File(path) as f:return f['agentview/rgb'][0]
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--donors-only",action="store_true");parser.add_argument("--proposal-kind",choices=("original","masked"),default="original");args=parser.parse_args()
    torch.set_num_threads(4);net=model()
    cache=OUT/'train_embeddings.npz'
    if cache.exists():
        with np.load(cache) as f:donor_ids=f['ids'].tolist();donors=f['embedding'].copy()
        assert donor_ids==S['train']
    else:
        donor_ids=S['train'];vectors=[]
        for id in donor_ids:
            e=ORIG[id];rgb=read_rgb(e['rgbd_path'])
            box=json.loads((R/'data/v3/proposals/motion_rgbd_rerank'/e['asset_group']/(e['source_episode']+'.json')).read_text())['selected']['box']
            vectors.append(embed(net,rgb,box))
            print(json.dumps(dict(donor=id,ready=True)),flush=True)
        donors=np.stack(vectors)
        OUT.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,ids=np.asarray(donor_ids),embedding=donors)
    if args.donors_only:
        print(json.dumps(dict(status="donor_embeddings_complete",count=len(donor_ids))),flush=True)
        return
    manifest=json.loads(M.read_text());rows=[]
    for e in manifest['rows']:
        id=e['id']
        folder='first_frame_templates' if args.proposal_kind=='original' else 'first_frame_masked_templates'
        prop=json.loads((R/'data/v3/proposals'/folder/'dev'/(id+'.json')).read_text())
        assert prop['no_query_future']
        rgb=read_rgb(e['rgbd_path'])
        embeddings=np.stack([embed(net,rgb,c['box']) for c in prop['candidates']])
        cls=embeddings[:,:384]@donors[:,:384].T
        avg=embeddings[:,384:]@donors[:,384:].T
        # Candidate scoring uses no names, labels, or query future.
        features=[]
        for i,c in enumerate(prop['candidates']):
            features.append(dict(box=c['box'],template_score=c['score'],dino_cls_max=float(cls[i].max()),dino_avg_max=float(avg[i].max()),dino_cls_top5=float(np.sort(cls[i])[-5:].mean()),dino_avg_top5=float(np.sort(avg[i])[-5:].mean())))
        reference=json.loads((R/'data/v3/proposals/motion_rgbd_rerank'/e['asset_group']/(e['source_episode']+'.json')).read_text())['selected']['box']
        ious=np.asarray([overlap(c['box'],reference) for c in features])
        result=dict(id=id,weak_reference='full-sequence automatic visual ROI; evaluation only',candidate_count=len(features),rows=features,ious=ious.tolist())
        rows.append(result)
        print(json.dumps(dict(id=id,candidates=len(features),max_iou=float(ious.max()),ranks={key:int(np.argmax([x[key] for x in features]))+1 for key in ('template_score','dino_cls_max','dino_avg_max','dino_cls_top5','dino_avg_top5')})),flush=True)
    metrics={}
    for key in ('template_score','dino_cls_max','dino_avg_max','dino_cls_top5','dino_avg_top5'):
        values=[]
        for q in rows:
            rank=int(np.argmax([c[key] for c in q['rows']]))
            values.append(q['ious'][rank])
        metrics[key]=dict(top1_iou_ge_05=sum(v>=.5 for v in values),mean_iou=float(np.mean(values)),per_case=[round(v,3) for v in values])
    target=AUDIT if args.proposal_kind=='original' else R/'audits/v3/stage2_dino_masked_template_dev_8.json'
    target.write_text(json.dumps(dict(status='development_only',proposal_kind=args.proposal_kind,proposal_features_query_frame_0_only=True,simulator_truth_used=False,metrics=metrics,cases=rows),indent=2))
    print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
