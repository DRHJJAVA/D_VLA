"""Name-free first-frame proposals using RGB template pixels masked by TRAIN depth foreground.
Query: frame-0 RGB only. TRAIN template ROI comes from full-sequence visual localization.
No names, query future, actions, simulator object truth or contacts."""
from pathlib import Path
import json,sys
import cv2,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_firstframe_template_proposals import overlap
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
E={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
OUT=R/'data/v3/proposals/first_frame_masked_templates'
def train_templates():
    donors=[]
    for id in S['train']:
        e=E[id]
        box=json.loads((R/'data/v3/proposals/motion_rgbd_rerank'/e['asset_group']/(e['source_episode']+'.json')).read_text())['selected']['box']
        x1,y1,x2,y2=box
        with h5py.File(e['rgbd_path']) as f:
            rgb=f['agentview/rgb'][0];depth=f['agentview/depth_m'][0]
        ring=depth[max(0,y1-8):min(depth.shape[0],y2+8),max(0,x1-8):min(depth.shape[1],x2+8)].copy()
        ry=y1-max(0,y1-8);rx=x1-max(0,x1-8)
        ring[ry:ry+y2-y1,rx:rx+x2-x1]=np.nan
        background=float(np.nanmedian(ring))
        dep=depth[y1:y2,x1:x2]
        mask=((dep>.05)&(dep<background-.015)).astype('uint8')
        mask=cv2.erode(mask,np.ones((3,3),'uint8'),iterations=1)
        if mask.sum()<30:continue
        patch=rgb[y1:y2,x1:x2]
        donors.append((id,patch,mask))
    assert len(donors)>=15
    # Select visually diverse training exemplars without asset names.
    cache=R/'data/v3/features/dino_template_rerank/train_embeddings.npz'
    with np.load(cache) as z:
        ids=z['ids'].tolist();vec=z['embedding'][:,:384].copy()
    lookup={id:i for i,id in enumerate(ids)}
    feat=np.stack([vec[lookup[d[0]]] for d in donors]);sim=feat@feat.T
    chosen=[int(np.argmax(sim.mean(axis=1)))]
    while len(chosen)<min(10,len(donors)):
        dist=sim[:,chosen].max(axis=1);dist[chosen]=np.inf
        chosen.append(int(np.argmin(dist)))
    print(json.dumps(dict(selected_train_templates=[donors[i][0] for i in chosen])),flush=True)
    return [donors[i] for i in chosen]
def propose(id,donors):
    with h5py.File(E[id]['rgbd_path']) as f:query=f['agentview/rgb'][0]
    query=query.astype(np.float32)/255.
    candidates=[]
    for donor,patch,mask in donors:
        if donor==id:continue
        for scale in (.70,.85,1.,1.15,1.30):
            h,w=patch.shape[:2]
            H=max(12,int(round(h*scale)));W=max(12,int(round(w*scale)))
            if H>=query.shape[0] or W>=query.shape[1]:continue
            templ=cv2.resize(patch,(W,H),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR).astype(np.float32)/255.
            mm=cv2.resize(mask,(W,H),interpolation=cv2.INTER_NEAREST).astype('uint8')
            if mm.sum()<25:continue
            for flip in (False,True):
                tt=cv2.flip(templ,1) if flip else templ
                m=cv2.flip(mm,1) if flip else mm
                response=cv2.matchTemplate(query,tt,cv2.TM_CCORR_NORMED,mask=m)
                response=np.nan_to_num(response,nan=-1.,posinf=-1.,neginf=-1.)
                for _ in range(2):
                    _,score,_,at=cv2.minMaxLoc(response)
                    x,y=at
                    if score>0 and y>30:
                        candidates.append(dict(box=[x,y,x+W,y+H],score=float(score),donor=donor,scale=scale,flipped=flip,mask_fraction=float(m.mean())))
                    cv2.rectangle(response,(max(0,x-W//2),max(0,y-H//2)),(min(response.shape[1]-1,x+W//2),min(response.shape[0]-1,y+H//2)),-1.,-1)
    candidates.sort(key=lambda z:z['score'],reverse=True)
    dedup=[]
    for c in candidates:
        if max((overlap(c['box'],q['box']) for q in dedup),default=0)<.5:dedup.append(c)
        if len(dedup)>=100:break
    return dict(id=id,frame_index=0,candidates=dedup,method='RGB masked template correlation; masks from TRAIN first-frame RGB-D foreground',no_query_future=True,no_names=True,no_simulator_truth=True)
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--tag',required=True);a=p.parse_args()
    rows=json.loads(Path(a.manifest).read_text())['rows']
    E.update({e['id']:e for e in rows})
    donors=train_templates()
    audit=[]
    for e in rows:
        id=e['id'];target=OUT/a.tag/(id+'.json')
        if target.exists():
            result=json.loads(target.read_text());assert result['no_query_future']
        else:
            result=propose(id,donors)
            assert result['candidates']
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(result,indent=2))
        if a.tag=='dev':
            ref=json.loads((R/'data/v3/proposals/motion_rgbd_rerank'/e['asset_group']/(e['source_episode']+'.json')).read_text())['selected']['box']
            vals=[overlap(c['box'],ref) for c in result['candidates']]
            row=dict(id=id,candidates=len(vals),top1_iou=vals[0],top5_iou=max(vals[:5]),top20_iou=max(vals[:20]),best_iou=max(vals),best_rank=int(np.argmax(vals))+1,top_score=result['candidates'][0]['score'])
            audit.append(row);print(json.dumps(row),flush=True)
        else:print(json.dumps(dict(id=id,candidates=len(result['candidates']),top_score=result['candidates'][0]['score'])),flush=True)
    if a.tag=='dev':
        summary={key:sum(x[key]>=.5 for x in audit) for key in ('top1_iou','top5_iou','top20_iou','best_iou')}
        target=R/'audits/v3/stage2_masked_template_dev_8.json'
        target.write_text(json.dumps(dict(status='development_only',reference='full-sequence automatic visual ROI, diagnostic only',summary=summary,rows=audit),indent=2))
        print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
