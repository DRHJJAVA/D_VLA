"""Expand DINO-selected first-frame candidates by fixed 1.5x for full-object coverage.
Selection uses query first-frame RGB only; expansion constant chosen from visual crop audit, not simulator contacts."""
from pathlib import Path
import json,math
R=Path('/mnt/nvme1/drh/D_VLA')
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
A=R/'audits/v3/stage2_dino_masked_template_dev_8.json'
OUT=R/'data/v3/proposals/firstframe_dino_expanded/dev'
FACTOR=1.5
def expand(box):
    x1,y1,x2,y2=map(float,box)
    cx=(x1+x2)/2;cy=(y1+y2)/2
    w=(x2-x1)*FACTOR;h=(y2-y1)*FACTOR
    return [max(0,math.floor(cx-w/2)),max(0,math.floor(cy-h/2)),min(448,math.ceil(cx+w/2)),min(448,math.ceil(cy+h/2))]
def main():
    entries=json.loads(M.read_text())['rows']
    audit={x['id']:x for x in json.loads(A.read_text())['cases']}
    for e in entries:
        id=e['id'];features=audit[id]['rows']
        rank=max(range(len(features)),key=lambda i:features[i]['dino_cls_max'])
        selected=features[rank]
        result=dict(id=id,query_frame=0,method='masked RGB template candidates reranked by frozen DINO CLS similarity to TRAIN crops; fixed 1.5x box expansion',base_box=selected['box'],box=expand(selected['box']),candidate_rank=rank+1,dino_similarity=selected['dino_cls_max'],no_query_future=True,no_names=True,no_simulator_truth=True)
        dest=OUT/(id+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert json.loads(dest.read_text())==result
        else:dest.write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
if __name__=='__main__':main()
