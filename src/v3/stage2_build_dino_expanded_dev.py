"""Build first-frame RGB-D inference inputs from fixed DINO-expanded development boxes."""
from pathlib import Path
import hashlib,json,sys
import torch
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
import stage2_build_firstframe_template_val as builder
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
BOX=R/'data/v3/proposals/firstframe_dino_expanded/dev'
OUT=R/'data/v3/samples/firstframe_dino_expanded_dev'
def main():
    entries=json.loads(M.read_text())['rows']
    builder.E.update({e['id']:e for e in entries})
    builder.OUT=OUT
    torch.set_num_threads(4)
    net=None;rows=[]
    for e in entries:
        id=e['id']
        path=OUT/'val'/(id+'.npz');graph=OUT/'graphs/val'/(id+'.npz')
        if path.exists() and graph.exists():
            rows.append(dict(id=id,status='reused'));continue
        if path.exists() or graph.exists():raise RuntimeError('partial sample: '+id)
        if net is None:net=builder.model()
        row=builder.build(id,net,BOX/(id+'.json'))
        rows.append(row);print(json.dumps(row),flush=True)
    manifest=OUT/'manifest.json'
    config=dict(status='first_frame_dev_inputs_complete',development_only=True,query_frame=0,boxes_fixed_before_contact_scoring=True,box_factor=1.5,source_manifest_sha256=hashlib.sha256(M.read_bytes()).hexdigest(),rows=rows)
    if manifest.exists():
        old=json.loads(manifest.read_text());assert old['source_manifest_sha256']==config['source_manifest_sha256']
    else:manifest.write_text(json.dumps(config,indent=2))
    print(json.dumps(dict(complete=True,episodes=len(rows))),flush=True)
if __name__=='__main__':main()
