"""First-frame seeded RGB-D component within fixed DINO-expanded ROI.
Uses depth, base candidate and spatial connectivity only; no simulator truth or query future."""
from pathlib import Path
import json
import cv2,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
BOX=R/'data/v3/proposals/firstframe_dino_expanded/dev'
OUT=R/'data/v3/proposals/seeded_depth_components/dev'
def component(depth,base,outer):
    x1,y1,x2,y2=outer
    bx1,by1,bx2,by2=base
    local=depth[y1:y2,x1:x2]
    sub=depth[by1:by2,bx1:bx2]
    valid=sub[(sub>.05)&np.isfinite(sub)]
    if len(valid)<30:raise ValueError('no valid seed depth')
    z=float(np.median(valid))
    # Camera-depth band chosen before reference-contact audit.
    band=((local>.05)&np.isfinite(local)&(np.abs(local-z)<.06)).astype('uint8')
    band=cv2.morphologyEx(band,cv2.MORPH_CLOSE,np.ones((3,3),'uint8'))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(band,8)
    seed=np.zeros(local.shape,'uint8')
    seed[max(0,by1-y1):min(local.shape[0],by2-y1),max(0,bx1-x1):min(local.shape[1],bx2-x1)]=1
    choices=[]
    for i in range(1,n):
        overlap=int(((labels==i)&(seed>0)).sum())
        if overlap:choices.append((overlap,i))
    if not choices:raise ValueError('no depth component overlaps seed')
    _,which=max(choices)
    mask=labels==which
    yy,xx=np.nonzero(mask)
    if len(xx)<64:raise ValueError('component too small')
    box=[int(x1+xx.min()),int(y1+yy.min()),int(x1+xx.max()+1),int(y1+yy.max()+1)]
    return np.stack((x1+xx,y1+yy),1).astype(np.int16),box,z
def main():
    entries=json.loads(M.read_text())['rows']
    for e in entries:
        id=e['id'];p=json.loads((BOX/(id+'.json')).read_text())
        assert p['no_query_future'] and p['query_frame']==0
        with h5py.File(e['rgbd_path']) as f:depth=f['agentview/depth_m'][0]
        pixel,box,z=component(depth,p['base_box'],p['box'])
        dest=OUT/(id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            with np.load(dest) as old:assert len(old['pixel'])==len(pixel) and np.array_equal(old['pixel'],pixel)
        else:np.savez_compressed(dest,pixel=pixel,component_box=np.asarray(box),base_box=np.asarray(p['base_box']),outer_box=np.asarray(p['box']),seed_depth=np.asarray(z),query_frame=np.asarray(0),no_simulator_truth=np.asarray(True))
        print(json.dumps(dict(id=id,component_points=len(pixel),component_box=box,seed_depth=round(z,3))),flush=True)
if __name__=='__main__':main()
