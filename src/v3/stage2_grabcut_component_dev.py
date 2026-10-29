"""First-frame RGB-D seeded GrabCut segmentation inside frozen DINO ROI.
No object names, labels, simulator truth, actions or query future."""
from pathlib import Path
import json
import cv2,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
BOX=R/'data/v3/proposals/firstframe_dino_expanded/dev'
OUT=R/'data/v3/proposals/grabcut_components/dev'
def segment(rgb,depth,base,outer):
    x1,y1,x2,y2=outer;bx1,by1,bx2,by2=base
    # The frozen expanded box is the only region allowed to become foreground.
    mask=np.full(depth.shape,cv2.GC_BGD,dtype=np.uint8)
    mask[y1:y2,x1:x2]=cv2.GC_PR_BGD
    mask[by1:by2,bx1:bx2]=cv2.GC_PR_FGD
    d=depth[by1:by2,bx1:bx2]
    valid=d[(d>.05)&np.isfinite(d)]
    if len(valid)<30:raise ValueError('no valid seed')
    z=float(np.median(valid))
    sure=(np.abs(d-z)<.015)&(d>.05)
    inner=mask[by1:by2,bx1:bx2];inner[sure]=cv2.GC_FGD
    if np.sum(inner==cv2.GC_FGD)<20:raise ValueError('too few foreground seed pixels')
    bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64)
    cv2.grabCut(rgb,mask,None,bg,fg,5,cv2.GC_INIT_WITH_MASK)
    fgmask=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)).astype('uint8')
    fgmask[:y1]=0;fgmask[y2:]=0;fgmask[:,:x1]=0;fgmask[:,x2:]=0
    n,labels,stats,centers=cv2.connectedComponentsWithStats(fgmask,8)
    seed=np.zeros(depth.shape,'uint8');seed[by1:by2,bx1:bx2]=sure.astype('uint8')
    options=[]
    for i in range(1,n):
        overlap=int(((labels==i)&(seed>0)).sum())
        if overlap:options.append((overlap,i))
    if not options:raise ValueError('no component contains seed')
    _,which=max(options)
    selected=labels==which
    yy,xx=np.nonzero(selected)
    if len(xx)<64:raise ValueError('component too small')
    box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
    return np.stack((xx,yy),1).astype(np.int16),box,z
def main():
    entries=json.loads(M.read_text())['rows']
    for e in entries:
        id=e['id'];p=json.loads((BOX/(id+'.json')).read_text())
        assert p['no_query_future'] and p['query_frame']==0
        with h5py.File(e['rgbd_path']) as f:
            rgb=f['agentview/rgb'][0];depth=f['agentview/depth_m'][0]
        pixel,box,z=segment(rgb,depth,p['base_box'],p['box'])
        dest=OUT/(id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            with np.load(dest) as old:assert np.array_equal(old['pixel'],pixel)
        else:np.savez_compressed(dest,pixel=pixel,component_box=np.asarray(box),base_box=np.asarray(p['base_box']),outer_box=np.asarray(p['box']),seed_depth=np.asarray(z),query_frame=np.asarray(0),no_simulator_truth=np.asarray(True))
        print(json.dumps(dict(id=id,component_points=len(pixel),component_box=box,seed_depth=round(z,3))),flush=True)
if __name__=='__main__':main()
