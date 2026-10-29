"""Extract up to three independent first-frame RGB-D surface candidates for pose validation.
The top candidates were ranked using only frame-zero RGB and TRAIN exemplars.
No query future, simulator contact/object truth, names, text or actions are read.
"""
from pathlib import Path
import json,math,sys,hashlib
import h5py,cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_grabcut_component_dev import segment
from stage2_foreground3d_components import getcloud
MAN=R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json'
CAND=R/'data/v3/proposals/firstframe_dino_candidates/poseval16'
OUT=R/'data/v3/proposals/poseval16_surface_candidates'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def expand(box,w,h,factor=1.5):
 x1,y1,x2,y2=map(float,box);cx=(x1+x2)/2;cy=(y1+y2)/2
 bw=(x2-x1)*factor;bh=(y2-y1)*factor
 return [max(0,math.floor(cx-bw/2)),max(0,math.floor(cy-bh/2)),
         min(w,math.ceil(cx+bw/2)),min(h,math.ceil(cy+bh/2))]
def reduce3d(path,pixel,base,seedz):
 xyz,z=getcloud(path,pixel)
 pairs=cKDTree(xyz).query_pairs(.006,output_type='ndarray')
 if len(pairs)==0:raise ValueError('no 3D edges')
 n=len(pixel)
 g=coo_matrix((np.ones(2*len(pairs),np.uint8),
    (np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(n,n)).tocsr()
 count,labels=connected_components(g)
 x=pixel[:,0];y=pixel[:,1]
 seed=(x>=base[0])&(x<base[2])&(y>=base[1])&(y<base[3])&(abs(z-seedz)<.015)
 overlaps=np.bincount(labels[seed],minlength=count)
 which=int(np.argmax(overlaps));keep=labels==which
 if keep.sum()<64 or overlaps[which]<20:raise ValueError('3D component too small')
 clean=pixel[keep]
 return clean,xyz[keep],z[keep],count,int(overlaps[which])
def main():
 entries=json.loads(MAN.read_text())['rows'];assert len(entries)==16
 rows=[]
 for e in entries:
  id=e['id'];path=Path(e['rgbd_path'])
  source=CAND/(id+'.json');record=json.loads(source.read_text())
  assert record['query_frame']==0 and record['no_query_future']
  with h5py.File(path) as f:
   rgb=f['agentview/rgb'][0];depth=f['agentview/depth_m'][0]
  assert rgb.shape[:2]==depth.shape==(448,448)
  outcomes=[]
  for rank,idx in enumerate(record['ranked_candidate_indices'][:3],1):
   c=record['candidates'][idx]
   base=c['box'];outer=expand(base,448,448)
   dest=OUT/id/(f'rank_{rank}.npz')
   if dest.exists():
    with np.load(dest) as old:
     assert str(old['candidate_sha256'])==sha(source)
     assert int(old['query_frame'])==0
     n=len(old['pixel'])
    outcomes.append(dict(rank=rank,candidate_index=idx,status='reused',points=n,box=base))
    continue
   try:
    pix,bbox,seedz=segment(rgb,depth,base,outer)
    clean,xyz,z,components,seedcount=reduce3d(path,pix,base,seedz)
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.partial.npz')
    if tmp.exists():raise RuntimeError('inspect partial: '+str(tmp))
    np.savez_compressed(tmp,pixel=clean,xyz=xyz.astype(np.float32),
       depth=z.astype(np.float32),base_box=np.asarray(base),
       outer_box=np.asarray(outer),component_box=np.asarray([
          clean[:,0].min(),clean[:,1].min(),clean[:,0].max()+1,clean[:,1].max()+1]),
       candidate_sha256=np.asarray(sha(source)),candidate_index=np.asarray(idx),
       rank=np.asarray(rank),query_frame=np.asarray(0),no_simulator_truth=np.asarray(True))
    tmp.replace(dest)
    outcomes.append(dict(rank=rank,candidate_index=idx,status='ok',
       points=len(clean),box=base,components=components,seed_points=seedcount))
   except (ValueError,cv2.error) as ex:
    outcomes.append(dict(rank=rank,candidate_index=idx,status='segmentation_failed',
       box=base,error=str(ex)))
  row=dict(id=id,query_frame=0,candidates=outcomes)
  rows.append(row);print(json.dumps(row),flush=True)
 manifest=OUT/'manifest.json'
 result=dict(status='poseval16_top3_firstframe_surfaces_complete',
    no_query_future=True,no_simulator_truth=True,no_names=True,
    source_manifest_sha256=sha(MAN),candidate_manifest_sha256=sha(CAND/'manifest.json'),
    rows=rows)
 if manifest.exists():
  old=json.loads(manifest.read_text())
  assert old['source_manifest_sha256']==result['source_manifest_sha256']
 else:manifest.write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(status='complete',episodes=len(rows),valid=sum(sum(c['status'] in ('ok','reused') for c in row['candidates']) for row in rows))),flush=True)
if __name__=='__main__':main()
