"""Build test inference samples from first RGB-D frame only; no future/test cue access."""
from pathlib import Path
import json,sys
import h5py,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_point_graphs import fps,neighbors,interpolate
R=Path('/mnt/nvme1/drh/D_VLA')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
E={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
OUT=R/'data/v3/samples/visual_interaction_v1'
def build(e):
 aid=e['asset_group'];ep=e['source_episode'];id=e['id']
 x1,y1,x2,y2=json.loads((R/'data/v3/proposals/motion_rgbd_rerank'/aid/(ep+'.json')).read_text())['selected']['box']
 with h5py.File(e['rgbd_path']) as f:
  first=f['agentview/depth_m'][0];rgb=f['agentview/rgb'][0];K=f['agentview/K'][0];T=f['agentview/camera_to_world'][0]
 with h5py.File(R/'data/v3/features/dinov2_object_crops'/aid/(ep+'.h5')) as f:
  patch=f['patch'][0].astype(np.float32);cx1,cy1,cx2,cy2=json.loads(f.attrs['crop_box'])
 ring=first[max(0,y1-8):min(first.shape[0],y2+8),max(0,x1-8):min(first.shape[1],x2+8)].copy()
 ry=y1-max(0,y1-8);rx=x1-max(0,x1-8);ring[ry:ry+y2-y1,rx:rx+x2-x1]=np.nan
 background=float(np.nanmedian(ring))
 d=first[y1:y2,x1:x2];yy,xx=np.nonzero((d>.05)&(d<background-.015))
 assert len(xx)>=64
 px=x1+xx;py=y1+yy;z=d[yy,xx]
 cam=np.stack(((px-K[0,2])*z/K[0,0],(py-K[1,2])*z/K[1,1],z),1)
 xyz=cam@T[:3,:3].T+T[:3,3]
 gx=np.clip(((px-cx1)*32/max(1,cx2-cx1)).astype(int),0,31)
 gy=np.clip(((py-cy1)*32/max(1,cy2-cy1)).astype(int),0,31)
 dino=patch[gy,gx]
 sample=OUT/'test_inference'/(id+'.npz')
 graph=OUT/'graphs/test_inference'/(id+'.npz')
 sample.parent.mkdir(parents=True,exist_ok=True);graph.parent.mkdir(parents=True,exist_ok=True)
 if sample.exists() or graph.exists():raise FileExistsError(id)
 np.savez_compressed(sample,xyz=xyz.astype(np.float32),dino_feat=dino.astype(np.float32),rgb=rgb[py,px],pixel=np.stack((px,py),1).astype(np.int16),source_box=np.asarray([x1,y1,x2,y2]),first_frame_only=np.asarray(True))
 idx1=fps(xyz,512);xyz1=xyz[idx1];idx2=fps(xyz1,128);xyz2=xyz1[idx2];idx3=fps(xyz2,32);xyz3=xyz2[idx3]
 nb1=neighbors(xyz,xyz1,.04);nb2=neighbors(xyz1,xyz2,.08);nb3=neighbors(xyz2,xyz3,.16)
 fp3_idx,fp3_weight=interpolate(xyz3,xyz2);fp2_idx,fp2_weight=interpolate(xyz2,xyz1);fp1_idx,fp1_weight=interpolate(xyz1,xyz)
 np.savez_compressed(graph,idx1=idx1,idx2=idx2,idx3=idx3,nb1=nb1,nb2=nb2,nb3=nb3,fp3_idx=fp3_idx,fp3_weight=fp3_weight,fp2_idx=fp2_idx,fp2_weight=fp2_weight,fp1_idx=fp1_idx,fp1_weight=fp1_weight)
 return dict(id=id,points=len(xyz))
def main():
 rows=[]
 for id in S['test']:
  sample=OUT/'test_inference'/(id+'.npz');graph=OUT/'graphs/test_inference'/(id+'.npz')
  if sample.exists() and graph.exists():rows.append(dict(id=id,reused=True));continue
  r=build(E[id]);rows.append(r);print(json.dumps(r),flush=True)
 (OUT/'test_inference_manifest.json').write_text(json.dumps(dict(only_first_rgbd_frame=True,no_test_future_frames=True,rows=rows),indent=2))
 print(json.dumps(dict(complete=True,count=len(rows))),flush=True)
if __name__=='__main__':main()
