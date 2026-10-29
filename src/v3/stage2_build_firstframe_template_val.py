"""Build VAL inference inputs from first RGB-D frame and first-frame RGB template box only."""
from pathlib import Path
import json,math,sys
import h5py,numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_point_graphs import fps,neighbors,interpolate
from stage2_extract_dino import model,preproc
from stage2_local_teacher import crop_box
R=Path('/mnt/nvme1/drh/D_VLA')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
E={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
OUT=R/'data/v3/samples/firstframe_template_val'
def build(id,net,proposal_path=None):
 e=E[id]
 proposal_path=Path(proposal_path) if proposal_path is not None else R/'data/v3/proposals/first_frame_templates/val'/(id+'.json')
 proposal=json.loads(proposal_path.read_text())
 assert proposal['no_query_future']
 x1,y1,x2,y2=proposal['box'] if 'box' in proposal else proposal['candidates'][0]['box']
 with h5py.File(e['rgbd_path']) as f:
  rgb=f['agentview/rgb'][0];depth=f['agentview/depth_m'][0];K=f['agentview/K'][0];T=f['agentview/camera_to_world'][0]
 cb=crop_box([x1,y1,x2,y2],*depth.shape)
 cx1,cy1,cx2,cy2=int(math.floor(cb[0])),int(math.floor(cb[1])),int(math.ceil(cb[2])),int(math.ceil(cb[3]))
 with torch.inference_mode():
  patch=net.forward_features(preproc(rgb[None,cy1:cy2,cx1:cx2]))[:,1:,:].reshape(32,32,384).cpu().numpy()
 ring=depth[max(0,y1-8):min(depth.shape[0],y2+8),max(0,x1-8):min(depth.shape[1],x2+8)].copy()
 ry=y1-max(0,y1-8);rx=x1-max(0,x1-8);ring[ry:ry+y2-y1,rx:rx+x2-x1]=np.nan
 background=float(np.nanmedian(ring))
 d=depth[y1:y2,x1:x2];yy,xx=np.nonzero((d>.05)&(d<background-.015))
 if len(xx)<64:raise ValueError(f'few foreground points: {id}: {len(xx)}')
 px=x1+xx;py=y1+yy;z=d[yy,xx]
 cam=np.stack(((px-K[0,2])*z/K[0,0],(py-K[1,2])*z/K[1,1],z),1)
 xyz=cam@T[:3,:3].T+T[:3,3]
 gx=np.clip(((px-cx1)*32/max(1,cx2-cx1)).astype(int),0,31)
 gy=np.clip(((py-cy1)*32/max(1,cy2-cy1)).astype(int),0,31)
 dino=patch[gy,gx]
 sample=OUT/'val'/(id+'.npz');graph=OUT/'graphs/val'/(id+'.npz')
 sample.parent.mkdir(parents=True,exist_ok=True);graph.parent.mkdir(parents=True,exist_ok=True)
 if sample.exists() or graph.exists():raise FileExistsError(id)
 np.savez_compressed(sample,xyz=xyz.astype(np.float32),dino_feat=dino.astype(np.float32),pixel=np.stack((px,py),1).astype(np.int16),source_box=np.asarray([x1,y1,x2,y2]),first_frame_only=np.asarray(True),proposal_kind=np.asarray('first_frame_template'))
 idx1=fps(xyz,512);xyz1=xyz[idx1];idx2=fps(xyz1,128);xyz2=xyz1[idx2];idx3=fps(xyz2,32);xyz3=xyz2[idx3]
 nb1=neighbors(xyz,xyz1,.04);nb2=neighbors(xyz1,xyz2,.08);nb3=neighbors(xyz2,xyz3,.16)
 fp3_idx,fp3_weight=interpolate(xyz3,xyz2);fp2_idx,fp2_weight=interpolate(xyz2,xyz1);fp1_idx,fp1_weight=interpolate(xyz1,xyz)
 np.savez_compressed(graph,idx1=idx1,idx2=idx2,idx3=idx3,nb1=nb1,nb2=nb2,nb3=nb3,fp3_idx=fp3_idx,fp3_weight=fp3_weight,fp2_idx=fp2_idx,fp2_weight=fp2_weight,fp1_idx=fp1_idx,fp1_weight=fp1_weight)
 return dict(id=id,points=len(xyz),box=[x1,y1,x2,y2],crop=[cx1,cy1,cx2,cy2])
def main():
 torch.set_num_threads(4);net=model()
 rows=[]
 for id in S['val']:
  row=build(id,net);rows.append(row);print(json.dumps(row),flush=True)
 (OUT/'manifest.json').write_text(json.dumps(dict(query_only_first_frame=True,proposal='first-frame RGB template from TRAIN demonstrations',dino='fresh frozen DINO from query first-frame template crop',no_query_future=True,rows=rows),indent=2))
 print(json.dumps(dict(complete=True,count=len(rows))),flush=True)
if __name__=='__main__':main()
