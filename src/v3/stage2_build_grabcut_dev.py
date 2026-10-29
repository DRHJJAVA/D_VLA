"""Build query-first-frame RGB-D object clouds from fixed GrabCut components.
No query future, simulator truth, actions, names or manual region labels."""
from pathlib import Path
import hashlib,json,math,sys
import h5py,numpy as np,torch
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src'))
sys.path.insert(0,str(R/'src/v3'))
from build_point_graphs import fps,neighbors,interpolate
from stage2_extract_dino import model,preproc
from stage2_local_teacher import crop_box
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
COMP=R/'data/v3/proposals/grabcut_components/dev'
OUT=R/'data/v3/samples/grabcut_firstframe_dev'
LOCK=R/'audits/v3/stage2_grabcut_dev_components_locked.json'
def build(e,net):
    id=e['id']
    with np.load(COMP/(id+'.npz')) as f:
        pixel=f['pixel'].copy();source_box=f['component_box'].copy();outer=f['outer_box'].copy()
        assert int(f['query_frame'])==0 and bool(f['no_simulator_truth'])
    with h5py.File(e['rgbd_path']) as f:
        rgb=f['agentview/rgb'][0];depth=f['agentview/depth_m'][0];K=f['agentview/K'][0];T=f['agentview/camera_to_world'][0]
    valid=(depth[pixel[:,1],pixel[:,0]]>.05)&np.isfinite(depth[pixel[:,1],pixel[:,0]])
    pixel=pixel[valid]
    if len(pixel)<64:raise ValueError('too few visual object pixels: '+id)
    px=pixel[:,0];py=pixel[:,1];z=depth[py,px]
    cam=np.stack(((px-K[0,2])*z/K[0,0],(py-K[1,2])*z/K[1,1],z),1)
    xyz=cam@T[:3,:3].T+T[:3,3]
    cb=crop_box(outer.tolist(),*depth.shape)
    cx1,cy1,cx2,cy2=int(math.floor(cb[0])),int(math.floor(cb[1])),int(math.ceil(cb[2])),int(math.ceil(cb[3]))
    with torch.inference_mode():
        patch=net.forward_features(preproc(rgb[None,cy1:cy2,cx1:cx2]))[:,1:,:].reshape(32,32,384).cpu().numpy()
    gx=np.clip(((px-cx1)*32/max(1,cx2-cx1)).astype(int),0,31)
    gy=np.clip(((py-cy1)*32/max(1,cy2-cy1)).astype(int),0,31)
    dino=patch[gy,gx]
    sample=OUT/'val'/(id+'.npz');graph=OUT/'graphs/val'/(id+'.npz')
    sample.parent.mkdir(parents=True,exist_ok=True);graph.parent.mkdir(parents=True,exist_ok=True)
    if sample.exists() or graph.exists():raise FileExistsError(id)
    np.savez_compressed(sample,xyz=xyz.astype(np.float32),dino_feat=dino.astype(np.float32),pixel=pixel.astype(np.int16),source_box=source_box,first_frame_only=np.asarray(True),proposal_kind=np.asarray('masked_DINO_expanded_GrabCut'))
    idx1=fps(xyz,512);xyz1=xyz[idx1];idx2=fps(xyz1,128);xyz2=xyz1[idx2];idx3=fps(xyz2,32);xyz3=xyz2[idx3]
    nb1=neighbors(xyz,xyz1,.04);nb2=neighbors(xyz1,xyz2,.08);nb3=neighbors(xyz2,xyz3,.16)
    fp3_idx,fp3_weight=interpolate(xyz3,xyz2);fp2_idx,fp2_weight=interpolate(xyz2,xyz1);fp1_idx,fp1_weight=interpolate(xyz1,xyz)
    np.savez_compressed(graph,idx1=idx1,idx2=idx2,idx3=idx3,nb1=nb1,nb2=nb2,nb3=nb3,fp3_idx=fp3_idx,fp3_weight=fp3_weight,fp2_idx=fp2_idx,fp2_weight=fp2_weight,fp1_idx=fp1_idx,fp1_weight=fp1_weight)
    return dict(id=id,points=len(xyz),component_box=source_box.tolist(),outer_box=outer.tolist())
def main():
    assert json.loads(LOCK.read_text())['status']=='firstframe_grabcut_components_frozen_before_contact_scoring'
    torch.set_num_threads(4)
    entries=json.loads(M.read_text())['rows']
    net=None;rows=[]
    for e in entries:
        id=e['id'];sample=OUT/'val'/(id+'.npz');graph=OUT/'graphs/val'/(id+'.npz')
        if sample.exists() and graph.exists():
            rows.append(dict(id=id,status='reused'));continue
        if sample.exists() or graph.exists():raise RuntimeError('partial sample '+id)
        if net is None:net=model()
        row=build(e,net);rows.append(row);print(json.dumps(row),flush=True)
    manifest=OUT/'manifest.json'
    config=dict(status='firstframe_grabcut_dev_inputs_complete',query_frame=0,no_simulator_truth=True,source_manifest_sha256=hashlib.sha256(M.read_bytes()).hexdigest(),component_lock_sha256=hashlib.sha256(LOCK.read_bytes()).hexdigest(),rows=rows)
    if manifest.exists():assert json.loads(manifest.read_text())['source_manifest_sha256']==config['source_manifest_sha256']
    else:manifest.write_text(json.dumps(config,indent=2))
    print(json.dumps(dict(complete=True,episodes=len(rows))),flush=True)
if __name__=='__main__':main()
