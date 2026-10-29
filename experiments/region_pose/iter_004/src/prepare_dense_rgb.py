from pathlib import Path
import os,sys,json,hashlib,shutil
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,h5py,torch
from torch.nn import functional as F
R=Path('/mnt/nvme1/drh/D_VLA');I=Path(__file__).resolve().parents[1];OLD=I.parent/'iter_003'
sys.path[:0]=[str(R/'src/v3'),str(R/'src')]
from stage2_extract_dino import model,preproc
from stage2_robot_only_depth import RobotDepth
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15);net=model()
 m=json.loads((OLD/'manifest.json').read_text());refs={r['id']:r for r in json.loads((OLD/'evidence/base/protocol.json').read_text())['episodes']}
 audits=[];source={}
 for row in m['rows']:
  key=row['key'];old=OLD/'samples'/(key+'.npz');dest=I/'samples'/old.name;gr=OLD/'graphs'/old.name
  source[str(old)]=hashlib.sha256(old.read_bytes()).hexdigest();source[str(gr)]=hashlib.sha256(gr.read_bytes()).hexdigest()
  with np.load(old) as z:a={k:z[k] for k in z.files}
  if dest.exists():
   with np.load(dest) as z:
    assert all(np.array_equal(a[k],z[k]) for k in a if k!='dino_feat')
    count=len(z['input_rgb_pixel']);box=z['input_crop_box'].tolist()
   assert (I/'graphs'/old.name).exists()
  else:
   ref=refs[row['episode']]
   with np.load(ref['component']) as z:pixel0=z['pixel'].astype(np.int64)
   with np.load(OLD/'evidence/motion'/(row['episode']+'.npz')) as z:H=z['hand_to_world'][row['query_frame']];q=z['measured_finger_q'][row['query_frame']]
   with h5py.File(ref['rgbd'],'r') as f:
    g=f['agentview'];t=row['query_frame'];depth=g['depth_m'][t];rgb=g['rgb'][t];K=g['K'][t];T=g['camera_to_world'][t];d0=g['depth_m'][0]
    robot=RobotDepth(depth.shape[0]);rd=robot.render(q,np.linalg.inv(H)@T,K);robot.close()
    pu,pv=pixel0[:,0],pixel0[:,1]
    valid=(abs(depth[pv,pu]-d0[pv,pu])<.005)&(depth[pv,pu]<rd[pv,pu]-.002)&np.isfinite(depth[pv,pu])&(depth[pv,pu]>0)
    full=pixel0[valid];count=len(full);sample=a['pixel'].astype(np.int64)
    mask=np.zeros(depth.shape,bool);mask[full[:,1],full[:,0]]=True
    assert mask[sample[:,1],sample[:,0]].all()
    assert count>=len(sample)
    im=np.full_like(rgb,127);im[mask]=rgb[mask]
    x1,y1=full.min(0);x2,y2=full.max(0)+1;box=[int(x1),int(y1),int(x2),int(y2)]
    with torch.inference_mode():patch=net.forward_features(preproc(im[None,y1:y2,x1:x2]))[:,1:,:].reshape(1,32,32,384).permute(0,3,1,2)
    gx=(sample[:,0]-x1+.5)/(x2-x1)*2-1;gy=(sample[:,1]-y1+.5)/(y2-y1)*2-1
    grid=torch.as_tensor(np.stack([gx,gy],1)[None,None],dtype=patch.dtype,device=patch.device)
    with torch.inference_mode():feat=F.grid_sample(patch,grid,align_corners=False).squeeze(0).squeeze(1).T.cpu().numpy()
    assert feat.shape==a['dino_feat'].shape and np.isfinite(feat).all()
    a['dino_feat']=feat.astype(np.float16);a['input_rgb_pixel']=full.astype(np.int16);a['input_crop_box']=np.array(box,np.int32)
   with dest.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**a)
   dest.with_suffix('.tmp').replace(dest)
   shutil.copy2(gr,I/'graphs'/old.name)
  with np.load(old) as before,np.load(dest) as after:
   assert all(np.array_equal(before[k],after[k]) for k in before.files if k!='dino_feat')
  assert hashlib.sha256((I/'graphs'/old.name).read_bytes()).hexdigest()==source[str(gr)]
  audits.append(dict(key=key,episode=row['episode'],split=row['split'],geometry_points=len(a['pixel']),full_input_rgb_pixels=count,sampled_rgb_retention=len(a['pixel'])/count,crop_box=box,all_nondino_arrays_identical=True,graph_identical=True))
  print(json.dumps(audits[-1]),flush=True)
 (I/'audits/dense_rgb_preparation.json').write_text(json.dumps(dict(rows=audits,source_sha256=source,only_dino_features_changed=True),indent=2))
 m.update(feature_iteration='004',source_iteration_003_manifest_sha256=hashlib.sha256((OLD/'manifest.json').read_bytes()).hexdigest(),input_rgb_uses_full_causal_foreground=True)
 (I/'manifest.json').write_text(json.dumps(m,indent=2))
 print('FEATURE_PREPARATION_COMPLETE',flush=True)
if __name__=='__main__':main()
