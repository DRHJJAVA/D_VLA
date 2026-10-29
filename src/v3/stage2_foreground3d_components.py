"""Remove disconnected distractors from first-frame visual foreground in 3D.
Select the 6 mm spatial component containing the RGB-D automatic seed.
No simulator contacts, object truth, names, actions, or manual masks."""
from pathlib import Path
import json
import h5py,numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
R=Path('/mnt/nvme1/drh/D_VLA')
SOURCE=R/'data/v3/proposals/grabcut_components'
OUT=R/'data/v3/proposals/foreground3d_components'
DEV=json.loads((R/'data/v3/manifests/stage2_localization_dev_8.json').read_text())['rows']
TV=json.loads((SOURCE/'train_val_448/manifest.json').read_text())['rows']
def getcloud(path,pix):
 with h5py.File(path) as f:
  dep=f['agentview/depth_m'][0];K=f['agentview/K'][0];T=f['agentview/camera_to_world'][0]
 x=pix[:,0];y=pix[:,1];z=dep[y,x]
 valid=(z>.05)&np.isfinite(z)
 if not valid.all():raise ValueError('invalid depth in component '+str(path))
 cam=np.stack(((x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z),1)
 xyz=cam@T[:3,:3].T+T[:3,3]
 return xyz,z
def reduce(entry,split):
 id=entry['id']
 source=(SOURCE/'dev'/(id+'.npz')) if split=='dev' else (SOURCE/'train_val_448'/split/(id+'.npz'))
 path=Path(entry['rgbd_path'])
 with np.load(source) as f:
  pixel=f['pixel'].copy();base=f['base_box'].copy();outer=f['outer_box'].copy();seedz=float(f['seed_depth'])
 xyz,z=getcloud(path,pixel)
 pairs=cKDTree(xyz).query_pairs(.006,output_type='ndarray')
 if not len(pairs):raise ValueError('no 3D graph edges: '+id)
 n=len(pixel)
 graph=coo_matrix((np.ones(2*len(pairs),np.uint8),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(n,n)).tocsr()
 count,labels=connected_components(graph)
 x=pixel[:,0];y=pixel[:,1]
 seed=(x>=base[0])&(x<base[2])&(y>=base[1])&(y<base[3])&(abs(z-seedz)<.015)
 overlaps=np.bincount(labels[seed],minlength=count)
 which=int(np.argmax(overlaps))
 selected=labels==which
 if selected.sum()<64 or overlaps[which]<20:raise ValueError('selected 3D surface too small: '+id)
 clean=pixel[selected]
 dest=OUT/split/(id+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  with np.load(dest) as old:assert np.array_equal(old['pixel'],clean)
 else:np.savez_compressed(dest,pixel=clean,component_box=np.asarray([clean[:,0].min(),clean[:,1].min(),clean[:,0].max()+1,clean[:,1].max()+1]),base_box=base,outer_box=outer,seed_depth=np.asarray(seedz),query_frame=np.asarray(0),no_simulator_truth=np.asarray(True),rgbd_path=np.asarray(str(path)),connectivity_radius_m=np.asarray(.006))
 return dict(id=id,split=split,before=n,after=len(clean),removed=n-len(clean),components=count,seed_points=int(seed.sum()),seed_in_selected=int(overlaps[which]),rgbd_path=str(path))
def main():
 assert json.loads((SOURCE/'train_val_448/manifest.json').read_text())['status']=='uniform_448_train_val_grabcut_complete'
 rows=[]
 for e in TV:
  row=reduce(e,e['split']);rows.append(row);print(json.dumps(row),flush=True)
 for e in DEV:
  row=reduce(e,'dev');rows.append(row);print(json.dumps(row),flush=True)
 manifest=OUT/'manifest.json'
 data=dict(status='foreground3d_visual_seeded_complete',radius_m=.006,no_simulator_truth=True,rows=rows)
 if manifest.exists():assert json.loads(manifest.read_text())['rows']==rows
 else:manifest.write_text(json.dumps(data,indent=2))
 print(json.dumps(dict(status='complete',train=20,val=4,dev=8)),flush=True)
if __name__=='__main__':main()
