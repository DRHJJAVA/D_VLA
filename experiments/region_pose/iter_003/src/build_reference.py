"""Remote interactive 3-D viewer of measured RGB-D contact estimates; no new training."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,numpy as np,h5py,hashlib
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from stage2_probe_motion_fingerpads import cloud,corners
from stage2_robot_only_depth import RobotDepth
R=Path('/mnt/nvme1/drh/D_VLA');D=Path(__file__).resolve().parents[1]/'evidence/motion'
OUT=Path(__file__).resolve().parents[1]/'evidence/reference_cloud'
PAGE=Path(__file__).resolve().parents[1]/'evidence/reference_internal.html'
def ply(path,p,color,flag,side):
 with path.open('x') as f:
  f.write('ply\nformat ascii 1.0\ncomment measured world coordinates in meters; flags are contact ESTIMATES, not contact truth\nelement vertex '+str(len(p))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar contact_estimate\nproperty char finger_side\nend_header\n')
  for xyz,rgb,fl,si in zip(p,color,flag,side):f.write(' '.join([*(f'{v:.7f}' for v in xyz),*(str(int(v)) for v in rgb),str(int(fl)),str(int(si))])+'\n')
def main():
 OUT.mkdir(parents=True,exist_ok=False);assert not PAGE.exists()
 refs=json.loads((Path(__file__).resolve().parents[1]/'evidence/base/protocol.json').read_text())['episodes'];by={x['id']:x for x in refs}
 frames=json.loads((D/'review_frames.json').read_text());datasets=[];summary=[]
 for item in frames:
  ident=item['id'];t=item['frame'];row=by[ident];name=ident+'_attempt'+str(item['attempt'])
  with np.load(D/(ident+'.npz')) as z:
   hand=z['hand_to_world'][t];q=z['measured_finger_q'][t];a,b=z['frame_offsets'][t:t+2]
   cp=z['points_world'][a:b];sides=z['point_side'][a:b];cams=z['point_camera'][a:b];pixels=z['point_pixel'][a:b]
  if not item['accepted_by_motion_visual_gate']:cp=cp[:0];sides=sides[:0];cams=cams[:0];pixels=pixels[:0]
  context=[];colors=[]
  with h5py.File(row['rgbd'],'r') as f:
   robot=RobotDepth(f['agentview/rgb'].shape[1])
   contactrgb=np.array([f[['agentview','robot0_eye_in_hand'][int(cam)]]['rgb'][t,int(v),int(u)] for cam,(u,v) in zip(cams,pixels)],np.uint8).reshape(-1,3)
   for cam in ['agentview','robot0_eye_in_hand']:
    g=f[cam];dep=g['depth_m'][t];K=g['K'][t];T=g['camera_to_world'][t]
    rd=robot.render(q,np.linalg.inv(hand)@T,K)
    p=cloud(dep,K,T);ph=(p-hand[:3,3])@hand[:3,:3]
    roi=(dep<rd-.002).reshape(-1)&(abs(ph[:,0])<.10)&(ph[:,1]>q[1]-.09)&(ph[:,1]<q[0]+.09)&(ph[:,2]>.065)&(ph[:,2]<.25)
    context.append(p[roi]);colors.append(g['rgb'][t].reshape(-1,3)[roi])
   robot.close()
  p=np.concatenate(context);rgb=np.concatenate(colors)
  if len(p):
   _,ix=np.unique(np.rint(p/.002).astype(int),axis=0,return_index=True);p=p[ix];rgb=rgb[ix]
  before=len(p);foreground='local RGB-D context; no accepted-contact seed'
  if len(cp) and len(p):
   tree=cKDTree(p);pairs=tree.query_pairs(.006,output_type='ndarray')
   if len(pairs):
    rr=np.r_[pairs[:,0],pairs[:,1]];cc=np.r_[pairs[:,1],pairs[:,0]]
    graph=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(p),len(p)))
    _,labels=connected_components(graph,directed=False)
   else:labels=np.arange(len(p))
   dist,near=tree.query(cp);wanted=np.unique(labels[near[dist<.005]])
   keep=np.isin(labels,wanted)
   if keep.any():p=p[keep];rgb=rgb[keep];foreground='automatic connected visible surface around accepted-contact seeds; not simulator object mask'
  # Keep every contact point at its original measured XYZ, even if context was voxel reduced.
  points=np.r_[p,cp];rawrgb=np.r_[rgb,contactrgb];flag=np.r_[np.zeros(len(p),np.uint8),np.ones(len(cp),np.uint8)]
  side=np.r_[np.full(len(p),-1,np.int8),sides.astype(np.int8)]
  np.savez_compressed(OUT/(name+'.npz'),xyz_world_m=points,rgb=rawrgb,contact_estimate=flag,finger_side=side,contact_source_camera=cams,contact_source_pixel=pixels,frame=t,hand_to_world=hand)
  displayrgb=rawrgb.copy();displayrgb[flag>0]=[30,245,70]
  ply(OUT/(name+'.ply'),points,displayrgb,flag,side)
  # Hand-relative coordinate display is a rigid transform, not a new reconstructed surface.
  local=(points-hand[:3,3])@hand[:3,:3];center=local.mean(0) if len(local) else np.array([0,0,.1])
  vis=(local-center)*1000
  contacts=[]
  for j in range(len(cp)):
   contacts.append({'world_m':cp[j].round(7).tolist(),'side':int(sides[j])+1,'camera':['agentview','robot0_eye_in_hand'][int(cams[j])],'pixel':pixels[j].tolist(),'frame':t})
  pads=[]
  for sign,k in [(1,0),(-1,1)]:
   plane=q[k]-sign*.0005
   pts=np.array([[-.008,plane,.0854],[.008,plane,.0854],[.008,plane,.1014],[-.008,plane,.1014]])
   pads.append(((pts-center)*1000).round(3).tolist())
  datasets.append({'name':name,'frame':t,'point':vis.round(3).tolist(),'rgb':rawrgb.tolist(),'flags':flag.tolist(),'side':side.tolist(),'contact_start':len(p),'contacts':contacts,'pads':pads,'held':item['accepted_by_motion_visual_gate'],'both':item['both_sides_visible'],'counts':item['visible_points_per_side']})
  summary.append({'id':ident,'attempt':item['attempt'],'frame':t,'context_points_before_component':before,'context_points':len(p),'contact_points':len(cp),'sides':item['visible_points_per_side'],'foreground':foreground,'npz':str(OUT/(name+'.npz')),'ply':str(OUT/(name+'.ply'))})
  print(name,len(p),len(cp),flush=True)
 PAGE.write_text('Internal reference clouds; not a model prediction.',encoding='utf-8')
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'locked.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(OUT.iterdir())+[PAGE,Path(__file__)] if p.is_file()}},indent=2))
 print('complete',flush=True)
if __name__=='__main__':main()
