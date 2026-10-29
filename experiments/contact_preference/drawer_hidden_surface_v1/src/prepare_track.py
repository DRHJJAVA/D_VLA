import os,sys,json,ast,traceback,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ['MUJOCO_GL']='egl'
import numpy as np,h5py
from PIL import Image,ImageDraw
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/contact_preference/drawer_hidden_surface_v1';C=R/'experiments/contact_frames/cf017/cases/drawer_demo_0'
sys.path.insert(0,str(R));from src.v3.stage2_robot_only_depth import RobotDepth
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as z:
  for b in iter(lambda:z.read(4194304),b''):h.update(b)
 return h.hexdigest()
def back(dep,K,T,uv):
 u,v=uv.T;z=dep[v,u];P=np.c_[(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z];return P@T[:3,:3].T+T[:3,3]
def main():
 cfg=json.loads((D/'config.json').read_text());ref=np.load(C/'artifacts/reference.npz');tr=np.load(C/'artifacts/tracks.npz');mot=np.load(C/'motion.npz')
 tree=ast.parse((R/'experiments/contact_preference/stage012_v1/stage2_reuse_localize.py').read_text());loop=next(x for x in ast.walk(tree) if isinstance(x,ast.For) and x.lineno==83);nodes=[x for x in loop.body if isinstance(x,ast.Assign) and 86<=x.lineno<=89];ns={'np':np,'ref':ref,'mot':mot,'ref_frame':44,'radius':.03};exec(compile(ast.Module(body=nodes,type_ignores=[]),'old_source_selection','exec'),ns)
 source=ref['surface_xyz'][ns['sel']];suv=ref['surface_uv'][ns['sel']];assert len(source)==319
 centers=ns['centers'];X0=ref['track_xyz'];oldsmall=np.linalg.norm(X0[:,None]-centers[None],axis=2).min(1)<.03;oldwide=np.linalg.norm(X0[:,None]-centers[None],axis=2).min(1)<.08
 box=ns['box'];uv=ref['track_uv'];oldbox=(uv[:,0]>=box[0]-10)&(uv[:,0]<=box[2]+10)&(uv[:,1]>=box[1]-10)&(uv[:,1]<=box[3]+10)
 hpath=R/'experiments/contact_frames/cf017/rgbd/drawer_demo_0.h5';robot=RobotDepth(448);audit={'time':now(),'total_old_tracks':len(X0),'initial_small_tracks':int(oldsmall.sum()),'initial_wide_tracks':int(oldwide.sum()),'old_stage012_box_tracks':int(oldbox.sum()),'rows':[]}
 try:
  with h5py.File(hpath,'r') as h:
   for t in cfg['frames']:
    K=h['agentview/K'][t];T=h['agentview/camera_to_world'][t];dep=h['agentview/depth_m'][t];H=mot['hand_to_world'][t];rd=robot.render(mot['measured_finger_q'][t],np.linalg.inv(H)@T,K)
    U=np.rint(tr['tracks'][t]).astype(int);bounds=(U[:,0]>=0)&(U[:,0]<448)&(U[:,1]>=0)&(U[:,1]<448);vis=tr['visibility'][t]>.9;depthok=np.zeros(len(U),bool);notrobot=depthok.copy();cur=np.full_like(X0,np.nan,dtype=float);ii=np.flatnonzero(bounds);z=dep[U[ii,1],U[ii,0]];depthok[ii]=np.isfinite(z)&(z>0);notrobot[ii]=z<rd[U[ii,1],U[ii,0]]-.002;cur[ii]=back(dep,K,T,U[ii]);disp=np.linalg.norm(cur-X0,axis=1)*1000
    rec={'frame':t,'groups':{}}
    for n,mask in [('small_source_neighborhood',oldsmall),('motion_neighborhood_8cm',oldwide),('old_box_plus10px',oldbox)]:
     v=mask&bounds&vis;good=v&depthok&notrobot
     rec['groups'][n]={'initial':int(mask.sum()),'visibility_or_bounds_rejected':int((mask&~(bounds&vis)).sum()),'invalid_depth_after_visible':int((v&~depthok).sum()),'robot_depth_rejected_after_visible':int((v&depthok&~notrobot).sum()),'usable':int(good.sum()),'usable_train':int((good&~ref['heldout'].astype(bool)).sum()),'usable_heldout':int((good&ref['heldout'].astype(bool)).sum()),'observed_displacement_mm':np.round(disp[good],2).tolist(),'static_within5mm':int((good&(disp<=5)).sum()),'moving_over5mm':int((good&(disp>5)).sum())}
    audit['rows'].append(rec)
   X=ref['surface_xyz'];uv=ref['surface_uv'].astype(int);mask=np.linalg.norm(X[:,None]-centers[None],axis=2).min(1)<.08
   candidates=np.flatnonzero(mask);grid=uv[candidates]//3;_,idx=np.unique(grid,axis=0,return_index=True);sel=candidates[idx]
   if len(sel)>400:sel=sel[np.linspace(0,len(sel)-1,400).astype(int)]
   seeds=uv[sel];seedxyz=X[sel];hold=((seeds[:,0]*73856093)^(seeds[:,1]*19349663))%5==0
   # Preserve exact initial real-depth source and seeds.
   K=h['agentview/K'][0];T=h['agentview/camera_to_world'][0];dep=h['agentview/depth_m'][0];assert np.max(abs(back(dep,K,T,seeds)-seedxyz))<1e-7
   np.savez_compressed(D/'data/source_seed.npz',source_world=source,source_uv=suv,seeds_uv=seeds,seeds_world=seedxyz,heldout=hold,centers=centers,old_small_track=oldsmall,old_wide_track=oldwide)
   base=Image.fromarray(h['agentview/rgb'][0]);dr=ImageDraw.Draw(base)
   for x,y in suv:dr.ellipse((int(x)-1,int(y)-1,int(x)+1,int(y)+1),fill=(255,110,30))
   for (x,y),held in zip(seeds,hold):dr.ellipse((int(x)-2,int(y)-2,int(x)+2,int(y)+2),outline=(0,180,255) if held else (220,50,190),width=1)
   base.save(D/'images/source_seed.jpg')
   audit['new_seeds']={'count':len(sel),'heldout':int(hold.sum()),'real_source_points':319,'unexpanded_contact_source':True};save(D/'audits/old_track_attrition.json',audit)
   video=np.stack([h['agentview/rgb'][i] for i in range(len(h['frame_indices']))])
 finally:robot.close()
 weight=R/'experiments/region_pose/iter_010/weights/scaled_offline.pth'
 inp=[hpath,C/'artifacts/reference.npz',C/'artifacts/tracks.npz',C/'motion.npz',weight,D/'config.json',D/'data/source_seed.npz',Path(__file__)]
 save(D/'input_locked.json',{str(p):sha(p) for p in inp});save(D/'stage_state.json',{'time':now(),'stage':'tracking','seeds':len(seeds),'frames':len(video)})
 print(json.dumps({'stage':'tracking','seeds':len(seeds),'holdout':int(hold.sum())}),flush=True)
 import torch
 torch.manual_seed(2026);torch.cuda.set_per_process_memory_fraction(.4)
 sys.path.insert(0,str(R/'third_party/co_tracker_010'));from cotracker.predictor import CoTrackerPredictor
 model=CoTrackerPredictor(checkpoint=str(weight),offline=True,v2=False,window_len=60).cuda().eval()
 tensor=torch.from_numpy(video).permute(0,3,1,2)[None].float().cuda();query=torch.from_numpy(np.c_[np.zeros(len(seeds)),seeds].astype(np.float32))[None].cuda()
 with torch.inference_mode():tracks,visibility=model(tensor,queries=query,backward_tracking=False)
 tracks=tracks[0].cpu().numpy();visibility=visibility[0].cpu().numpy()
 assert tracks.shape==(len(video),len(seeds),2)
 np.savez_compressed(D/'data/tracks.npz',tracks=tracks,visibility=visibility)
 save(D/'stage_state.json',{'time':now(),'stage':'tracked','seeds':len(seeds),'frames':len(video)})
 print(json.dumps({'stage':'tracked','shape':list(tracks.shape)}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(D/'failure_track.json',{'time':now(),'type':type(e).__name__,'stack':[{'file':Path(v.filename).name,'line':v.lineno} for v in traceback.extract_tb(e.__traceback__)]})
  with (D/'failure_track.log').open('w') as z:traceback.print_exc(file=z)
  print('FAILED:'+type(e).__name__,flush=True);sys.exit(1)
