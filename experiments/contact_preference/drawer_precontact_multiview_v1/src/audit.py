import os,sys,json,hashlib,importlib.util
from pathlib import Path
os.environ['MUJOCO_GL']='egl'
import numpy as np,h5py
from scipy.spatial import cKDTree
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/contact_preference/drawer_precontact_multiview_v1';C=R/'experiments/contact_frames/cf017/cases/drawer_demo_0';sys.path.insert(0,str(R))
sp=importlib.util.spec_from_file_location('analysis_helpers',D/'src/analyze.py');B=importlib.util.module_from_spec(sp);sp.loader.exec_module(B);A=B.A
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 cfg=json.loads((D/'config.json').read_text());cases=json.loads((D/'cases.json').read_text());mot=np.load(C/'motion.npz');h=h5py.File(R/'experiments/contact_frames/cf017/rgbd/drawer_demo_0.h5');robot=A.RobotDepth(448)
 locked=json.loads((D/'input_locked.json').read_text());locked.update(json.loads((D/'seeds_locked.json').read_text()));changed=[p for p,v in locked.items() if sha(p)!=v];assert not changed
 source_checks=[]
 for c in cases:
  s=np.load(D/'data'/(c['key']+'_source.npz'));cam=c['camera'];sf=c['source_frame'];P=A.back(h[cam+'/depth_m'][sf],h[cam+'/K'][sf],h[cam+'/camera_to_world'][sf],s['uv']);err=float(np.max(np.linalg.norm(P-s['world'],axis=1)));assert err<1e-7;assert len(s['seeds_uv'])==600
  source_checks.append({'source':c['key'],'points':len(P),'reconstruction_max_m':err,'heldout':int(s['heldout'].sum())})
 ref=np.load(D/'identity_v2/robot0_eye_in_hand_40_to_60.npz');patch=np.flatnonzero(ref['contact_compatible']);assert len(patch)==94
 lineage=[]
 for t in cfg['target_frames']:
  a=np.load(D/'identity_v2'/('robot0_eye_in_hand_40_to_'+str(t)+'.npz'));H=mot['hand_to_world'][t];q=mot['measured_finger_q'][t]
  K=h['agentview/K'][t];T=h['agentview/camera_to_world'][t];robot.render(q,np.linalg.inv(H)@T,K);mesh,_=A.M.geo.surface(robot,[2,5]);mapped=a['mapped_world'][patch];valid=np.isfinite(mapped).all(1)
  gaps=np.full(len(patch),np.nan)
  if valid.any():gaps[valid]=A.M.geo.distance(mesh,a['mapped_hand'][patch][valid])[0]
  uvs=[];inside=[]
  for cam in cfg['cameras']:
   pp=np.zeros((len(patch),2));z=np.zeros(len(patch));pp[valid],z[valid]=A.proj(mapped[valid],h[cam+'/K'][t],h[cam+'/camera_to_world'][t]);uvs.append(pp.tolist());inside.append((valid&(z>0)&(pp[:,0]>=0)&(pp[:,0]<448)&(pp[:,1]>=0)&(pp[:,1]<448)).tolist())
  lineage.append({'frame':t,'source_ids':patch.tolist(),'source_uv':a['source_uv'][patch].tolist(),'target_uv':uvs,'in_view':inside,'mapped_hand':np.nan_to_num(a['mapped_hand'][patch]).tolist(),'valid':valid.tolist(),'supported':a['supported'][patch].tolist(),'contact':a['contact_compatible'][patch].tolist(),'hidden_camera_counts':a['hidden_by_finger'][:,patch].sum(1).tolist(),'visible_camera_counts':a['visible'][:,patch].sum(1).tolist(),'contact_count':int(a['contact_compatible'][patch].sum()),'supported_count':int(a['supported'][patch].sum()),'gap_min_median_max_mm':np.percentile(gaps[np.isfinite(gaps)]*1000,[0,50,100]).tolist() if np.isfinite(gaps).any() else None})
 # Camera-split sensitivity uses existing pooled inliers; not independent ground truth.
 base=np.load(D/'data/robot0_eye_in_hand_40_to_60.npz');row=json.loads((D/'data/robot0_eye_in_hand_40_to_60.json').read_text());k=int(base['source_cluster'][patch[0]]);ts=base['train_source'];lab=base['train_cluster'];preds=[];counts=[]
 for ci,cam in enumerate(cfg['cameras']):
  s=np.load(D/'data'/(cam+'_40_source.npz'));tr=np.load(D/'data'/(cam+'_40_tracks.npz'));ids=ts[(ts[:,0]==ci)&(lab==k),1];uv=np.clip(np.rint(tr['tracks'][20,ids]).astype(int),0,447);Y=A.back(h[cam+'/depth_m'][60],h[cam+'/K'][60],h[cam+'/camera_to_world'][60],uv);rr,tt=A.fit(s['seeds_world'][ids],Y);preds.append(A.apply(ref['source_world'][patch],rr,tt));counts.append(len(ids))
 splitdiff=np.linalg.norm(preds[0]-preds[1],axis=1)*1000
 support_other=[]
 for c in cases:
  if c['key']=='robot0_eye_in_hand_40':continue
  a=np.load(D/'identity_v2'/(c['key']+'_to_60.npz'));dd,ii=cKDTree(a['source_world']).query(ref['source_world'][patch]);ok=(dd<=.002)&a['supported'][ii];errs=np.linalg.norm(a['mapped_world'][ii[ok]]-ref['mapped_world'][patch[ok]],axis=1)*1000
  support_other.append({'source':c['key'],'same_initial_world_neighborhood_2mm':int((dd<=.002).sum()),'with_supported_mapping':int(ok.sum()),'mapped_difference_p95_mm':float(np.percentile(errs,95)) if len(errs) else None})
 audit={'time':now(),'input_lock_count':len(locked),'changed':changed,'source_checks':source_checks,'contact_source_points':94,'contact_1mm_cells':len(np.unique(np.floor(ref['mapped_world'][patch]/.001),axis=0)),'contact_gap_mm':np.percentile(ref['distance_visual_finger'][patch]*1000,[0,50,100]).tolist(),'contact_loo_max_mm':float(ref['loo_spread'][patch].max()*1000),'camera_split_inliers':counts,'camera_split_disagreement_mm':np.percentile(splitdiff,[0,50,95,100]).tolist(),'other_source_comparison':support_other,'physical_truth':False,'scoring_training':False}
 save(D/'audits/completion_numeric.json',audit);save(D/'lineage.json',lineage)
 print(json.dumps(audit,ensure_ascii=False));print('lineage',json.dumps([{k:x[k] for k in ['frame','supported_count','contact_count','gap_min_median_max_mm']} for x in lineage]))
 # Attach world/hand samples for optional 3D view, preserve provenance.
 j=json.loads((D/'summary_identity_v2.json').read_text())
 for r in j['rows']:
  a=np.load(D/'identity_v2'/(r['key']+'.npz'));keep=np.array(r['keep_ids'],int)
  r['kept_hand']=np.nan_to_num(a['mapped_hand'][keep]).tolist()
  b=json.loads((D/'data'/(r['key']+'.json')).read_text());r['before_identity_contact']=b['contact_compatible'];r['before_identity_localized']=b['localized_hidden']
 j['lineage']=lineage;j['audit']=audit;save(D/'review.json',j)
 robot.close();h.close()
if __name__=='__main__':main()
