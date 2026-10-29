from pathlib import Path
import numpy as np,h5py,json,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
B=Path('/mnt/nvme1/drh/D_VLA')
R=B/'experiments/contact_preference/stage3_pilot_v2'
out=R/'alignment_review';out.mkdir(exist_ok=True)
cases=json.loads((R/'evidence_summary.json').read_text())['cases']
report=[]
for ci,c in enumerate(cases):
 key=c['key'];reg=np.load(B/'experiments/region_pose/iter_024/train'/(key+'.npz'))
 q=int(key.split('query')[-1])
 hp=B/'data/v3/trainval_rgbd_448'/c['asset']/(c['demo']+'.h5')
 with h5py.File(hp,'r') as h:
  def proj(xyz,cam,t):
   K=h[cam+'/K'][t];T=h[cam+'/camera_to_world'][t].astype(float)
   p=np.linalg.solve(T,np.c_[xyz,np.ones(len(xyz))].T).T[:,:3]
   uv=p@K.T;uv=uv[:,:2]/uv[:,2:]
   return uv,p[:,2]
  uv,z=proj(reg['query_xyz'],'agentview',q)
  pixel=reg['query_pixel']
  xyerr=float(abs(uv-pixel).max());yxerr=float(abs(uv-pixel[:,::-1]).max())
  pix=pixel if xyerr<yxerr else pixel[:,::-1]
  pix=np.rint(pix).astype(int)
  dep=h['agentview/depth_m'][q]
  rgb=h['agentview/rgb'][q]
  qr={'q':q,'query_reprojection_xy_max':xyerr,'query_reprojection_yx_max':yxerr,'source_depth_error_max_m':float(abs(z-dep[pix[:,1],pix[:,0]]).max()),'source_rgb_equal':bool(np.array_equal(reg['query_rgb'],rgb[pix[:,1],pix[:,0]]))}
  for row in c['rows']:
   t=row['frame'];a=np.load(R/row['data']);gr=a['display_class']==0
   record={'case':ci,'key':key,'frame':t,'source_check':qr,'cameras':{}}
   for cam in ['agentview','robot0_eye_in_hand']:
    uv,z=proj(a['mapped_world_xyz'],cam,t);ij=np.rint(uv).astype(int)
    ok=(z>0)&(ij[:,0]>=0)&(ij[:,0]<448)&(ij[:,1]>=0)&(ij[:,1]<448)
    dd=np.full(len(z),np.nan);dd[ok]=h[cam+'/depth_m'][t][ij[ok,1],ij[ok,0]]
    residual=z-dd
    stats={}
    for name,sel in [('all',ok),('green',ok&gr)]:
     v=residual[sel];stats[name]={'in_image':int(sel.sum()),'depth_agree_3mm':int((abs(v)<=.003).sum()),'in_front_gt3mm':int((v<-.003).sum()),'behind_gt3mm':int((v>.003).sum()),'median_mm':float(np.median(v)*1000) if len(v) else None,'p95_abs_mm':float(np.percentile(abs(v),95)*1000) if len(v) else None}
    record['cameras'][cam]=stats
    if ci==2:
     np.savez_compressed(out/(cam+'_f%d_projection.npz'%t),uv=uv,z=z,measured_depth=dd,residual=residual,in_image=ok,green=gr)
   report.append(record)
(out/'depth_alignment.json').write_text(json.dumps({'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),'rows':report,'interpretation':'Behind includes real occlusion; front residual is visible geometry conflict but not identity/contact truth.'},ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
