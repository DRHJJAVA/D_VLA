"""Independent triangle rasterization audit; no renderer segmentation used as geometry."""
from pathlib import Path
import json,numpy as np
from numba import njit
from scipy.ndimage import binary_erosion,distance_transform_edt
ROOT=Path("/mnt/nvme1/drh/D_VLA")
@njit(cache=True)
def raster(q,faces,K,h,w):
 zbuf=np.full((h,w),np.inf)
 for face in faces:
  a,b,c=q[face[0]],q[face[1]],q[face[2]]
  if min(a[2],b[2],c[2])<=.001:continue
  x0=K[0,0]*a[0]/a[2]+K[0,2];y0=K[1,1]*a[1]/a[2]+K[1,2]
  x1=K[0,0]*b[0]/b[2]+K[0,2];y1=K[1,1]*b[1]/b[2]+K[1,2]
  x2=K[0,0]*c[0]/c[2]+K[0,2];y2=K[1,1]*c[1]/c[2]+K[1,2]
  den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
  if abs(den)<1e-12:continue
  xmin=max(0,int(np.floor(min(x0,x1,x2))));xmax=min(w-1,int(np.ceil(max(x0,x1,x2))))
  ymin=max(0,int(np.floor(min(y0,y1,y2))));ymax=min(h-1,int(np.ceil(max(y0,y1,y2))))
  for y in range(ymin,ymax+1):
   for x in range(xmin,xmax+1):
    xx=x+.5;yy=y+.5
    u=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den
    v=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den
    ww=1-u-v
    if u>=-1e-8 and v>=-1e-8 and ww>=-1e-8:
     depth=1/(u/a[2]+v/b[2]+ww/c[2])
     if depth<zbuf[y,x]:zbuf[y,x]=depth
 return zbuf
def checks(pred,mask):
 pb=pred&~binary_erosion(pred);mb=mask&~binary_erosion(mask)
 if not pb.any() or not mb.any():return {"valid":False}
 d=np.r_[distance_transform_edt(~mb)[pb],distance_transform_edt(~pb)[mb]]
 return {"valid":True,"symmetric_boundary_median_px":float(np.median(d)),"symmetric_boundary_p95_px":float(np.percentile(d,95)),"silhouette_iou":float((pred&mask).sum()/(pred|mask).sum())}
def audit(path):
 e=json.loads(path.read_text())
 if e.get("triangle_projection_checks"):return
 obj=e["object"];s=np.load(ROOT/"data/v2/stage1"/obj/"surface.npz")
 row=json.loads((ROOT/"data/v2/stage1"/obj/(e["episode"]+"_states_contacts.json")).read_text())["rows"][e["frame"]]
 world=s["vertices"]@np.array(row["object_rotation"]).T+row["object_position"];results={}
 for cam in ["agentview","robot0_eye_in_hand"]:
  v=np.load(path.parent/(cam+"_"+str(e["frame"])+".npz"));T=v["camera_to_world"];q=(world-T[:3,3])@T[:3,:3]
  z=raster(q,s["faces"],v["K"],448,448);depth=v["depth_m"];mask=v["target_mask_offline"]
  predicted=np.isfinite(z)&(z<=depth+.003)
  report=checks(predicted,mask);joint=mask&np.isfinite(z)
  report.update(median_depth_error_mm=float(np.median(np.abs(z[joint]-depth[joint]))*1000),p95_depth_error_mm=float(np.percentile(np.abs(z[joint]-depth[joint]),95)*1000),method="independent full visual-mesh triangle rasterizer, perspective-correct depth, fixed 3mm occlusion tolerance")
  results[cam]=report
 e["triangle_projection_checks"]=results
 tmp=path.with_suffix(".json.tmp");tmp.write_text(json.dumps(e,ensure_ascii=False,indent=2));tmp.replace(path)
 print(e["id"],json.dumps(results),flush=True)
def unit_check():
 K=np.array([[10.,0,0],[0,10.,0],[0,0,1.]])
 q=np.array([[0.,0,2.],[2.,0,2.],[0.,2.,2.]])
 f=np.array([[0,1,2]])
 z=raster(q,f,K,12,12)
 assert z[2,2]==2 and np.isinf(z[11,11])
 qq=np.concatenate([q,q*.5]);ff=np.array([[0,1,2],[3,4,5]])
 assert raster(qq,ff,K,12,12)[2,2]==1
 assert np.array_equal(raster(qq,ff,K,12,12),raster(qq,ff[::-1].copy(),K,12,12))
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--case");a=p.parse_args();unit_check()
 for path in sorted((ROOT/"audits/v2/stage1").glob("*/summary.json")):
  if a.case and path.parent.name!=a.case:continue
  audit(path)
