"""Add observed RGB-D point clouds and geometric projection checks to review output."""
import sys,json,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage1_visualize import ROOT,OUT,REVIEW,CAMS,ZH,save,cloud,plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion,distance_transform_edt
def enrich(dest):
 e=json.loads((dest/"summary.json").read_text())
 if e.get("observed_cloud"):return
 obj=e["object"];rows=json.loads((OUT/obj/(e["episode"]+"_states_contacts.json")).read_text())["rows"]
 lab=np.load(dest/"labels.npz");mesh=lab["points_object"];tree=cKDTree(mesh);t=e["frame"];r=rows[t]
 frames=sorted(int(p.stem.rsplit("_",1)[1]) for p in dest.glob("agentview_*.npz"));before=frames[0]
 ri=rows[before];points=[];rgbpoints=[];scenes=[];scolors=[]
 for cam in CAMS:
  v=np.load(dest/(cam+"_"+str(before)+".npz"));valid=(v["depth_m"]>0)&(v["depth_m"]<3)&np.isfinite(v["depth_m"]);mask=v["target_mask_offline"]&valid
  world=v["xyz_world"][mask];p=(world-np.array(ri["object_position"]))@np.array(ri["object_rotation"])
  points.append(p);rgbpoints.append(v["rgb"][mask])
  grid=np.zeros(valid.shape,bool);grid[::3,::3]=True;ss=grid&valid
  scenes.append((v["xyz_world"][ss]-np.array(ri["object_position"]))@np.array(ri["object_rotation"]));scolors.append(v["rgb"][ss])
 p=np.concatenate(points);rgb=np.concatenate(rgbpoints);dist,idx=tree.query(p)
 visible_label_known=dist<=.003
 c=lab["contact"][idx]&visible_label_known;g=lab["enclosed"][idx]&visible_label_known
 np.savez_compressed(dest/"observed_object_points.npz",points_object=p,rgb=rgb,contact=c,enclosed=g,valid_label=visible_label_known,source_frame=before)
 pads=json.loads((dest/"cloud.json").read_text())["pads"]
 save(dest/"observed_cloud.json",{"display_bounds":[mesh.min(0).tolist(),mesh.max(0).tolist()],"points":np.round(p,5).tolist(),"rgb":rgb.tolist(),"contact":c.astype(int).tolist(),"enclosed":g.astype(int).tolist(),"pads":pads})
 sp=np.concatenate(scenes);sc=np.concatenate(scolors);take=np.linspace(0,len(sp)-1,min(30000,len(sp))).astype(int)
 save(dest/"scene_cloud.json",{"points":np.round(sp[take],5).tolist(),"rgb":sc[take].tolist(),"contact":[0]*len(take),"enclosed":[0]*len(take),"pads":[]})
 # Independent sampled-mesh projection boundary check, occlusion filtered by scene depth.
 checks={}
 world=mesh@np.array(r["object_rotation"]).T+np.array(r["object_position"])
 for cam in CAMS:
  v=np.load(dest/(cam+"_"+str(t)+".npz"));dep=v["depth_m"];T=v["camera_to_world"];K=v["K"];mask=v["target_mask_offline"]
  q=(world-T[:3,3])@T[:3,:3];u=np.rint(K[0,0]*q[:,0]/q[:,2]+K[0,2]).astype(int);yy=np.rint(K[1,1]*q[:,1]/q[:,2]+K[1,2]).astype(int)
  inside=(u>=0)&(u<448)&(yy>=0)&(yy<448)&(q[:,2]>0)
  buf=np.full(448*448,np.inf);np.minimum.at(buf,yy[inside]*448+u[inside],q[inside,2]);buf=buf.reshape(448,448)
  predicted=np.isfinite(buf)&(buf<=dep+.003)
  pb=predicted&~binary_erosion(predicted);gb=mask&~binary_erosion(mask)
  if gb.any() and pb.any():
   distances=np.r_[distance_transform_edt(~gb)[pb],distance_transform_edt(~pb)[gb]]
   checks[cam]={"symmetric_boundary_median_px":float(np.median(distances)),"symmetric_boundary_p95_px":float(np.percentile(distances,95)),"silhouette_iou":float((mask&predicted).sum()/(mask|predicted).sum()),"method":"80k visual-mesh surface splats, independent pinhole projection, scene-depth occlusion tolerance 3mm"}
 # More legible input/output figure. Display observed object cloud separately from full surface labels.
 fig=plt.figure(figsize=(15,9),layout="constrained");gs=fig.add_gridspec(2,3,height_ratios=[1,1.25])
 v=np.load(dest/("agentview_"+str(t)+".npz"));im=v["rgb"];dep=v["depth_m"];mask=v["target_mask_offline"]
 ax=fig.add_subplot(gs[0,0]);ax.imshow(im);ax.set_title("1 完整场景 RGB（稳定夹持帧）");ax.axis("off")
 ax=fig.add_subplot(gs[0,1]);d=ax.imshow(dep,cmap="viridis",vmin=.15,vmax=1.5);fig.colorbar(d,ax=ax,fraction=.035,pad=.02);ax.set_title("2 同步深度（单位：米）");ax.axis("off")
 ys,xs=np.where(mask);x0=max(0,int(xs.min())-35);x1=min(448,int(xs.max())+35);y0=max(0,int(ys.min())-40);y1=min(448,int(ys.max())+20)
 ax=fig.add_subplot(gs[0,2]);ax.imshow(im[y0:y1,x0:x1]);ax.set_title("3 夹持位置放大（原图裁剪展示）");ax.axis("off")
 ax=fig.add_subplot(gs[1,0],projection="3d");cloud(ax,mesh)
 # Show only measured samples in this panel; remove the audit mesh scatter.
 for item in list(ax.collections):item.remove()
 ax.scatter(*p.T,c=rgb/255,s=.5,rasterized=True);ax.set_title("4 RGB-D 实际观测点云（夹持前）")
 for col,title,msk,color,raw in [(1,"5 红色：接触邻域；粉叉：原始接触",lab["contact"],"#e04135",lab["raw_contacts_object"]),(2,"6 蓝色：两夹指间的包裹表面",lab["enclosed"],"#167ddd",None)]:
  ax=fig.add_subplot(gs[1,col],projection="3d");cloud(ax,mesh,msk,color,r,raw);ax.set_title(title)
 fig.suptitle(ZH[obj]+" | "+e["episode"]+" | 稳定帧 "+str(t)+"\n左下为真实深度观测；右下两图为完整表面的标签审计，灰色为其余表面。",fontsize=15)
 fig.savefig(dest/"comparison.png",dpi=140);plt.close(fig)
 e.update(observed_cloud=True,observed_frame=before,observed_target_points=len(p),observed_label_valid_fraction=float(visible_label_known.mean()),point_splat_diagnostic_only=checks)
 save(dest/"summary.json",e);print("ENRICHED",e["id"],len(p),checks,flush=True)
if __name__=="__main__":
 for p in sorted(REVIEW.glob("*/summary.json")):enrich(p.parent)
