"""Render stage-1 evidence only. Does not train or certify labels."""
import os,sys,json,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage1_collect import ROOT,LIB,OUT,TASKS,save,resolve,restore,state,ids
import numpy as np,h5py
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from restore_audit import OffScreenRenderEnv
from robosuite.utils.camera_utils import get_real_depth_map,get_camera_intrinsic_matrix,get_camera_extrinsic_matrix
FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"]=font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"]=False
REVIEW=ROOT/"audits/v2/stage1"
ZH={"alphabet_soup":"字母汤罐","tomato_sauce":"番茄酱罐","porcelain_mug":"白瓷杯","white_yellow_mug":"黄白杯"}
CAMS=["agentview","robot0_eye_in_hand"]
def gap(row,points):
 op=np.array(row["object_position"]);oR=np.array(row["object_rotation"])
 world=points@oR.T+op
 p0,p1=row["pads"];R=np.array(p0["rotation"]);a=np.array(p0["position"]);b=np.array(p1["position"])
 assert np.max(np.abs(R-np.array(p1["rotation"])))<1e-5,"nonparallel pads"
 s0=np.array(p0["halfsize"]);s1=np.array(p1["halfsize"]);c=(b-a)@R
 axis=int(np.argmax(np.abs(c)))
 assert axis==1,"unexpected closing axis"
 lo=np.maximum(-s0,c-s1);hi=np.minimum(s0,c+s1)
 if c[axis]>0:lo[axis]=s0[axis];hi[axis]=c[axis]-s1[axis]
 else:lo[axis]=c[axis]+s1[axis];hi[axis]=-s0[axis]
 q=(world-a)@R
 margin=np.minimum(q-lo,hi-q).min(axis=1)
 return margin,lo,hi,R,a
def labels(points,rows,event):
 tree=cKDTree(points);cv=[];ev=[];res=[];raw=[];projected=[];frames=[]
 for t in event["window"]:
  r=rows[t];contact=np.array([c["object"] for c in r["contacts"]]);dist,index=tree.query(contact)
  res.extend(dist.tolist());raw.extend(contact.tolist());valid=dist<=.003
  cp=points[index[valid]];projected.extend(cp.tolist())
  mask=np.zeros(len(points),bool)
  if len(cp):mask=cKDTree(cp).query(points)[0]<=.002
  margin,*_=gap(r,points)
  cv.append(mask);ev.append(margin>.0005);frames.append(margin)
 # Strict majority in the stable window; uncertain geometry boundary is retained.
 contact=np.mean(cv,axis=0)>.5;enclosed=np.mean(ev,axis=0)>.5
 ignore=(np.mean(np.abs(frames)<=.0005,axis=0)>.5)
 return contact,enclosed,ignore,np.array(raw),np.array(projected),np.array(res)
def corners(pad,row):
 s=np.array(pad["halfsize"]);v=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*s
 w=v@np.array(pad["rotation"]).T+np.array(pad["position"])
 return (w-np.array(row["object_position"]))@np.array(row["object_rotation"])
FACES=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
def cloud(ax,p,mask=None,color="#e04135",row=None,raw=None):
 idx=np.linspace(0,len(p)-1,min(20000,len(p))).astype(int)
 ax.scatter(*p[idx].T,s=.15,c="#bac4cd",alpha=.2,rasterized=True)
 if mask is not None and mask.any():ax.scatter(*p[mask].T,s=.9,c=color,alpha=.9,rasterized=True)
 if raw is not None and len(raw):ax.scatter(*raw.T,s=18,c="#d21c67",marker="x",depthshade=False)
 if row:
  for pad in row["pads"]:
   v=corners(pad,row);ax.add_collection3d(Poly3DCollection([v[f] for f in FACES],facecolors="#515d6c",edgecolors="#253044",alpha=.16,linewidths=.5))
 mid=(p.max(0)+p.min(0))/2;span=float(np.ptp(p,axis=0).max())*.64
 ax.set(xlim=(mid[0]-span,mid[0]+span),ylim=(mid[1]-span,mid[1]+span),zlim=(mid[2]-span,mid[2]+span))
 ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-55);ax.set_axis_off()
def render_one(env,ep,obj,event,rows,points):
 eid=event["id"];dest=REVIEW/eid;dest.mkdir(parents=True,exist_ok=True)
 contact,enclosed,ignore,raw,proj,res=labels(points,rows,event)
 t=event["window"][len(event["window"])//2];r=rows[t]
 np.savez_compressed(dest/"labels.npz",points_object=points,contact=contact,enclosed=enclosed,union=contact|enclosed,ignore=ignore,raw_contacts_object=raw,projected_contacts_object=proj,projection_distance_m=res)
 targets,pads,body=ids(env,obj);views={};frames=sorted({max(0,event["start"]-5),event["start"],t,min(event["end"],t+10)})
 for frame in frames:
  state(env,ep,frame)
  for camera in CAMS:
   rgb,buf=env.sim.render(camera_name=camera,width=448,height=448,depth=True)
   rgb=np.flipud(rgb).copy();depth=np.flipud(get_real_depth_map(env.sim,buf)).copy()
   seg=np.flipud(env.sim.render(camera_name=camera,width=448,height=448,segmentation=True)[...,1]).copy()
   K=get_camera_intrinsic_matrix(env.sim,camera,448,448);T=get_camera_extrinsic_matrix(env.sim,camera)
   v,u=np.indices(depth.shape);xyz=np.stack([(u-K[0,2])*depth/K[0,0],(v-K[1,2])*depth/K[1,1],depth],-1)@T[:3,:3].T+T[:3,3]
   mask=np.isin(seg,targets)
   np.savez_compressed(dest/(camera+"_"+str(frame)+".npz"),rgb=rgb,depth_m=depth,K=K,camera_to_world=T,xyz_world=xyz,target_mask_offline=mask)
   views[(frame,camera)]=(rgb,depth,mask)
   if frame==t:Image.fromarray(rgb).save(dest/(camera+"_rgb.png"))
 # Keep the entire recorded demonstration RGB stream for success/lift inspection.
 original=np.asarray(ep["obs/agentview_rgb"])
 Image.fromarray(original[0]).save(dest/"demonstration.gif",save_all=True,append_images=[Image.fromarray(x) for x in original[2::2]],duration=100,loop=0)
 fig=plt.figure(figsize=(15,9),layout="constrained")
 gs=fig.add_gridspec(2,3,height_ratios=[1,1.25])
 rgb,depth,mask=views[(t,"agentview")];wrgb,wd,wma=views[(t,"robot0_eye_in_hand")]
 ax=fig.add_subplot(gs[0,0]);ax.imshow(rgb);ax.set_title("1 采集 RGB：抬起后的稳定夹持");ax.axis("off")
 ax=fig.add_subplot(gs[0,1]);im=ax.imshow(depth,cmap="viridis",vmin=.15,vmax=1.5);ax.set_title("2 同步深度（米）");ax.axis("off");fig.colorbar(im,ax=ax,fraction=.035,pad=.02)
 ax=fig.add_subplot(gs[0,2]);ax.imshow(wrgb);ax.set_title("3 手腕相机：观察夹指与物体");ax.axis("off")
 for col,title,msk,colr,showraw in [(0,"4 物体完整表面与两侧夹指",None,"#b7c1ca",False),(1,"5 红色：接触邻域；粉叉：原始接触",contact,"#e04135",True),(2,"6 蓝色：两夹指间的包裹表面",enclosed,"#167ddd",False)]:
  ax=fig.add_subplot(gs[1,col],projection="3d");cloud(ax,points,msk,colr,r,raw if showraw else None);ax.set_title(title)
 fig.suptitle(ZH[obj]+" | "+event["episode"]+" | 帧 "+str(t)+"\n灰色为其余表面；下排完整表面来自仿真资产，仅用于标签审计，不是模型输入。",fontsize=16)
 fig.savefig(dest/"comparison.png",dpi=130);plt.close(fig)
 # Independent visual-mesh versus rendered depth consistency, with visibility rejection.
 state(env,ep,t);m=env.sim.model;d=env.sim.data;world=points@d.body_xmat[body].reshape(3,3).T+d.body_xpos[body]
 checks={}
 for camera in CAMS:
  rgb,dep,mask=views[(t,camera)];K=get_camera_intrinsic_matrix(env.sim,camera,448,448);T=get_camera_extrinsic_matrix(env.sim,camera)
  q=(world-T[:3,3])@T[:3,:3];u=np.rint(K[0,0]*q[:,0]/q[:,2]+K[0,2]).astype(int);v=np.rint(K[1,1]*q[:,1]/q[:,2]+K[1,2]).astype(int)
  inside=(u>=0)&(u<448)&(v>=0)&(v<448)&(q[:,2]>0);ii=np.flatnonzero(inside);ii=ii[mask[v[ii],u[ii]]]
  delta=np.abs(q[ii,2]-dep[v[ii],u[ii]]);visible=delta<.003
  checks[camera]={"target_pixels":int(mask.sum()),"mesh_surface_points_agree_with_depth_3mm":int(visible.sum()),"median_visible_depth_residual_m":float(np.median(delta[visible])) if visible.any() else None}
 summary={**event,"object":obj,"name_zh":ZH[obj],"frame":t,"contact_points":int(contact.sum()),"enclosed_points":int(enclosed.sum()),"surface_points":len(points),
 "contact_projection_median_mm":float(np.median(res)*1000),"contact_projection_max_mm":float(np.max(res)*1000),"contact_projection_rejected_fraction":float(np.mean(res>.003)),
 "parameters":{"projection_limit_m":.003,"contact_neighborhood_m":.002,"enclosure_boundary_ignore_m":.0005,"window":event["window"],"vote":"strict majority"},
 "camera_checks":checks,"manual_review":"pending","grasp_mode":"pending_visual_review","full_rgbd":"keyframes_only_pending_full_replay"}
 save(dest/"summary.json",summary)
 # JSON for browser orbit view: compact points + exact categorical flags.
 idx=np.unique(np.concatenate([np.linspace(0,len(points)-1,12000).astype(int),np.flatnonzero(contact),np.flatnonzero(enclosed)]))
 save(dest/"cloud.json",{"points":np.round(points[idx],5).tolist(),"contact":contact[idx].astype(int).tolist(),"enclosed":enclosed[idx].astype(int).tolist(),"pads":[corners(p,r).tolist() for p in r["pads"]]})
 print("RENDERED",eid,"contact",contact.sum(),"enclosed",enclosed.sum(),"projection_mm",np.median(res)*1000,flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument("--limit",type=int,default=0);a.add_argument("--object",default="all");args=a.parse_args()
 REVIEW.mkdir(parents=True,exist_ok=True)
 for obj,path,n in TASKS:
  if args.object not in ("all",obj):continue
  mf=OUT/obj/"manifest.json"
  if not mf.exists():continue
  manifest=json.loads(mf.read_text());events=manifest["events"][:n]
  if args.limit:events=events[:args.limit]
  missing=[e for e in events if not (REVIEW/e["id"]/"summary.json").exists()]
  if not missing:continue
  points=np.load(OUT/obj/"surface.npz")["points_object"]
  with h5py.File(manifest["source"]) as f:
   d=f["data"];env=OffScreenRenderEnv(bddl_file_name=str(resolve(d)),use_camera_obs=False)
   try:
    for e in missing:
     ep=d[e["episode"]];restore(env,ep)
     rows=json.loads((OUT/obj/(e["episode"]+"_states_contacts.json")).read_text())["rows"]
     render_one(env,ep,obj,e,rows,points)
   finally:env.close()
if __name__=="__main__":main()
