"""Finalize stage 1 evidence and stop. Never starts training."""
import os,sys,time,json,csv,subprocess
from pathlib import Path
ROOT=Path("/mnt/nvme1/drh/D_VLA");REVIEW=ROOT/"audits/v2/stage1";SRC=ROOT/"src/v2"
def save(p,d):
 t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
def main():
 deadline=time.monotonic()+7200
 while len(list(REVIEW.glob("*/summary.json")))<30:
  if time.monotonic()>deadline:raise TimeoutError("stage1 images incomplete")
  time.sleep(15)
 for script in ["stage1_enrich_review.py","stage1_projection_audit.py"]:
  subprocess.run([sys.executable,str(SRC/script)],check=True)
 import h5py,numpy as np
 from PIL import Image
 sys.path.insert(0,str(SRC))
 from stage1_visualize import gap,cloud,plt,ZH
 records=[];episode_records=[];frames=0
 for pp in sorted(REVIEW.glob("*/summary.json")):
  e=json.loads(pp.read_text());obj=e["object"];ep=e["episode"];dest=pp.parent
  manifest=json.loads((ROOT/"data/v2/stage1"/obj/"manifest.json").read_text())
  rows=json.loads((ROOT/"data/v2/stage1"/obj/(ep+"_states_contacts.json")).read_text())["rows"]
  with h5py.File(manifest["source"]) as f:
   source=f["data/"+ep];a=np.asarray(source["obs/agentview_rgb"])[:,::-1,:,:]
   count=len(source["states"]);times=np.asarray(source["states"][:,0]).tolist()
  if e.get("rgb_gif_convention")!="opencv":
   Image.fromarray(a[0]).save(dest/"demonstration.gif",save_all=True,append_images=[Image.fromarray(x) for x in a[2::2]],duration=100,loop=0)
   e["rgb_gif_convention"]="opencv"
  full=ROOT/"data/v2/stage1_rgbd"/obj/(ep+".h5")
  while not full.exists():
   if time.monotonic()>deadline:raise TimeoutError(str(full))
   time.sleep(10)
  with h5py.File(full) as f:
   cc={}
   for cam in ["agentview","robot0_eye_in_hand"]:
    g=f[cam];assert len(g["rgb"])==len(g["depth_m"])==len(g["K"])==len(g["camera_to_world"])==count
    sampled=g["depth_m"][[0,count//2,count-1]];assert np.isfinite(sampled).all() and sampled.min()>0
    cc[cam]={"frames":count,"resolution":list(g["rgb"].shape[1:3]),"depth_finite_checked_frames":[0,count//2,count-1]}
  save(full.parent/(ep+"_alignment.json"),{"source":manifest["source"],"episode":ep,"frame_mapping":"RGB-D frame i corresponds to original state/action frame i","sim_time":times,"camera_checks":cc})
  e["full_rgbd"]="complete";e["full_rgbd_path"]=str(full)
  # Retain exact geometric region separately from the conservative training-valid core.
  labels=np.load(dest/"labels.npz");points=labels["points_object"]
  margins=np.array([gap(rows[t],points)[0] for t in e["window"]])
  geometric=(margins>=0).mean(axis=0)>.5
  uncertain=geometric&~labels["enclosed"]
  np.savez_compressed(dest/"enclosure_geometry_audit.npz",geometric_enclosure=geometric,conservative_enclosure=labels["enclosed"],boundary_uncertain=uncertain)
  e["enclosed_geometric_points"]=int(geometric.sum());e["enclosed_boundary_uncertain_points"]=int(uncertain.sum())
  e["quality_flags"]=[]
  if not labels["enclosed"].any():e["quality_flags"].append("no_confident_enclosure_after_0.5mm_boundary_ignore")
  if e["contact_projection_rejected_fraction"]>0:e["quality_flags"].append("some_contact_projections_rejected")
  if any(v["symmetric_boundary_median_px"]>2 for v in e["triangle_projection_checks"].values()):e["quality_flags"].append("projection_median_gt_2px")
  # Add uncertainty to interactive full-surface cloud using nearest surface mapping.
  from scipy.spatial import cKDTree
  for name in ["cloud.json","observed_cloud.json"]:
   cp=dest/name;d=json.loads(cp.read_text());distance,idx=cKDTree(points).query(d["points"])
   d["uncertain"]=(uncertain[idx]&(distance<=.003)).astype(int).tolist();save(cp,d)
  # Independent close-up of the entire enclosure and its uncertain boundary.
  fig=plt.figure(figsize=(9,4.5),layout="constrained")
  for col,az in enumerate([-55,125]):
   ax=fig.add_subplot(1,2,col+1,projection="3d");cloud(ax,points,labels["enclosed"],"#167ddd",rows[e["frame"]])
   if uncertain.any():ax.scatter(*points[uncertain].T,s=.9,c="#e4a21c",alpha=.9,rasterized=True)
   ax.view_init(elev=25,azim=az);ax.set_title("视角 "+str(col+1))
  fig.suptitle(ZH[obj]+" "+ep+"：包裹区边界审计\n蓝色＝可靠内部；橙色＝几何包裹内但距边界不足 0.5 mm，暂不作确定标签",fontsize=12)
  fig.savefig(dest/"enclosure_audit.png",dpi=140);plt.close(fig)
  save(pp,e);records.append(e);frames+=count
  episode_records.append({"object":obj,"episode":ep,"rgbd_path":str(full),"frames":count,"cameras":cc})
  print("FINALIZED",e["id"],e["quality_flags"],flush=True)
 counts={o:sum(x["object"]==o for x in records) for o in sorted({x["object"] for x in records})}
 manifests=[json.loads(p.read_text()) for p in (ROOT/"data/v2/stage1").glob("*/manifest.json")]
 summary={"stage":1,"status":"awaiting_user_visual_review","stage2_authorized":False,"events":len(records),"object_counts":counts,"full_rgbd_episodes":len(episode_records),"synchronized_time_steps":frames,"camera_frames":frames*2,
 "restored_episodes_scanned":sum(len(m["episodes"]) for m in manifests),"max_state_roundtrip_error":max(ep["state_roundtrip_error"] for m in manifests for ep in m["episodes"]),
 "projection_median_gate_pass_events":sum(all(v["symmetric_boundary_median_px"]<=2 for v in e["triangle_projection_checks"].values()) for e in records),
 "manual_review_completed":0,"manual_review_required":30,"automatic_quality_flags":{e["id"]:e["quality_flags"] for e in records if e["quality_flags"]},
 "remaining":["30-event human review, including successful lift/rotation/slip and contact/enclosure labels","cup body/handle/rim grasp-mode coverage remains unverified; supplement missing modes within stage 1","two cans have almost identical dimensions; no size-generalization claim","the boundary-uncertain enclosure example must be accepted/relabelled/excluded before training"],"episodes":episode_records}
 save(REVIEW/"batch_summary.json",summary)
 q=REVIEW/"manual_review.csv"
 if not q.exists():
  with q.open("w",newline="") as f:
   w=csv.writer(f);w.writerow(["event_id","object","episode","lift_cm","relative_drift_mm","auto_flags","grasp_mode_manual","lift_confirmed","contact_correct","enclosure_correct","reviewer","verdict","notes"])
   for e in records:w.writerow([e["id"],e["object"],e["episode"],round(e["lift_m"]*100,2),round(e["drift_m"]*1000,2),";".join(e["quality_flags"]),"","","","","","",""])
 doc="# 第一阶段：采集数据与抓取区域可视化\n\n状态：首批数据已生成，待用户验收；未启动第二阶段。\n\n"
 doc+="## 采集了什么\n\n"
 for obj,n in counts.items():doc+="- "+ZH[obj]+"："+str(n)+" 个数值筛选合格的夹持候选。\n"
 doc+="\n共 "+str(len(records))+" 个事件，"+str(len(episode_records))+" 段完整双相机 RGB-D，"+str(frames)+" 个同步时间步。原始示范、动作和状态保留在 data/raw；第一阶段新增深度、内外参、时间映射保留在 data/v2/stage1_rgbd。字母汤罐复用既有 448×448 重建；其余完整录像为 256×256，各事件关键帧统一为 448×448。训练数据分辨率尚未定稿。\n\n"
 doc+="## 看哪些图\n\n打开远程 audits/v2/stage1/index.html 的验收服务。每个事件提供完整场景 RGB、深度、夹持位置放大、实际观测点云、接触区、包裹区、完整 RGB 示范动画及旋转点云。点击点云来源可切换完整场景、可见物体和资产完整表面。\n\n红色：接触投影 2 mm 邻域；蓝色：包裹区可靠内部；橙色：几何包裹边界不确定区；粉叉：原始仿真接触；灰色：其余表面。颜色表示区域类型，不表示模型评分。\n\n"
 doc+="## 自动检查结果\n\n恢复 "+str(summary["restored_episodes_scanned"])+" 段示范，最大状态往返误差 "+str(summary["max_state_roundtrip_error"])+"。独立完整三角网格投影：两相机轮廓中位误差均不超过 2 像素的事件为 "+str(summary["projection_median_gate_pass_events"])+"/30。每个事件的 95% 分位误差、轮廓重合度和深度误差保留在 summary.json，不把中位数合格等同于所有边界都准确。\n\n"
 doc+="接触投影大于 3 mm 的点不纳入接触标签。包裹体由真实夹指内表面、16 mm 接触垫宽度及高度限定；稳定 5 帧多数表决，距边界 0.5 mm 内保留为不确定。完整资产表面只用于标签审计，不作为相机观测输入。\n\n"
 doc+="## 尚未通过的验收项\n\n人工复核完成 0/30；未确认杯身、杯柄、杯口三种抓取方式覆盖。部分物体只能抓柄或杯口是否可行仍须实测。两种罐体尺寸近似，不能据此声称尺寸泛化。\n\n"
 for eid,flags in summary["automatic_quality_flags"].items():doc+="- "+eid+"："+", ".join(flags)+"。\n"
 doc+="\n特别是番茄酱罐 demo_7：存在约 106 个几何包裹点，但它们都位于 0.5 mm 边界不确定带，因此可靠蓝色区域为空；不能把这个事件直接当作可靠包裹监督。已另附双视角 enclosure_audit.png，橙色显示不确定区域。\n\n请先验收标签定义及具体样例。缺失抓取方式、异常标签处理和人工抽查仍属于第一阶段；未获得用户验收，不启动下一阶段。\n"
 report=ROOT/"docs/第一阶段_采集与区域可视化报告.md";report.write_text(doc)
 status=ROOT/"docs/STATUS.md";old=status.read_text();tag="\n## 2026-10-23 第一阶段首批可视化\n"
 if tag not in old:status.write_text(old+tag+"\n已生成四物体 30 个候选事件及完整 RGB-D，详见 [第一阶段报告](第一阶段_采集与区域可视化报告.md)。当前等待用户视觉验收，人工复核 0/30；抓取方式覆盖和边界异常尚待处理。未启动第二阶段，全部产物留在服务器。\n")
 print("STAGE1_REVIEW_READY",json.dumps({k:v for k,v in summary.items() if k!="episodes"},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
