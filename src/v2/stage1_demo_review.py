"""Stage-1 demo-only evidence. Never turns geometric audit regions into targets."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
root=Path("/mnt/nvme1/drh/D_VLA");audit=root/"audits/v2/stage1"
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"]=font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name()
plt.rcParams["axes.unicode_minus"]=False
dest=audit/"demo_review";dest.mkdir(exist_ok=True)
rowsout=[]
for summary_file in sorted(audit.glob("*/summary.json")):
 event=json.loads(summary_file.read_text());obj=event["object"];episode=event["episode"];dir=summary_file.parent
 records=json.loads((root/"data/v2/stage1"/obj/(episode+"_states_contacts.json")).read_text())["rows"]
 frames=sorted(int(p.stem.rsplit("_",1)[1]) for p in dir.glob("agentview_*.npz"))
 assert len(frames)>=3,(obj,episode,frames)
 fig=plt.figure(figsize=(15,10),layout="constrained");axes=fig.subplots(3,len(frames))
 for col,t in enumerate(frames):
  av=np.load(dir/("agentview_"+str(t)+".npz"));wv=np.load(dir/("robot0_eye_in_hand_"+str(t)+".npz"))
  axes[0,col].imshow(av["rgb"]);axes[0,col].set_title(["接近前","首次双指接触","稳定夹持","抬升后"][min(col,3)]+f" · 帧 {t}")
  im=axes[1,col].imshow(av["depth_m"],cmap="viridis",vmin=.15,vmax=1.5);axes[1,col].set_title("完整场景深度（米）")
  axes[2,col].imshow(wv["rgb"]);axes[2,col].set_title("手腕相机 RGB")
  for row in range(3):axes[row,col].axis("off")
 fig.colorbar(im,ax=axes[1,:].tolist(),fraction=.01,pad=.01,label="米")
 fig.suptitle(event["name_zh"]+" / "+episode+"：原始示范在同一时间轴上的 RGB-D\n这些是模型将看到的原始观测；下方还可播放完整示范。",fontsize=17)
 path=dest/(event["id"]+"_RGBD示范.png");fig.savefig(path,dpi=110);plt.close(fig)
 t=np.arange(len(records));z=np.array([r["object_position"][2] for r in records]);gz=np.array([r["gripper_position"][2] for r in records]);bilateral=np.array([len({c["side"] for c in r["contacts"]})==2 for r in records])
 fig,ax=plt.subplots(figsize=(14,3.2),layout="constrained")
 ax.plot(t,(z-z[event["start"]])*100,color="#1478c3",linewidth=2,label="物体相对首次接触的高度（cm）")
 ax.plot(t,(gz-gz[event["start"]])*100,color="#7b5cb4",alpha=.75,label="夹爪相对首次接触的高度（cm）")
 ax.fill_between(t,-4,30,where=bilateral,color="#e9812f",alpha=.12,step="mid",label="两指同时接触")
 ax.axvline(event["start"],color="#e9812f",linestyle="--")
 ax.axhline(5,color="#636d79",linestyle=":",linewidth=1)
 ax.set(xlabel="示范帧",ylabel="相对高度（cm）",ylim=(-4,max(30,(z-z[event["start"]]).max()*100+5)),title="动作与结果：接触、抬升、保持")
 ax.grid(alpha=.15);ax.legend(loc="upper left",ncol=3,fontsize=10)
 trace=dest/(event["id"]+"_动作轨迹.png");fig.savefig(trace,dpi=110);plt.close(fig)
 rowsout.append({"id":event["id"],"name":event["name_zh"],"episode":episode,"frames":len(records),"keyframes":frames,"lift_cm":round(event["lift_m"]*100,1),"drift_mm":round(event["drift_m"]*1000,1),"rgbd_figure":path.name,"trajectory_figure":trace.name,"demonstration_gif":"../"+event["id"]+"/demonstration.gif","full_rgbd":event.get("full_rgbd_path")})
 print("DEMO_REVIEW",event["id"],flush=True)
(dest/"index.json").write_text(json.dumps(rowsout,ensure_ascii=False,indent=2))
print("DONE",len(rowsout),flush=True)
