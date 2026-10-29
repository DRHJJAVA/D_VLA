from common import *
import numpy as np,re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def main():
 summary=json.loads((RUN/"artifacts/model_summary.json").read_text());prep=json.loads((RUN/"artifacts/preparation_summary.json").read_text());audit=json.loads((RUN/"artifacts/verification.json").read_text())
 z=np.load(RUN/"artifacts/dataset.npz");p=np.load(RUN/"artifacts/frame_predictions.npz");n=len(p["mean"]);x=np.arange(n)
 fig,ax=plt.subplots(3,1,figsize=(13,8),sharex=True,gridspec_kw={"height_ratios":[3,2,1]})
 ax[0].plot(x,p["mean"],label="network ensemble mean");ax[0].fill_between(x,p["low"],p["high"],alpha=.25,label="initialization range")
 known=z["weight"]>0;ax[0].scatter(x[known],z["target"][known],s=9,c="gray",label="geometric weak target (not truth)")
 ax[0].set(ylabel="compatibility score",ylim=(-.05,1.05));ax[0].legend(loc="upper left",fontsize=8)
 ax[1].plot(x,z["distance"]*1000,label="observed-source distance");ax[1].plot(x,z["sigma"]*1000,label="uncertainty scale")
 ax[1].set(ylabel="millimeters");ax[1].legend(fontsize=8)
 categories=np.where(p["contact_candidate"],2,np.where(p["observed_separation_candidate"],0,1))
 ax[2].imshow(categories[None],aspect="auto",interpolation="nearest",extent=[-.5,n-.5,0,1],cmap=matplotlib.colors.ListedColormap(["#5395c8","#a9a9a9","#e4ad35"]),vmin=0,vmax=2)
 ax[2].set(xlabel="RGB-D frame index",yticks=[],ylabel="blue=separation\ngray=unknown\nyellow=candidate")
 fig.suptitle("CF001: one-video weakly supervised contact evidence; no contact ground truth")
 fig.tight_layout()
 plot=ROOT/"docs/图示/contact_frames_cf001_timeline.png";plot.parent.mkdir(exist_ok=True);fig.savefig(plot,dpi=160);plt.close(fig)
 stamp=now();text=DOC.read_text()
 text=re.sub(r"- 最近修订时间：.*","- 最近修订时间："+stamp+"。",text)
 text=text.replace("- Verification Status：UNVERIFIED；本文件是方案，没有实验结果。","- Verification Status：CF001已执行并完成数值核对；真实接触判定与独立视频泛化未验证。")
 def ranges(v):return "、".join(str(r["start_frame"])+"–"+str(r["end_frame"]) for r in v) or "无"
 addition="\n### 14.8 CF001实际结果（"+stamp+"）\n\n"
 addition+="数据："+prep["episode"]+"，"+str(prep["frames"])+"帧；"+str(prep["tracks"])+"个真实初始追踪来源，"+str(prep["reference_real_surface_sources"])+"个1mm去重的真实参考表面点。手状态t0未知。视觉变换可拟合"+str(prep["fit_frames"])+"帧，弱目标可加权"+str(prep["weighted_frames"])+"帧，满足展示质量要求"+str(prep["reliable_frames"])+"帧。\n\n"
 addition+="新网络参数量："+str(summary["runs"][0]["parameters"])+"；三个seed各300步，无旧区域网络权重、无成功/held标签、无抬升或最短持续门槛。\n\n"
 addition+="| seed | 训练加权BCE/全部帧 | 同视频留出加权BCE/全部帧 | 训练动力学MAE(m) |\n| --- | --- | --- | --- |\n"
 for row in summary["runs"]:
  m=row["metrics"];addition+="| "+str(row["seed"])+" | "+format(m["train"]["weighted_bce_per_all_frames"],".6f")+" | "+format(m["within_video_audit"]["weighted_bce_per_all_frames"],".6f")+" | "+str(m["train_dynamics_mae_m"])+" |\n"
 addition+="\n接触候选帧："+ranges(summary["candidate_contact"])+"。\n\n已见表面分离候选帧："+ranges(summary["candidate_observed_separation"])+"。\n\n未知帧："+ranges(summary["unknown"])+"。区间均包含首尾帧。候选不是已证实成功接触；分离只针对已见表面，不能排除从未观察面的接触。\n\n"
 addition+="三个网络完整重载后逐帧分数最大差："+str(audit["prediction_reload_max_abs_difference"])+"；输入/源码hash核对通过，未知分类损失梯度为0。没有真实接触标注，因此不能计算接触准确率或失败排除率。辅助预测目标为视觉估计，不是物理真值。\n\n"
 addition+="![CF001逐帧证据与未知区间](图示/contact_frames_cf001_timeline.png)\n\n"
 addition+="源码与机器产物：experiments/contact_frames/cf001/；逐帧CSV：artifacts/frame_predictions.csv；权重：artifacts/seed_2026、seed_2027、seed_2028/last.pt。源码、输入、权重哈希见source_locked.json、input_locked.json、artifacts/model_summary.json与completed_locked.json。\n\n"
 addition+="限制与下一步：单视频拟合不能证明新视频能力；模型主要学习可见来源的几何弱兼容度，尚未解决几何近邻与真实力学接触的歧义，不能据此自动将全部候选加入正式区域监督。下一迭代须在本文件继续登记独立视频与更强证据检查，不用更多轮数掩盖语义不足。\n"
 assert "### 14.8 CF001实际结果" not in text
 DOC.write_text(text+addition)
 lock={str(f):sha(f) for f in RUN.rglob("*") if f.is_file() and f.suffix not in [".log",".pyc"] and f.name not in ["stage_state.json","completed_locked.json"] and "__pycache__" not in str(f)}
 save(RUN/"completed_locked.json",lock)
 stage("complete",counts=summary["counts"],document=str(DOC),contact_accuracy_unverified=True)
if __name__=="__main__":
 try:main()
 except Exception as e:fail(e);raise SystemExit(1)
