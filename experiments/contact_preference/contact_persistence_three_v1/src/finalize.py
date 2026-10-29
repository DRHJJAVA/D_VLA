from pathlib import Path
import json,datetime,hashlib
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/contact_preference/contact_persistence_three_v1'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as z:
  for b in iter(lambda:z.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
report=f"""
## {now}｜三段完整视频：两根夹指接触面与稳定延续实验结果

**范围确认**：用户明确只计算两根夹指；掌部外壳、腕部、机械臂和其他物体之间的接触都不纳入。当前估计原本只使用两根夹指几何（ids 2、5），本次范围确认不需要重算。掌部附近的观测不能改名为夹指接触。

本轮目录：experiments/contact_preference/contact_persistence_three_v1。网页：[三个完整场景逐帧接触面](接触偏好_三个完整场景逐帧接触面.html)。只使用 agentview 与 robot0_eye_in_hand。没有训练评分网络，没有使用人工区间或仿真物体位姿/接触真值，没有读取预留测试。新推关抽屉 demo0 属于开发案例，不是盲测。

### 实际执行与方法
三段完整 episode 分别为瓷杯 demo9（101 帧，0–100）、炉灶开关 demo0（115 帧，0–114）、推关上层抽屉 demo0（65 帧，0–64），共 281 帧、562 张双视角图；20 Hz，完整视频时长为 5.05/5.75/3.25 秒。不是只选接触窗口；无标记的帧也可逐帧查看。
每段来源帧固定为 1、N//3、2N//3，双视角真实深度反投影；冻结 CoTracker 追踪与独立刚体定位，保留真实源像素。完整视频可供监督侧使用，允许后出现的真实表面回溯，不能描述为实时因果预测。
表面接近两根夹指且法向、深度/夹指遮挡、留一稳定性与独立留出点支持时建立几何候选。后续只有独立物体定位仍可靠，已建立点与夹指的相对位置变化及定位不确定性合计不超过 2 mm、开度变化不超过 1 mm、没有自由空间冲突且没有明确分离，才延续。延续不替代物体独立定位，不靠夹爪位姿直接搬运未知表面。
本轮仍是预登记几何证据实验，不是学得的接触概率。几何与延续参数未按人工区间调节。移动部件发现用来找物体对应，可能漏掉一直静止的接触部件；不能当通用接触检测器。

### 逐案例结果
| 案例 | 有表面支持的帧 | 主要结论 |
|---|---|---|
| 瓷杯 demo9 | 48、53–76、95 | 运输中可延续一小片表面；66、69–76、95 含被夹指遮挡的支持。不是完整接触面。 |
| 开关 demo0 | 70–74 | 73–74 有延续，70–71 有遮挡支持；其余帧不足，保留未知。 |
| 推关抽屉 demo0 | 暂无 | 未建立可靠的两根夹指接触面；当前近处观测主要靠掌部，排除。不能据此断言没有真实夹指接触。 |

瓷杯的有效表面来源是第 67 帧（已经夹持），第 1/33 帧未建立合格近接触来源。因此本轮并未证明已经从夹持前恢复全部被遮挡表面。搬运时常只有约 7 个真实源代表格的小条带，显示三角片也只连接同一实际来源的合格相邻点，不填补没有观测的面积。
接触保持对照：相比仅用当前几何，瓷杯额外保留 22 个“源点×帧”观测，开关额外 3 个，抽屉 0；分别分布于 12、2、0 帧。两种方法的“至少一个支持点”的整帧区间相同，**没有新增整帧接触覆盖**。这表明能补保部分点，但不能宣称已解决整段遮挡。
按同一真实源点累计，瓷杯最大 1.30 秒、开关 0.20 秒；是有支持帧数/20，未知空档不计，不代表连续时长、完整真实接触时长，也不跨相机重复相加。代表格数不是实际面积。

推抽屉固定检查 25/30/33/35/40/45/50 帧，主相机有 28/30/23/54/64/83/64 个掌部 3 mm 邻近观测，但两指 3 mm 邻近观测均为 0。这里只说明这批可见点的几何位置，不是物理接触真值，也不能将掌部点充当两指表面。后续若继续，应从指侧真实表面可观测性与独立对应入手，而不是放宽到掌部。

### 验证、修复与验收结论
281 帧来源/变换/支持条件核查完成：真实源 XYZ 误差 0；重投影最大误差 3.41e-13 像素；独立变换最大误差 2.37e-16 m；不满足保存支持条件的点为 0；7 项原输入哈希无变化。
时间对齐独立检查支持回放 t 对应 obs(t−1)；末端位置与 hand 原点有固定约 96.95 mm 偏移，不能将两原点强行等同。t0 手状态未知。
修复记录：初次渲染导入路径、状态 float64/float32 精度断言、行向量/列向量旋转约定均已记录。最终分析为 src/analyze_v3.py，最终结果为 estimates_v2、summary_review.json；旧 estimates 保留但无效，不用于网页或结论。没有通过放宽误差阈值修复。
浏览核查瓷杯 70、开关 73/74、抽屉 40 帧双视角/点云/来源，以及逐帧播放、帧选择、几何对照。抽屉完整 MP4 浏览器解码正常，896×448、3.25 秒、无媒体错误。网页明确仅两根夹指，绿色建立/更新、粉色延续、橙色未知；源图只表示真实观测位置，不表示来源时刻已接触。
**工程核查通过；接触面质量和覆盖验收仍待用户查看。** 瓷杯部分延续可用，开关覆盖稀疏，抽屉未定位，不能宣布三类接触面都已解决。本轮不启动评分网络训练。下一步应按这些缺口决定实验，不将“未知”改成正或负。
"""
p=R/'docs/夹爪物体接触偏好学习_完整方案.md'
with p.open('a') as z:z.write(report)
with (R/'docs/STATUS.md').open('a') as z:z.write('\n'+now+' 三完整场景逐帧接触面实验已完成；仅两根夹指；瓷杯部分延续、开关稀疏、推抽屉未知。结果写回完整方案，评分网络未训练。\n')
dump(D/'audits/browser_review.json',{'time':now,'frames_viewed':{'mug':[70],'switch':[73,74],'push':[40]},'fullpage_visual_review':True,'previous_mug_review_retained':True,'tested':['frame input','next frame','case selection','geometry only toggle','sequential frame playback'],'push_mp4':{'duration':3.25,'width':896,'height':448,'readyState':4,'error':None},'scope':'two fingers only','limitations':['small partial surface not complete area','no reliable push finger surface','source caption cleared when no source']})
m=[]
for name in ['source_locked.json','input_locked.json']:
 lock=json.loads((D/name).read_text())
 for path,digest in lock.items():
  if isinstance(digest,str) and len(digest)==64:
   pp=Path(path)
   if not pp.is_absolute():pp=R/path
   if not pp.exists() or sha(pp)!=digest:m.append(str(pp))
assert not m
dump(D/'audits/final_integrity.json',{'time':now,'initial_lock_mismatches':m,'canonical_analysis':'src/analyze_v3.py','canonical_estimates':'data/*/estimates_v2','invalid_retained':'data/*/estimates','scope_confirmation':'scope_confirmation.json','model_training':False,'reserved_test_access':False,'document':str(p),'document_sha256_at_update':sha(p)})
dump(D/'stage_state.json',{'stage':'complete','time':now,'acceptance':'engineering checked; surface reliability and coverage require user review','scoring_training':False})
files={}
for p in sorted(D.rglob('*')):
 if p.is_file() and not p.is_symlink() and p.name!='completed_locked.json' and p.suffix not in ['.log','.pyc'] and '__pycache__' not in p.parts:
  files[str(p.relative_to(D))]=sha(p)
page=R/'docs/接触偏好_三个完整场景逐帧接触面.html'
dump(D/'completed_locked.json',{'time':now,'files':files,'review_page':{'path':str(page),'sha256':sha(page)},'logs_excluded':True})
print(json.dumps({'time':now,'stage':'complete','locked_files':len(files),'mismatches':m},ensure_ascii=False))
