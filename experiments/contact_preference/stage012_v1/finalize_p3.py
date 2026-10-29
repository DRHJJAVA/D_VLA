from pathlib import Path
import json,hashlib,re
from datetime import datetime,timezone,timedelta
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/stage012_v1'
doc=R/'docs/夹爪物体接触偏好学习_完整方案.md'
s=doc.read_text()
if '### P3｜' in s:raise RuntimeError('P3_already_present')
a=json.loads((O/'stage0_audit.json').read_text());b=json.loads((O/'stage1_audit.json').read_text());c=json.loads((O/'stage2_audit.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
ts=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
changes=[
('- 最后修订时间：2026-10-25T01:56:17+08:00','- 最后修订时间：'+ts),
('- 方案版本：P2，待实施设计；补充阶段 0–8 的操作、交付、验收和失败处理。','- 方案版本：P3；阶段 0–2 已按本方案执行并记录限定范围验收，阶段 3–8 未执行。'),
('- 当前状态：**仅完成方案文档；未按本方案运行数据处理、训练、推理或测试。**','- 当前状态：**阶段 0–1 的既有数据核查限定范围通过；阶段 2 的夹取表面来源可复用，但跨任务遮挡面和去重面积尚未通过总体验收。未启动本方案的接触证据训练、评分网络或保留测试。**'),
('**当前所有阶段均为未执行。下面是未来执行的验收约定，不是已经获得的实验结果。** 本次只更新这份方案；仍遵守“暂时别跑”。后续获得执行授权后，常规数据检查、实现修复及已授权范围内的对照由我完成，不要求用户每一步重复批准。最终真实模型效果由用户审核；受保护测试数据仍须按既有约定另行明确范围。','**以下是阶段验收约定；实际执行与结论见本文件 P3。** 用户已授权阶段 0–2 并明确要求完成后写回同一文档；阶段 3–8 仍未执行。最终真实模型效果由用户审核；受保护测试数据仍须按既有约定另行明确范围。'),
('| 0 数据与评估边界 | 未执行 | 清单/分组/隔离检查 | 未决定 |','| 0 数据与评估边界 | 限定范围通过，'+ts+' | 既有 30 段划分与 3 段跨任务开发案例，见 P3 | 可进入已核对数据的阶段 1；新形状/保留测试不在此次结论内 |'),
('| 1 当前几何与时间 | 未执行 | 校准误差、固定投影核查 | 未决定 |','| 1 当前几何与时间 | 限定范围通过，'+ts+' | 3 段跨任务回放及 1 段夹取来源，见 P3 | 可用于已核对帧的阶段 2；其余数据须逐段核对 |'),
('| 2 遮挡表面与面积 | 未执行 | 独立留出定位、来源和面积去重 | 未决定 |','| 2 遮挡表面与面积 | 已执行，整体未通过，'+ts+' | 3 段夹取复用、抽屉局部、开关/多物体未知、面积敏感性，见 P3 | 仅保留有来源的局部候选；不进入阶段 3 的跨任务接触训练 |')]
for old,new in changes:
 if s.count(old)!=1:raise RuntimeError('document_anchor_count_'+str(s.count(old))+'_'+old[:25])
 s=s.replace(old,new)
rows=[]
for fn in ['stage01_audit_v2.py','stage1_crossview_addendum.py','stage2_reuse_localize.py','stage2_area_addendum.py','build_gallery.py']:
 p=O/fn;rows.append('| '+fn+' | '+sha(p)+' |')
cases=b['cases'];qmax=max(v['q_t_vs_raw_obs_t_minus_1_max_abs'] for v in cases.values())
off=max(v['ee_to_hand_offset_deviation_m']['max'] for v in cases.values())*1000
cross={k:v['crossview_agent_to_wrist'] for k,v in cases.items()}
d=c['cross_task_localization']['drawer_demo_0'];st=c['cross_task_localization']['stove_demo_0'];multi=c['cross_task_localization']['two_objects_demo_0']
gallery=R/'docs/接触偏好_阶段012_案例核查.html'
add='''

### P3｜'''+ts+'''｜阶段 0–2 实际执行、案例图与验收

**范围与产物。** 本轮只处理方案的阶段 0、1、2。机器审计与脚本保留在远程 `experiments/contact_preference/stage012_v1/`；复用既有 `iter_024` / `iter_025` 的夹取表面来源和 `cf017` 已回放的跨任务 RGB-D，不重跑旧追踪、不复制完整视频。可查看的六个案例在 [阶段 0–2 表面定位核查](接触偏好_阶段012_案例核查.html)。这些图是来源与局部配准检查，**不是最终评分模型的输出，也不是物理接触真值**。本轮没有新接触标签、没有评分网络训练、没有访问预留 demo40–44 内容或新形状盲测。

**阶段 0：数据边界，限定范围通过。** 既有 `stage2_episode_split.json`（SHA256 `'''+a['split_sha256']+'''`）按完整 episode 分为训练 '''+str(a['split_counts']['train'])+'''、旧验证 '''+str(a['split_counts']['val'])+'''、历史测试 '''+str(a['split_counts']['test'])+'''，''' + str(a['unique_count']) + ''' 个 ID 唯一，三组交集均为空。其选择偏向旧 pick/place，不代表自然接触频率。三段已看过的跨任务开发视频分别为拉开抽屉 '''+str(a['case_inventory']['drawer_demo_0']['frames']['replay_rgbd'])+''' 帧、转炉灶开关 '''+str(a['case_inventory']['stove_demo_0']['frames']['replay_rgbd'])+''' 帧、同视频处理两个物体 '''+str(a['case_inventory']['two_objects_demo_0']['frames']['replay_rgbd'])+''' 帧；它们**不作为独立测试或网络训练成绩**。旧划分同资产跨训练/验证，不能证明新形状、新尺寸泛化；本轮未建立独立新形状保留集。预留盲测内容未打开，未对其具体 ID 与旧划分做交叉确认，相关范围不作通过结论。数据清单及三段原 HDF5／回放／运动文件哈希在 `stage0_audit.json`。

**阶段 1：时间与坐标，限定范围通过。** 三段跨任务回放的长度逐一等于官方 actions/obs 长度；全帧检查 `回放 t 的实测夹指开度 = 官方 obs[t−1]`，最大绝对差 '''+f'{qmax:.3e}'+'''；t0 的手状态保留未知。回放文件的 `frame_time` 每步 0.05 s，但原始 HDF5 没有独立逐帧时钟，本轮只把它称为回放时间元数据。相对于官方 EE 原点，实测 hand 原点在手坐标 z 方向约 −97 mm，三例偏移常量的最大离差 '''+f'{off:.3f}'+''' mm，不能把这个固定几何偏移误当一帧时间错位。三段 `cf017` 初始真深度来源点由像素、深度、K、camera_to_world 独立反投影至已保存世界点，误差 0；回投像素最大 <2×10⁻¹³ px。瓷杯 demo9 的夹前帧 36 共 2742 个查询点，RGB 与该帧逐点完全相同、XYZ 重建最大误差 0。三例手旋转正交和相机齐次矩阵检查见 `stage1_audit.json`。跨视角在 5 mm 内的已重叠点中，agent→wrist 最近邻误差中位数（抽屉/开关/双物体）为 '''+'/'.join(f"{cross[k]['matched_median_mm']:.2f}" for k in ['drawer_demo_0','stove_demo_0','two_objects_demo_0'])+''' mm；未匹配点可能由视野/遮挡造成，不能据此称标定有误。三例夹前当前 XYZ、像素、RGB 在旧区域构建与后续目标构建前后逐数组相同；**未做任意未来帧扰动的反事实测试**，所以因果输入验收限于这三例和现有实现。其它 episode 与物理采样率不在本轮通过范围。

**阶段 2：夹取来源可复用，跨任务与面积总体未通过。** 旧 `iter_024` / `iter_025` 完成锁内的三例经 SHA256 逐文件核对、查询 XYZ/像素相同。瓷杯 demo9 夹前36→持物55有 '''+str(c['grasp_reuse'][0]['positive_query_surface_points'])+''' 个当前稳定正候选；番茄罐 demo7 夹前48→持物73有 '''+str(c['grasp_reuse'][1]['positive_query_surface_points'])+''' 个；白黄杯 demo4 夹前40→持物66有 '''+str(c['grasp_reuse'][2]['positive_query_surface_points'])+''' 个。绿色来自夹前真实点，橙色来自持物时真实深度；它们证明**来源可追溯的候选片**，不证明全部遮挡接触面积或跨任务适用。它们是不同物体/局部位置的展示，不把同事件多帧当作独立抓取。

拉抽屉开发例从第 0 帧真实深度取把手/前缘附近 '''+str(d['initial_true_depth_points'])+''' 个来源像素。原 `cf017` 的宽前景包含静止柜体，整块刚体拟合不可直接复用；本轮仅对局部动态轨迹做确定性 RANSAC（留出轨迹不参与拟合），第 70 帧有 '''+str(d['local_fit']['70']['inliers'])+''' 个拟合内点，局部来源中心移动 '''+f"{d['local_fit']['70']['source_center_displacement_mm']:.2f}"+''' mm；2 个近邻留出轨迹误差 '''+', '.join(f"{v:.2f}" for v in d['local_fit']['70']['heldout_near_errors_mm'])+''' mm。这只支持**第 70 帧局部映射**。第 88 帧 3 个留出中一个误差 '''+f"{max(d['local_fit']['88']['heldout_near_errors_mm']):.2f}"+''' mm，不能延续成整段通过。开关的第 0 帧 '''+str(st['initial_true_depth_points'])+''' 个真实深度点落在可见旋钮近手区域，但第 86 帧该小片仅 '''+str(st['visible_valid_tracked_knob_points_at_reference'])+''' 个可见有效轨迹且只 1 个初始留出轨迹；**被遮挡的旋转接触面位置和面积未知**。双物体宽前景混入其它对象/背景，未通过独立物体身份核查，故没有给两个物体生成接触面真值。当前远程现成回放只有拉开抽屉，没有“推/关闭抽屉”案例；LIBERO 官方存在关闭抽屉示范文件，但本轮远程到数据源的连接失败，未取得也未伪造该画面。

面积验收未通过：初始真实深度局部片按连续像素三角片粗估，抽屉 1px/2px 网格分别 '''+f"{d['initial_source_triangulated_area_mm2']['one_pixel_grid']:.1f}/{d['initial_source_triangulated_area_mm2']['two_pixel_grid']:.1f}"+''' mm²，旋钮 '''+f"{st['initial_source_triangulated_area_mm2']['one_pixel_grid']:.1f}/{st['initial_source_triangulated_area_mm2']['two_pixel_grid']:.1f}"+''' mm²；变化过大，不能作为稳定的唯一面积，更不能叫物理接触面积。重复同一源像素只计一次的逻辑已实行，但跨帧完整面积去重、遮挡受控实验、独立相机/时间块验证仍不足。**结论是阶段 2 已完成本轮核查，但整体效果门未通过；只保留上述有来源的局部候选，未知维持未知，不进入阶段 3 的跨任务接触证据训练。** 下一轮若继续，应优先补旋钮/抽屉的局部身份追踪与真正留出视角、改稳定面积估计，再用真实推抽屉视频按同样冻结流程核对；不得靠动作类别、抬升或整物体随手假设硬填目标。

**实现哈希与隔离。** 以下是本轮新增脚本的 SHA256；首版阶段 0/1 脚本仅因 `cf017` 产物实际位于 `artifacts/` 而在读取阶段停止，v2 只修路径，未更改数据或阈值。原失败脚本保留。
| 远程脚本 | SHA256 |
|---|---|
'''+ '\n'.join(rows)+'''

审计文件为 `stage0_audit.json`、`stage1_audit.json`、`stage2_audit.json`；案例页及 7 张本轮图的哈希记录在 `completed_locked.json`。所有数据、图、源码、报告仅在远程；原始示范只读。P3 不改变 P0–P2 的设计正文，以上结果替代第 15.12 节对应的旧“未执行”状态。阶段 3–8 仍未执行，最终真实预测必须由用户审核。
'''
s+=add
doc.write_text(s)
out={'created_beijing':ts,'stage0':'limited_pass_existing_episode_split_only','stage1':'limited_pass_three_cross_task_one_grasp_geometry','stage2':'executed_overall_not_passed_grasp_reuse_drawer_frame70_local_only_stove_unknown','gallery':str(gallery.relative_to(R)),'stage3_started':False,'blind_reserved_data_read':False}
(O/'stage_state.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
files=[p for p in O.iterdir() if p.is_file() and p.name!='completed_locked.json']
files+=[gallery]+[p for p in (R/'docs/图示/接触偏好_阶段012_正式').iterdir() if p.is_file()]
lock={str(p.relative_to(R)):sha(p) for p in files}
(O/'completed_locked.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2))
print(json.dumps({'timestamp':ts,'document_sha256':sha(doc),'stage0':out['stage0'],'stage1':out['stage1'],'stage2':out['stage2'],'locked_files':len(lock),'gallery_sha256':sha(gallery)},ensure_ascii=False))
