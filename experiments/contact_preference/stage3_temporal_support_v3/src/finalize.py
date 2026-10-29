from pathlib import Path
import json,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/stage3_temporal_support_v3';doc=R/'docs/夹爪物体接触偏好学习_完整方案.md';page=R/'docs/接触偏好_阶段3_遮挡期间支持保持对照.html'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
s=json.loads((O/'summary.json').read_text());a=json.loads((O/'audits/independent.json').read_text());wa=json.loads((O/'audits/webdata_and_projection.json').read_text());va=json.loads((O/'audits/visibility.json').read_text())
assert a['all_passed'] and wa['all_passed'] and va['all_passed']
browser={'time':now,'tab':113,'all_12_frame_counts_checked':True,'main_wrist_and_color_controls_checked':True,'final_visual_checks':['whiteyellow demo4 frame68 main: cloud follows cup wall/rim after camera correction','tomato demo7 frame70 wrist: cloud and occluded green strip project at can/fingers','porcelain support source and temporal table inspected'],'final_projection_audit':'audits/webdata_and_projection.json','not_physical_contact_ground_truth':True}
(O/'audits/browser_review.json').write_text(json.dumps(browser,ensure_ascii=False,indent=2))
hashes={str(p.relative_to(O)):sha(p) for p in sorted((O/'src').glob('*.py'))}
# Record dependencies as postread inventory without pretending they were all prelocked.
deps=[R/'experiments/contact_preference/stage3_pilot_v1/src/build_supports_v2.py',R/'experiments/contact_preference/stage3_pilot_v2/src/build_supports.py',R/'src/v3/stage2_robot_only_depth.py']
(O/'dependency_postread_locked.json').write_text(json.dumps({str(p):sha(p) for p in deps},ensure_ascii=False,indent=2))
text=['','## '+now+'｜阶段3 v3：遮挡期间正支持保持与夹爪专属范围核查结果','',
'输出：experiments/contact_preference/stage3_temporal_support_v3。网页：[阶段3遮挡期间支持保持对照](接触偏好_阶段3_遮挡期间支持保持对照.html)。本轮已完成小规模计算、独立复算、网页核查；未训练评分网络，未重新拟合证据模型，未读取人工区间作监督，未访问预留测试。旧版及全部原结果保留。','',
'### 具体执行与对照','',
'用原stage3_pilot_v2/points_evidence作旧版对照，固定3事件原查询点集2742/1183/3720点和原12个端点；没有用多视角新增点改变比较分母。原各点坐标、RGB、像素、来源索引、对应映射逐项相同。当前双相机标定来自各帧H5，回放t对应obs t−1。',
'沿用旧版留一事件训练的分离密度族（指数/半正态与base/proxy误差），不使用混合类别先验作为接触概率。每点先取得跨模型最保守的单时刻几何正/负超额支持。活动夹指/垫面包含外侧指面；固定外壳归属、深穿透、不相对法线等不提供正支持，未使用必须抬升、固定任务动作类别或两指间唯一范围。',
'相邻端点运动比较：基于保存的RGB-D拟合映射，计算随手输运与世界静止的三维残差；噪声方差是两时刻原测量误差平方和。运动正支持=max(tanh((静止残差平方−随手残差平方)/(4方差)),0)，base/proxy取最小值。它是运输段的运动解释证据；同一刚体变换映射出的很多点不算独立运动测量，也不能单独证明物理接触。',
'正支持seed=min(当前几何正支持,当前运动正支持)；carry=min(上一端点正支持,当前几何正支持,相邻手坐标连续性)，P=max(seed,carry)。N为保守分离支持，U=1−P−N。重复帧不累加，也不按样本数缩小误差。第一持物端点无前序持物比较，P=0；对几何明确分离点仍可有负支持，接触附近未知。该设计会牺牲起始端点召回，不能宣称已解决接触起止帧识别。',
'当前可见与机器人遮挡使用同一支持公式。由双相机深度与已知机器人渲染深度分开：可见、机器人遮挡、其他/来源不明遮挡、前方自由空间冲突、视野外或深度缺失。无可见视角且遮挡来源不明、任一视角自由空间冲突等转未知。机器人遮挡是几何代理证据，不是接触传感真值。',
'显示类别为P/N/U最大项（并列未知）。外壳归属单独显示灰色排除，底层仍未知；下表未知列不重复包含灰色排除。连续量并非校准概率。','',
'### 逐端点结果','',
'|事件/帧|旧版正支持|新版正|新版负|新版未知|灰色排除|新版正中可见|新版正中机器人遮挡|',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for c in s['cases']:
 for row in c['rows']:
  ct=row['new_counts']
  text.append(f"|{c['title']} {row['frame']}|{row['baseline_counts'][0]}|{ct[0]}|{ct[1]}|{ct[2]}|{ct[3]}|{row['green_visible']}|{row['green_robot_occluded']}|")
text+=['','从第二个持物端点起，机器人遮挡下保持绿色的点为瓷杯21–30、罐头19–21、白黄杯3–7。相邻检查端点间，同一来源点从前一端点绿色延续到当前机器人遮挡绿色的数量：瓷杯0/21/21/28，罐头0/20/19，白黄杯0/4；首个0对应上一端点尚未初始化。以上只证明已检查端点的支持连续性，不证明间隔中的每一帧接触，也不是独立事件成功率。','',
'旧版部分绿色点的遮挡来源并不能由机器人解释：瓷杯每帧0–8、罐头21–23、白黄杯0–4，此轮转未知。这不是这些点实际接触失败的标签。三例所有外壳归属、深穿透、当前前方冲突、其他/来源不明遮挡点的新版正支持均为0。该检查证明实现遵守证据范围，不证明已排尽全部物理误接触；原目标前景与表面身份仍可能有错误。',
'历史保持消融：只有瓷杯66帧17个点的正支持比seed更高，其中2点改变为绿色；其余11端点没有额外提升。故本轮主要收益是把遮挡与冲突分开，并增加运动证据约束，不能将全体运输支持归功于历史保持，不能宣称全面提高准确率。','',
'### 验收','',
f"- 来源点集与既有结果相同；独立用密度公式和高斯似然比重算正负未知、运动及carry，最大差{a['independent_math_max_error']:.3g}。21输入文件、2初始源码/配置锁以及旧版59文件锁核对未变。",
'- 全12端点、24相机投影用H5原深度与重新渲染的已知机器人深度独立重算，遮挡类别逐点一致；外壳/穿透/非相对法线/前方冲突/其他遮挡的正支持检查通过。密度数值对照中10σ间隔无正支持且负支持接近1，非物理精度测试。',
f"- 页面24组实际相机标定逐项一致，显式逆矩阵与独立线性求解的最大三维差{wa['projection_inverse_max_m']:.3g}m。显示坐标取整误差≤5.01e−9m，支持量≤5.01e−7，30图像地址存在。三案例切换与12帧计数已核对，demo4主相机和罐头腕相机实际叠加已复查。",
'- 页面可切换旧版/本轮单时刻几何消融，勾选负支持或未知；点击同一表面点查看原像素及全部端点P/N/U、运动与历史保持。绿色饱和显示，不按低分透明度隐藏。',
'- 初版网页误用融合参考坐标相机适配参数，已保存初版数据并改用当前真实标定。此前罐头腕视角空投影诊断由该显示错误造成，不是实际视野外结论；最终独立检查该例1183点均在腕相机画面内。空投影保护逻辑仍保留。此修复只影响显示，不改变原支持/遮挡计算。',
'',
'验收结论：本轮小规模处理与工程核查完成，已有表面在被机器人遮挡的检查端点可保留弱正支持；未建立物理接触准确率结论，不自动通过最终阶段验收。当前首端点未知、原对应身份、毫米级近而未接触不可辨识、端点稀疏仍是限制。没有把稀疏端点间隔累加成接触时长训练目标。若要继续，应优先核查起始/释放附近连续帧，而不是直接扩大评分网络；该扩展本轮没有执行。','',
'本轮代码SHA256：','']
for k,h in hashes.items():text.append(f'- {k}：{h}')
with doc.open('a',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
state={'time':now,'status':'complete','phase':'stage3_temporal_support_pilot','cases':3,'endpoints':12,'scoring_network_trained':False,'new_evidence_model_fit':False,'engineering_audit':'passed','physical_contact_acceptance':'pending_user_review_not_established','page':str(page),'document':str(doc)}
(O/'stage_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
(O/'audits/document_update.json').write_text(json.dumps({'time':now,'path':str(doc),'sha256':sha(doc),'same_document_append':True},ensure_ascii=False,indent=2))
files=[p for p in O.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.log','.pid','.pyc'] and p.name not in ['completed_locked.json','final_integrity.json']]
files.append(page);lock={str(p):sha(p) for p in files};(O/'completed_locked.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2))
bad=[p for p,h in lock.items() if sha(p)!=h];assert not bad
(O/'audits/final_integrity.json').write_text(json.dumps({'time':now,'files':len(lock),'mismatches':bad,'logs_and_mutable_main_document_not_in_completion_lock':True},ensure_ascii=False,indent=2))
print(json.dumps({'time':now,'files_locked':len(lock),'state':state},ensure_ascii=False))
