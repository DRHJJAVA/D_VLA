from pathlib import Path
import json,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
r=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_preference/multiview_temporal_fusion_v1');root=r.parents[2]
now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
s=json.loads((r/'fusion_review_summary.json').read_text())
a=json.loads((r/'audits/completion.json').read_text())
w=json.loads((r/'audits/web_and_geometry.json').read_text())
assert w['all_passed']
browser={'time':now,'tab':112,'checked_frames':{c['key']:[v['frame'] for v in c['contact_vicinity_review']] for c in s['cases']},'all_12_frame_counts_equal_saved_metrics':True,'visual_checks':['porcelain55 main before/after and actual wrist source pixel','tomato68 main all new surface','whiteyellow66 main before/after','whiteyellow66 wrist with unknown toggle'],'limitations':'visual checks do not establish per-point physical identity or true contact','image_urls_verified':30}
(r/'audits/browser_review.json').write_text(json.dumps(browser,ensure_ascii=False,indent=2))
(r/'audits/resolved_implementation.json').write_text(json.dumps({'time':now,'issues':[{'issue':'geometry distance returns tuple','resolution':'v2 explicitly unpacks distance and closest; final web geometry checks passed'},{'issue':'display builder repeatedly decompresses arrays','resolution':'v2 caches arrays once; only own first builder stopped; no fusion recomputation'},{'issue':'summary output inherited old webdata directory','resolution':'exact summary bytes copied into webdata_v2 and path fix recorded'},{'issue':'case switch temporarily retained old frame/source UI','resolution':'loading state clears frames/canvases/source, all three case switches checked'}]},ensure_ascii=False,indent=2))
hashes={str(p.relative_to(r)):sha(p) for p in sorted((r/'src').glob('*.py'))}
text=['','## '+now+'｜腕部与固定相机跨时点云融合：三案例小规模结果','',
'本轮响应腕相机/固定相机跨时配准融合请求，只处理既有训练案例，不训练评分网络，不生成新的接触正负标签。复用既有RGB-D对应得到的每帧刚体映射和全部留一变换，没有用模拟器物体位姿或接触真值，也没有新跑跟踪器或强行ICP配准。输出目录：experiments/contact_preference/multiview_temporal_fusion_v1。',
'',
'### 做了什么与为什么','',
'1. 使用瓷杯demo9（参考36，持物55/57/60/65/66）、番茄酱demo7（参考48，持物68/70/73/78）、白黄杯demo4（参考40，持物66/68/71），共15个时刻、30个主/腕视图。回放t对应官方obs t−1；本轮没有t0。它们是3个事件，不能按帧数计成独立抓取。',
'2. 每个相机按自身内外参反投影实际深度，持物时刻用W=当前实测H×当前Q的逆映射到参考世界坐标；参考时刻恒等映射。保存真实像素、RGB、相机、时刻和来源索引。1mm网格只选真实源点，不插值、不平均生成表面。',
'3. 用已知机器人几何重渲染深度，要求源点在机器人深度前至少2mm；物体身份用原查询前景锚点、当前深度表面连通、外观一致性、两尺度朝向法线及至少两个不同帧的空间支持共同约束。两相机同一帧不算两帧。候选离已知参考表面超过3mm，或留一位移超过2mm，均不接受为支持表面。无法区分另一物体、杯内外壁、边界或配准不稳的来源留未知。',
'4. 本轮身份限制较保守，支持点均在参考表面3mm内，因此主要验证采样补充和局部表面扩展，并未实现大面积隐藏面的重建。法线/颜色/几何一致是证据，不是物体身份真值；不能保证所有其他物体混入都已排除。',
'5. 展示夹指附近候选仅使用活动夹指/指垫表面3mm邻域，排除夹爪外壳更近和机器人碰撞几何内部点；若映射点在当前任一相机测量表面前方超过3mm，则该时刻转未知。两相机都位于测量表面之后仅表示可能被遮挡，不证明接触。橙色不是训练正支持或模型预测。',
'',
'### 结果','',
'下表数量是实际源点代表的1mm单元，不能换算成真实接触面积；毫米级配准抖动也可能增加单元数。',
'',
'|案例|原参考单元|融合支持单元|其中主相机单独未支持的单元|未知单元|',
'|---|---:|---:|---:|---:|']
for c in s['cases']:
 text.append(f"|{c['key']}|{c['reference_voxels']}|{c['supported_voxels']}|{c['wrist_contributed_beyond_main_supported_voxels']}|{c['unknown_voxels']}|")
text+=['','|案例/持物帧|融合夹指附近候选（去冲突）|新增夹指附近单元|腕相机额外夹指附近单元|腕相机增益且双视图后方|前方冲突转未知|','|---|---:|---:|---:|---:|---:|']
for c in s['cases']:
 for v in c['contact_vicinity_review']:
  text.append(f"|{c['title']} {v['frame']}|{v['near_no_front_conflict_cells']}|{v['new_near_cells']}|{v['wrist_extra_near_cells']}|{v['wrist_extra_near_behind_both']}|{v['near_front_conflict_unknown']}|")
text+=['','瓷杯每个检查时刻有20–24个腕相机额外夹指附近单元，其中8–13个位于当前两相机测量表面后方，提供了补充遮挡区域的候选证据。罐头与白黄杯的整体表面有腕相机贡献，但检查时刻夹指附近额外贡献为0；这两例该处增益主要来自主相机跨时观测。不能把整体点云变密宣传为腕相机补出了所有接触面。',
'',
'### 验收与显示','',
'工程核查通过：35个输入文件和初始/附录源码锁、旧阶段3 v2完成锁核对；所有实际源像素/深度独立重投影、世界/参考坐标映射及机器人排除检查通过；支持代表的不同帧见证、法线、外观、留一稳定性和真实代表索引通过。页面序列化坐标与原结果最大差约5e−8m，30张图像地址存在，全部12时刻橙色点满足保存几何与当前深度冲突检查。浏览器核查三案例切换、各帧计数、demo4主/腕同视角和未知显示。',
'',
'网页：[双相机跨时融合与遮挡区域](接触偏好_双相机跨时融合与遮挡区域.html)。左侧原参考表面、右侧融合表面；蓝色新增支持表面，橙色夹指附近候选，紫色未知默认隐藏。可切换当前主/腕视角，点击融合点追溯实际来源帧、像素、相机和不同帧见证。未知显示每例均匀抽样最多3000个代表，但完整未知数组与计数保留。投影在当前夹爪前景上的历史物体点可能是遮挡关系，不表示把夹爪点融成物体。',
'',
'结论：本次小规模融合与来源审计已完成，可供人工检查。对瓷杯发现局部遮挡补充价值；其余两例未证明腕相机对夹指附近有增益。物理接触与逐点身份未验收，不能将橙色自动变成正标签。保持未知；本轮没有评分网络训练，也没有访问预留测试。下一步如继续阶段3，应在这些融合表面上评估正/负支持与未知的变化，不能直接以网格增加替代接触可信度。',
'',
'实现过程保留记录：几何距离返回元组导致首版统计停止，v2解包后完成；首版网页构建反复解压数组较慢，v2缓存后完成；继承的汇总路径修正为实际webdata_v2。均未修改旧结果，也未放宽监督门槛。详情见audits/resolved_implementation.json。',
'',
'本轮源码SHA256：','']
for k,v in hashes.items():text.append(f'- {k}：{v}')
doc=root/'docs/夹爪物体接触偏好学习_完整方案.md'
with doc.open('a',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
state={'time':now,'stage':'complete_pilot_fusion','status':'complete','cases':3,'held_frames':12,'views':30,'engineering_audit':'passed','physical_contact_acceptance':'not_established','scoring_network_training':False,'document':str(doc),'page':str(root/'docs/接触偏好_双相机跨时融合与遮挡区域.html')}
(r/'stage_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
(r/'audits/document_update.json').write_text(json.dumps({'time':now,'document':str(doc),'sha256_at_completion':sha(doc),'same_document_appended':True},ensure_ascii=False,indent=2))
files=[p for p in r.rglob('*') if p.is_file() and p.suffix not in ['.log','.pid','.pyc'] and '__pycache__' not in p.parts and p.name not in ['completed_locked.json','final_lock_verification.json']]
files.append(root/'docs/接触偏好_双相机跨时融合与遮挡区域.html')
lock={str(p):sha(p) for p in files}
(r/'completed_locked.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2))
bad=[p for p,h in lock.items() if sha(p)!=h]
assert not bad
(r/'audits/final_lock_verification.json').write_text(json.dumps({'time':now,'files':len(lock),'mismatches':bad,'logs_and_mutable_main_document_excluded':True},ensure_ascii=False,indent=2))
print(json.dumps({'time':now,'locked_files':len(lock),'state':state},ensure_ascii=False))
