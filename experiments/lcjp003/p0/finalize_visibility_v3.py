import pathlib,json,hashlib,datetime
r=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003')
p=r/'p0'
root=r.parents[1]
def read(n): return json.loads((p/n).read_text())
def sha(f): return hashlib.sha256(f.read_bytes()).hexdigest()
n=read('network_check_result.json')
assert n['status']=='completed' and n['optimizer_updates']==1 and n['windows_checked']==10
assert all(v['finite'] and v['gradient_l2']>0 and v['changed_values']>0 for v in n['modules'].values())
assert all(v['finite'] and v['gradient_l2']>0 for v in n['score_chain'].values())
assert all(v==0 for v in n['invalid_nan_invariance'].values())
lock=read('network_input_lock.json')
assert sha(p/'network_input_lock.json')==n['input_lock_sha256']
for name,digest in lock['source'].items(): assert sha(r/'src'/name)==digest
for name,digest in lock['data'].items(): assert sha(root/name)==digest
for name,digest in lock['features'].items(): assert sha(p/'features'/name)==digest
assert sha(p/'visibility_v3_config.json')==lock['visibility_config_sha256']
assert sha(p/'features_audit.json')==lock['frozen_features_audit_sha256']
assert sha(p/'one_update_engineering_only.pt')==n['checkpoint_sha256']
assert lock['source']['model.py']=='1dbed592eaf87ed1adc076863712a678ecdce7e5abba4e096956c77a218c8725'
assert lock['source']['drifting.py']=='65556dc569493557f7571dfd9dae77e3c6fc6eb88a868bcf58e463808442ac18'
f=read('features_audit.json')
assert f['all_features_finite'] and f['all_input_hashes_match'] and f['all_feature_inputs_causal']
assert not f['heldout_or_dev_content_read'] and f['raw_readonly']
b=f['frozen_backbone']
assert b['unchanged'] and not b['any_gradient'] and b['requires_grad_count']==0
assert b['frozen_digest_before']==b['frozen_digest_after']
s=read('visibility_v3_synthetic.json'); real=read('visibility_v3_real.json'); c=read('visibility_v3_contracts.json')
assert s['status']==real['status']==c['status']=='passed'
assert real['all_original_sampled_points_preserved']
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S%z')
old=read('result.json')
gates=old['gates'].copy()
gates.update(cross_view_visibility='passed_with_conservative_secondary_coverage',feature_boundary='passed',future_observation_exclusion='passed',input_source_checkpoint_hashes='passed')
result=dict(created_bjt=now,status='P0_PASSED_VISIBILITY_V3',input_protocol_version='p0_surface_visibility_v3',
 training_demos=2,heldout_read=False,windows=10,optimizer_updates=1,gates=gates,network=n,
 cross_view={'synthetic':s,'real':real,'contracts':c},next_stage_started=False,
 limitations=['Small P0 geometry and gradient check; scoring efficacy and task success not evaluated.',
 'Secondary use decreased from7277 to2992 with all20480 source points retained.',
 'Sphere foreground visible-interior secondary coverage511/1576; overall95.5 percent is not per-object coverage.',
 'Depth tolerance calibrated for current simulated EGL observations; real sensor requires separate calibration.'],
 pending='P0 repair and revalidation complete; P1 and full training not started.',
 prior_evidence_archive='p0_versions/before_visibility_v3',evidence={})
for name in ['visibility_v3_config.json','visibility_v3_synthetic.json','visibility_v3_real.json','visibility_v3_contracts.json','data_audit.json','features_audit.json','network_check_result.json','network_input_lock.json','dinov3_probe.json','drifting_unit.json']:
 result['evidence'][name]=sha(p/name)
(p/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
alpha=n['score_chain']['alpha']['gradient_l2']; beta=n['score_chain']['beta']['gradient_l2']
text=f"""
## P0 遮挡边缘融合修复与复验完成（{now}）

本次结论：P0 的双视角几何与网络学习链路检查通过。范围为两个训练 demo_0、10 个窗口、一次工程参数更新。网络结构、动作损失、数据划分保持不变；P1 和正式训练尚未启动，旧 v001 继续暂停。

### 修复了什么

- 副视角深度使用局部连续表面估计，不跨前景/背景深度断层插值。容差由独立 EGL 平面误差测量固定：单视角0.149901毫米、双视角0.299802毫米；没有按本次视频挑阈值。
- 将副视角区分为可靠、遮挡、不确定和不可用。只有可靠副视角进入融合；来源相机点及特征保留，不删除遮挡边缘点，不生成接触或评分标签。
- 有效图像边缘的特征取样改为边界延拓，修复零填充造成的特征衰减。图像外点仍由可见性条件排除。
- 旧输入、失败结果和源文件已归档于 experiments/lcjp003/p0_versions/before_visibility_v3。新输入协议为 p0_surface_visibility_v3。

### 几何结果与覆盖代价

| 检查 | 修复前 | 修复后 |
|---|---:|---:|
| 原6个诊断帧副视角参与数 |4517|2013|
| 其中回投误差大于2像素 |749|0|
| 参与融合对应的回投误差P95 |13.5244像素|0.004898像素|
| 另4个边界帧副视角参与数 |2760|979|
| 其中回投误差大于2像素 |541|0|
| 参与融合对应的回投误差P95 |14.0455像素|0.005003像素|
| 来源点保留 |20480|20480|

回投检查保持原独立的双线性深度检查方式；新选择器没有按回投误差筛点。新的P95描述筛选后参与融合的对应，不能解读为所有旧对应均被校正。副视角参与合计7277降至2992，部分可用副视角也会被保守拒绝；这些点继续使用来源视角。

独立解析几何包括斜平面、深度台阶、薄片和球面：接纳22618个副视角对应，错误接纳0个；可见内部点总覆盖95.50%。其中球面前景仅511/1576（32.42%）使用副视角，其余回退来源视角。95.50%包含大量平面区域，不能代表曲面覆盖率。后续曲面融合改进应同时看错误接纳与覆盖率，不能只放宽阈值。当前容差适用于本次模拟深度；真实传感器需重新测量噪声。

### 网络与输入复验

- 10/10真实窗口前向和动作损失有限；无效点及补齐历史注入NaN后，有效输出最大差异0。
- 动作损失传回点内权重alpha、部位权重beta：梯度L2分别为{alpha:.8g}、{beta:.8g}，均有限且非零。
- 12/12模块梯度有限且非零，一次AdamW更新后均有参数变化、参数仍有限。该更新仅为P0工程检查。
- 6个历史窗口、每个27个数组通过禁止读取未来观测检查；改变未来动作只改变监督，不改变条件输入。
- 有效边缘常量特征保持1；来源特征1、副视角99但不可见时输出仍为1，两者都可见时输出50。
- 冻结视觉/语言骨干前后摘要一致、无梯度；当前基线仍使用SmolVLM2冻结特征。DINOv3官方权重已下载并单独验证，没有在此复验中替换基线。
- 全部10份样本、10份特征、源文件、可见性配置、输入锁及检查点SHA256复核一致。模型与Drifting损失源码hash保持原值。
- 本次完整网络复验耗时{n['seconds']:.1f}秒，峰值GPU分配{n['peak_gpu_allocated_gib']:.3f} GiB。没有读取开发集或留出集。

详细数值记录：experiments/lcjp003/p0/result.json；几何记录 visibility_v3_synthetic.json、visibility_v3_real.json；输入/边缘检查 visibility_v3_contracts.json；网络检查 network_check_result.json。此前失败结论保留为历史，本节及visibility_v3结果为当前P0状态。
"""
doc=root/'docs'/'LCJP003_状态条件三维部位评分与DriftingVLA设计方案.md'
with doc.open('a',encoding='utf-8') as out: out.write(text)
print(json.dumps({'status':result['status'],'document':str(doc),'network_seconds':n['seconds'],'score_gradient_l2':{'alpha':alpha,'beta':beta},'source_points_retained':20480,'secondary_used':2992,'hashes_verified':True},ensure_ascii=True))
