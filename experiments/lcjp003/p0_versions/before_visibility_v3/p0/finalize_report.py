import json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003')
def read(name):return json.loads((R/'p0'/name).read_text())
net=read('network_check_result.json')
assert net['status']=='completed' and net['optimizer_updates']==1
data=read('data_audit.json');extra=read('corrected_input_validation.json');vis=read('cross_view_visibility_audit.json');weights=read('dinov3_probe.json');feat=read('features_audit.json')
assert weights['status']=='passed' and feat['status']=='completed'
assert all(v['finite'] and v['gradient_l2']>0 and v['parameter_delta_max']>0 for v in net['modules'].values())
stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S%z')
result={
 'created_bjt':stamp,'status':'P0_NOT_PASSED_CROSS_VIEW_VISIBILITY',
 'input_protocol_version':data.get('input_protocol_version'),
 'training_demos':2,'heldout_read':False,'windows':10,'optimizer_updates':1,
 'gates':{
  'official_dinov3_download_load':'passed',
  'input_action_time_alignment':'passed',
  'future_observation_exclusion':'passed',
  'source_camera_projection_and_rgb_sampling':'passed',
  'frozen_feature_parameters_unchanged':'passed',
  'all_real_windows_forward_finite':'passed',
  'action_loss_to_point_and_part_score_gradients':'passed',
  'one_update_all_module_parameters_change_and_finite':'passed',
  'cross_view_visibility':'not_passed'
 },
 'network':net,
 'cross_view':{k:v for k,v in vis.items() if k in ['totals','total_cycle_gt2px_fraction','outlier_edge_fraction','outlier_destination_depth_nearer_fraction','all_cycle_error_median_px','all_cycle_error_p95_px','interpretation']},
 'next_stage_started':False,
 'pending':'Correct boundary and occlusion handling in cross-view feature visibility, preserve all points and source-view observations, then repeat affected checks.',
 'evidence':{name:hashlib.sha256((R/'p0'/name).read_bytes()).hexdigest() for name in ['data_audit.json','corrected_input_validation.json','cross_view_visibility_audit.json','features_audit.json','network_check_result.json','network_input_lock.json','dinov3_probe.json','drifting_unit.json']}
}
(R/'p0/result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
doc=next((R.parent.parent/'docs').glob('LCJP003_*DriftingVLA*.md'))
marker='P0 实测结果（采样修复后，非整体通过）'
assert marker not in doc.read_text()
lines=['','### '+stamp+' '+marker,'',
'本次使用两个训练demo的10个窗口（6个完整窗口、4个起止边界窗口），完整16帧、2048点、16层动作专家，8组动作噪声；执行一次AdamW更新。未读取留出视频，未启动P1或恢复旧v001训练。','',
'| 检查 | 实际结果 |','|---|---|',
'| DINOv3-S+ | 官方115MB权重SHA匹配，严格加载无缺失/多余键，双相机当前图像输出有限 |',
'| 时间与因果 | 使用RGB-D同帧pre-action状态；114个条件输入数组在未来哨兵/未来动作扰动检查中逐项一致 |',
'| 渲染采样坐标 | 根据EGL实测4xMSAA sample0修复；源相机重投影P95约1.71e-5像素，RGB采样检查最大误差1.53e-5像素 |',
'| 冻结特征 | 修复后10窗全部重新提取；权重前后哈希相同、零梯度 |',
'| 完整网络 | 10窗输出与动作损失均有限；无效历史/点填NaN后的输出差异0 |',
'| 评分梯度 | 点内权重alpha L2='+str(net['score_chain']['alpha']['gradient_l2'])+'；部位权重beta L2='+str(net['score_chain']['beta']['gradient_l2'])+'；均由动作Drifting损失反传 |',
'| 一次更新 | '+str(len(net['modules']))+'个模块均有有限非零梯度，参数实际改变，更新后全部参数有限 |',
'| 跨视角可见性 | 4517个允许融合对应中749个未通过2像素循环一致性检查（16.58%）；中位0.0058像素，P95=13.52像素。未通过 |','',
'跨视角异常约85.98%位于深度不连续边缘；仍有平滑区异常，因此不能只用整体中位数宣布通过，也不把所有异常直接等同为错误物体匹配。当前剩余工作是遮挡/边缘处副视角特征的可见性处理；完整点云、源视角观测、失败对应及修复前记录均保留。该项修复和复验前不进入P1。','',
'网络检查总耗时'+format(net['seconds']/60,'.2f')+'分钟，峰值本进程GPU分配'+format(net['peak_gpu_allocated_gib'],'.2f')+'GiB。完整数值、输入/源码/权重/检查点SHA见 experiments/lcjp003/p0/result.json；P0检查点 one_update_engineering_only.pt 只记录这一次工程更新，不作为正式训练初始化。','']
with doc.open('a') as f:f.write('\n'.join(lines))
print(json.dumps({'status':result['status'],'document_updated':True,'score_chain':net['score_chain'],'module_count':len(net['modules']),'seconds':net['seconds'],'peak_gpu_gib':net['peak_gpu_allocated_gib']}))
