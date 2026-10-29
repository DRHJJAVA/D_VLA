from common14 import *
import numpy as np

def main():
    audit = json.loads((BASE / 'audits/completion.json').read_text())
    assert audit['frozen_prediction_max_difference'] == 0
    summ = json.loads((BASE / 'evaluation/summary.json').read_text())
    diag = json.loads((BASE / 'audits/demo1_transport_diagnosis.json').read_text())
    doc = ROOT / 'docs/小型验证方案_夹持运动证据与区域表示_v1.md'
    msg = '\n\n### 32.2 冻结预测与运输诊断结果（' + now() + '）\n\n三段固定新任务均实际完成CF014 with_motion三种子last600推理，没有训练或阈值调整。双物体388帧、抽屉177帧、炉灶开关115帧，总计680帧；每段三种子，独立重推差0。四通道96×96 RGB-D、128×10点输入、50维状态及24维运动描述符的尺寸、有效性、t−1实测开度、手坐标变换和视频帧数均核对。模型及运动描述符源码与CF014逐字节一致，原输入准备核心AST仅清单路径/长视频显存上限适配。旧CF014权重和完成锁未变。\n\n|案例|总帧数|平均分数≥0.8的模型候选区间|\n|---|---|---|\n'
    for x in summ['cases']:
        msg += '|' + x['id'] + '|' + str(x['frames']) + '|' + ('、'.join((str(a) + '–' + str(b) for (a, b) in x['candidate_ranges'])) or '无') + '|\n'
    msg += '\n上述区间均为模型真实输出，不是动作名称规则或人工标签。没有新的逐帧接触真值，不计算新任务准确率/成功率。失败操作可能包含真实接触，既有整段抓空标记不再一概当无接触真值；当前不改历史人工记录，不把它们加入训练。新任务按固定demo0选取，未根据分数换案例，未打开预留demo40–44。官方任务说明只用于案例挑选和页面标题，不进入模型。\n\n**demo1运输低分**：85–116帧均分0.976，117–137帧均分0.282，后者跟踪点中位69、几何代理距离中位0.046mm、实际/预期运动方向余弦中位0.993，状态和运动输入均有效。固定权重将整个运动分支置零后仍为0.283，接近辅助头同段均分0.271。说明不能把低分解释为没有接触/全部失去观察，也不能只归因新运动分支；原有视觉/点轨迹与时序状态对运输接触的表征仍不稳。相关代理和输入干预不是物理接触真值，尚未确定唯一因果根因。\n\n**范围限制**：新场景为首帧真实表面、整段夹爪路径12cm空间并集适配；包括背景/固定结构、128点预算分散、首帧不可见面无法补出。全局刚体拟合不适合所有多物体/关节操作，旧准备程序输出的自动监督字段本轮完全未用于训练或评价，不能拿拟合失败冒充接触失败。此为离线视频定性检查，不能声称完全在线或已完成最终遮挡接触区域提取。\n\n页面：[CF017跨任务接触预测查看](CF017_跨任务接触预测查看.html)，三段新视频及原demo1均为无标记双相机视频，分数曲线与逐帧选择同步。新数据来自官方LIBERO-datasets，下载文件SHA256与官方LFS逐一一致，全部保留远程。\n'
    with doc.open('a') as f:
        f.write(msg)
    assets = ROOT / 'docs/图示/cf017_contact'
    page = ROOT / 'docs/CF017_跨任务接触预测查看.html'
    payload = json.loads((assets / 'data.json').read_text())
    files = [page, assets / 'data.json'] + [ROOT / 'docs' / x['src'] for x in payload['cases']]
    save(BASE / 'audits/display_locked.json', dict(time=now(), files=lock(files)))
    save(BASE / 'audits/resolved_history.json', dict(time=now(), resolved=True, failures=['failure_prepare.json', 'failure_finish.json', 'cases/two_objects_demo_0/failure.json', 'failure_predict.json', 'failure_resume.json', 'failure_audit.json'], cause='首次长视频CoTracker自限显存不足，仅提升上限；FFmpeg不在PATH，改用已安装imageio_ffmpeg；审计原始float64与保存float32误要求完全相等，改核对明确float32转换；视频HTTP无Range导致跳帧回零，替换同端口本机Range服务。均未改权重或预测。', no_training=True))
    save(BASE / 'source_final_locked.json', lock([p for p in (BASE / 'src').glob('*') if p.is_file()] + [BASE / 'config.json']))
    stage('complete_frozen_new_task_review')
    paths = [p for p in BASE.rglob('*') if p.is_file() and (not p.is_symlink()) and (p.name not in ['completed_locked.json', 'finalize.log']) and (not p.name.endswith('.pyc')) and ('raw' not in p.relative_to(BASE).parts) and ('rgbd' not in p.relative_to(BASE).parts)]
    save(BASE / 'completed_locked.json', lock(paths))
    verify(json.loads((BASE / 'completed_locked.json').read_text()))
    print(json.dumps(dict(time=now(), completed=True, locked_files=len(paths), large_inputs_locked_separately=True)))
if __name__ == '__main__':
    main()
