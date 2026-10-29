from pathlib import Path
import json,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path('/mnt/nvme1/drh/D_VLA');BASE=ROOT/'experiments/contact_frames/cf015'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for a in iter(lambda:f.read(8388608),b''):h.update(a)
 return h.hexdigest()
def main():
 stamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
 audit=json.loads((BASE/'audits/completion.json').read_text());assert audit['passed']
 doc=ROOT/'docs/小型验证方案_夹持运动证据与区域表示_v1.md'
 marker='### 30.4 CF015 诊断结论与下一项对照'
 assert marker not in doc.read_text()
 text='\n\n'+marker+'（'+stamp+'）\n\n'
 text+='本轮完成冻结诊断，未训练新网络、未修改旧标签/权重/阈值，未读取RGB-D原图或预留测试。训练四事件先统计并冻结，再五段旧复核同规则执行，人工范围最后汇总。所有原预测重算差小于1e-6；80项CF014完成锁核对未变。主诊断为7输入变体×3种子×9事件，另复核无运动控制；点量分解为5变体×3种子×9事件。变体不是独立事件或物理反事实。\n\n'
 text+='**已确认的主要问题是监督缺少“近但没有实际接触”的区分证据，并且当前运动分支依赖局部点量线索；尚不能把失败全部归因于运动特征不含接触信息。** 以下分别记录机制证据与因果边界。\n\n'
 text+='1. **主接触目标发生了语义变化，这是CF014实现引入的变化。** 原CF008/010的软目标逐点等于near_target×interaction_support；CF014将该乘积放入权重，而正候选目标统一置1。四训练事件517帧实际为97正、159明确分离负、261未知。97正中21帧原软目标<0.1、38帧<0.5。在预测0.5处，这38帧原软目标的BCE梯度向下，新目标梯度向上；降低权重只减小力度，不改变方向。但21个极弱正帧仅占正权重总质量1.45%，不能仅凭数量认定它们造成全部退步。\n\n'
 text+='2. **负证据覆盖的缺口确定存在。** 159个负帧的自动接近目标最大仅4.05e-7，按这一代理指标全部属于明确分离；没有near_target≥0.5的负帧。修正未知后，25个原带权重的0目标变未知，其中7帧自动接近目标≥0.5：tomato7的42/43/44、tomato6的60/61、soup3的42/43。它们没有被独立证明是失败，不应恢复为硬负；但因此训练没有教模型在近距离情形区分有无实际作用。四事件中的同类近距离未知仍由辅助接近任务参与表示学习。这里指出监督缺口，不按人工范围补负标签。\n\n'
 text+='3. **模型的主接触输出与接近输出高度相似。** 九事件×三种子完整视频相关系数0.9702–0.9997；完整视频相关性包含阶段/远近共同变化，不单独证明模型完全复制辅助头。冻结最终权重处、按各自实际600步曝光加权，辅助接近损失对共享融合层/GRU的梯度范数为主接触的1.60–88.64倍，中位18.83倍。主任务已经接近拟合时也会出现这种梯度差，因此不能外推整个训练过程一直被辅助任务主导。两条证据与近距离负证据缺失一致，但不是独立物理真值。\n\n'
 text+='4. **运动分支被使用了，但点量通道是重要依赖。** 保持模型权重固定，只将输入组置训练均值：\n\n'
 text+='| 明确负段 | 原分数 | 实测运动组置均值 | 相对手残差组置均值 | 有效运动比例置均值 | 两个点量通道置均值 |\n|---|---:|---:|---:|---:|---:|\n'
 text+='| tomato3 49–74 | 0.9070 | 0.9219 | 0.8899 | 0.9069 | 0.7859 |\n'
 text+='| 白黄杯5 44–52 | 0.9855 | 0.9905 | 0.9272 | 0.9833 | 0.6343 |\n\n'
 text+='两个点量通道分别为log1p(有效运动空间权重总量)与log1p(当前有效深度空间权重总量)，不是接触关系；它们包含视距、可见面积和局部场景密度信息。保持有效标志、其他运动量及输入不变，点量干预改变了高分输出，支持模型依赖此类线索。操作是输入分布扰动，不证明自然数据里的物理因果，也不证明去掉通道重训必然获益。直接干预后五个复核事件真实接触高分帧分别59→57、51→45、39→39、61→53、47→47；不能只看负段下降就采用它。\n\n'
 text+='5. **新运动表示未明确绑定目标物体身份。** 代码核对：24维描述符输入为当前点、速度、有效性和手位姿；以夹爪附近高斯空间权重汇总当前有效深度，没有读上游component物体掩膜，也没有目标身份参数。它可能混合邻物/背景，且汇总后丢失逐点对应身份；这里只确认表示结构，现存24维聚合量不能反推出混入背景的比例。不能把“可能混入背景”写成已经定量证实的唯一根因。\n\n'
 text+='三维运动是否没有信息：当前证据不支持这一断言。tomato3抓空/实际接触区间的实测位移中位分别约1.88/9.18mm每帧、相对手残差3.88/1.93mm每帧；白黄杯5接触前/实际接触实测位移0.87/4.95mm每帧，但残差2.79/3.88mm每帧，方向相反。这些是复核描述，未拟合阈值；不能据它们新增必须运动、必须小残差等规则，静止接触和不同操作仍可能存在。\n\n'
 text+='**下一项优化应做单因素重训，先检验已定位的点量依赖。** 保持CF014监督/初始化/计划/预算和三种子不变，仅让两点量通道不进入主接触语义分支；不得同时改软目标、辅助损失或网络规模。对照全部五复核及训练片，考察抓空与接触前误报是否下降、demo0/9是否被牺牲。这个对照只验证点量依赖能否在重训后避免，并不解决已确认的监督缺口。之后再独立设计监督对照，明确区分接触状态、支持强度和观测置信，近距离负证据只能来自可观察的真实分离/非接触证据，不能用没有运动或人工区间充当标签。\n\n'
 text+='本轮仅完成用户要求的原因诊断，未把冻结输入扰动当新模型发布，也未启动上述重训。所有产物位于experiments/contact_frames/cf015，完整原预测、梯度与曝光分解、逐事件/逐种子输入敏感度保留；人工范围只用于后置复核。旧CF014结果页保持不变，第二步未验收。\n'
 with doc.open('a',encoding='utf-8') as f:f.write(text)
 out=dict(stage='complete_frozen_cause_diagnosis',time=stamp,network_training=False,document=str(doc),document_sha_at_completion=sha(doc),
    confirmed=['weak target hardened into positive label with confidence weight','no proximal negative supervision under existing near proxy','frozen model dependence on weighted point mass channels'],
    not_proven=['all motion representations useless','background contamination fraction','single exclusive cause','benefit of any new training fix'])
 (BASE/'stage_state.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 files=[p for p in BASE.rglob('*') if p.is_file() and p.suffix not in ['.log','.pyc'] and '__pycache__' not in str(p) and p.name!='completed_locked.json']
 (BASE/'completed_locked.json').write_text(json.dumps({str(p):sha(p) for p in files},ensure_ascii=False,indent=2))
 print(json.dumps(dict(stage=out['stage'],time=stamp,files=len(files)),ensure_ascii=False))
if __name__=='__main__':main()
