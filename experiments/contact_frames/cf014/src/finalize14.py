from common14 import *
def main():
    assert json.loads((BASE/'audits/completion_audit.json').read_text())['passed']
    assert (BASE/'audits/clean_video_audit.json').exists()
    assert (BASE/'audits/browser_review.json').exists()
    display=json.loads((BASE/'audits/display_manifest.json').read_text())
    assert all(sha(p)==h for p,h in display['files'].items())
    m=json.loads((BASE/'evaluation/review_metrics.json').read_text());stamp=now()
    doc=ROOT/'docs/小型验证方案_夹持运动证据与区域表示_v1.md'
    marker='### 29.1 CF014 实际接触与抓空对照结果'
    assert marker not in doc.read_text()
    text='\n\n'+marker+'（'+stamp+'）\n\n'
    text+='已完成两组各三个随机种子、每次600步，共3600次更新；每组231,671参数。两组初始输出完全相同，均复用CF010原始初始化和600步计划，不输入运动组的分支始终为零。权重于'+json.loads((BASE/'weights_frozen.json').read_text())['time']+'冻结，再生成五段留出复核视频的运动特征与模型预测。人工范围仅用于冻结后的指标复核，训练程序不读取人工范围；没有访问预留测试。\n\n'
    text+='**结论：不能认定运动特征已解决实际接触与抓空的区分；本轮不替换旧方法。** 五段复核视频等权接触覆盖由78.29%增至85.03%，但两个明确负区间等权高分比例由77.78%增至94.23%。后者分别是一个抓空段和一个接触前靠近段；这不是全视频准确率。阈值固定0.8，采用三个种子均值。\n\n'
    text+='| 复核视频 | 人工接触范围 | 不输入运动覆盖 | 输入运动覆盖 | 明确负段高分帧：不输入→输入 |\n|---|---|---:|---:|---|\n'
    for r in m['rows']:
        a=r['models']['without_motion'];b=r['models']['with_motion'];neg=a.get('explicit_negative')
        bad=(str(neg['ensemble_high_frames'])+'→'+str(b['explicit_negative']['ensemble_high_frames'])+'/'+str(neg['frames'])) if neg else '未提供，不计负例'
        text+='| '+r['episode']+('（训练片，单列）' if r['split']=='training_review' else '')+' | '+str(r['human_positive'])+' | '+format(a['positive']['ensemble_high_fraction'],'.2%')+' | '+format(b['positive']['ensemble_high_fraction'],'.2%')+' | '+bad+' |\n'
    text+='\n番茄酱demo3的49–74帧高分误报26→23，负段平均分0.9914→0.9070，仍严重误判；三个加入运动的种子分别误报22/22/25帧。白黄杯demo5的44–52帧高分误报5→9，三个加入运动的种子都误报9/9帧，负段平均分0.7373→0.9855。demo0真实接触覆盖63.64%→69.32%，demo9保持100%，不能以这些召回改善掩盖负段退步。\n\n'
    text+='与旧CF010比较，两组CF014均更容易报高分：旧CF010在demo3明确抓空段误报10/26、demo5接触前段0/9。两组CF014共享了未知监督语义修正，因此不能把对旧CF010的变化全部归因于运动输入，也不能据此恢复把未知当负例的旧做法。直接配对对照显示，当前运动输入对抓空只有小幅改善，对接近阶段反而恶化。\n\n'
    text+='验证边界：此轮测试由当前及历史RGB-D得到的实际三维运动、随手刚体运动残差与可用性，经24→64→128分支进入接触GRU；**不是CF013学习权重迁移实验**。既有点输入可能使用离线追踪，不宣称整条输入在线。四段训练、五段历史已看过的复核视频不足以证明泛化。实际接触监督仍为自动弱监督，几何分离是负证据，缺少运动支持不再自动为负。没有新增必须抬升或固定动作类别规则。\n\n'
    text+='已核对：25训练输入、36评估输入、6最终权重、9组预测文件哈希；六组600步计划、相同初始化、优化器步数、逐点监督独立重建、未知主损失梯度零、原始模型与新零分支初始输出差零。新运动分支最终权重范数约2.42–2.51，确实参与训练。全部逐种子预测、指标和源码保存于experiments/contact_frames/cf014，旧CF010–CF013均未修改。\n\n'
    text+='展示页：CF014_运动特征与实际接触抓空对照.html。六段视频使用主相机和腕相机原始RGB重新编码，去掉旧视频中CF005页脚；无追踪点覆盖。显示两组分数、三种子范围、候选区间及人工复核区间，训练demo7独立标注。预测数值在展示修改前后保持不变。\n\n'
    text+='下一步依据：应优先检查自动正证据是否仍把接近/共同背景运动当有效作用、训练负证据是否仅覆盖远离而缺少近距离抓空，以及局部汇总是否混入背景。以上是待验证解释，不能根据本轮直接认定单一根因。后续若设计对照，仍不得把这六个人工区间作为训练标签、调阈值或选择权重；不能仅凭运动预测误差下降宣称接触识别提升。本轮按用户要求完成验证，暂不自动再启动一轮网络训练，第二步未验收。\n'
    with doc.open('a',encoding='utf-8') as f:f.write(text)
    save(BASE/'audits/final_integrity.json',dict(time=stamp,document=str(doc),document_sha_at_completion=sha(doc),
         display_files=len(display['files']),completion_audit_passed=True,
         conclusion='motion input does not reliably discriminate contact from failed/near attempts',
         next_training_started=False))
    stage('complete_contact_feature_ablation')
    files=[p for p in BASE.rglob('*') if p.is_file() and p.suffix not in ['.log','.pyc'] and '__pycache__' not in str(p) and p.name!='completed_locked.json']
    save(BASE/'completed_locked.json',lock(files))
    print(json.dumps(dict(stage='complete',time=stamp,locked_files=len(files)),ensure_ascii=False))
if __name__=='__main__':main()
