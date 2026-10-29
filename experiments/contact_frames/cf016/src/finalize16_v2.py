from common14 import *
import numpy as np
def main():
 audit=json.loads((BASE/'audits/completion_audit.json').read_text())
 assert audit['control_prediction_max_difference']==0 and audit['new_prediction_max_difference']==0
 metrics=json.loads((BASE/'evaluation/review_metrics.json').read_text())
 doc=ROOT/'docs/小型验证方案_夹持运动证据与区域表示_v1.md'
 text='\n\n### 31.3 完成结果（'+now()+'）\n\n三个种子各完成600实际更新，只取last600，权重于'+json.loads((BASE/'weights_frozen.json').read_text())['time']+'冻结，再读取人工范围复核。模型源AST仅新增两维屏蔽，训练源AST仅改完成计数6为3；所有训练配置、初始张量、自动监督、特征和归一化与CF014一致。54组新旧冻结模型重推与保存分数最大差均0，旧CF014/CF015锁不变。\n\n固定0.8门槛，三种子平均分数形成候选，人工范围外不计负例：\n\n|事件|真实区间候选帧 CF014→CF016|明确负区间高分帧 CF014→CF016|\n|---|---|---|\n'
 for row in metrics['rows']:
  a=row['models']['control'];b=row['models']['no_point_mass'];p=a['positive'];q=b['positive']
  pos=f"{p['ensemble_high_frames']}/{p['frames']} → {q['ensemble_high_frames']}/{q['frames']}"
  if 'explicit_negative' in a:
   x=a['explicit_negative'];y=b['explicit_negative'];neg=f"{x['ensemble_high_frames']}/{x['frames']} → {y['ensemble_high_frames']}/{y['frames']}"
  else:neg='未提供明确负例'
  text+='|'+row['episode']+('（训练复核）' if row['split']=='training_review' else '')+'|'+pos+'|'+neg+'|\n'
 text+='\n五个复核事件等权真实覆盖85.03%→82.00%；两个明确负区间等权高分误报94.23%→92.52%。不能将平均误报微降称改进：番茄酱demo3抓空误报23/26→25/26，三种子负区间平均分数均上升；白黄杯demo5未接触仅减少1帧，罐头demo1真实覆盖51/82→38/82明显退化。demo0总体候选保留且覆盖61/88→63/88，但后两个种子覆盖分别75.0%→73.9%、96.6%→90.9%，不能称每个种子均改善；demo9三个种子均保留47/47。\n\n结论：仅移除点量并重新训练，没有达到减少抓空与未接触误报的目标，不替换CF014。本轮保留全部模型和不利结果，不挑种子、不改阈值。CF015对固定权重的输入置均值能降分，不能推出重训收益；本轮说明在现有监督和共享辅助任务下，仅屏蔽点量不足以解决问题。仍不能单凭本轮确定自动监督或辅助损失谁是唯一根因。\n\n下一项值得单独设计的是近距与接触监督的混淆：训练中的接近但未建立作用关系缺少可靠区分监督，以及共享近距辅助损失是否让主分数继续跟随接近程度。本轮不叠加修复，不把人工失败区间加入训练，不把不运动直接标为失败。\n\n网页：[CF016 去除点量输入接触对照](CF016_去除点量输入接触对照.html)，复用六段已锁定无标记双相机视频，显示新旧曲线、三种子范围、完整候选和人工复核带。迭代目录experiments/contact_frames/cf016。首次文件存在断言属于已解决实施历史，未发生优化，不盲重跑。\n'
 if '### 31.3 完成结果' not in doc.read_text():
  with doc.open('a') as f:f.write(text)
 page=ROOT/'docs/CF016_去除点量输入接触对照.html';dp=ROOT/'docs/图示/cf016_contact/data.json';payload=json.loads(dp.read_text())
 assets=[page,dp]+[ROOT/'docs'/c['src'] for c in payload['cases']]
 assert all(p.is_file() for p in assets)
 for c in payload['cases']:
  assert c['src'].endswith('_clean.mp4')
  for arm in ['control','no_point_mass']:
   assert len(c['models'][arm]['mean'])==c['n']
 save(BASE/'audits/display_manifest.json',dict(time=now(),files=lock(assets),reused_clean_videos=True,video_count=6,manual_bands_review_only=True))
 save(BASE/'source_final_locked.json',lock([p for p in (BASE/'src').glob('*') if p.is_file()]+[BASE/'config.json']))
 save(BASE/'summary.json',dict(time=now(),id='CF016',status='completed_no_overall_benefit',metrics=metrics['aggregate'],recommendation='do_not_replace_CF014',manual_training_use=False))
 stage('complete_single_factor_point_mass_ablation')
 paths=[p for p in BASE.rglob('*') if p.is_file() and not p.is_symlink() and p.name not in ['completed_locked.json','finalize.log','finalize_v2.log'] and not p.name.endswith('.pyc')]
 save(BASE/'completed_locked.json',lock(paths))
 verify(json.loads((BASE/'completed_locked.json').read_text()))
 print(json.dumps(dict(stage='complete',locked_files=len(paths),time=now())))
if __name__=='__main__':main()
