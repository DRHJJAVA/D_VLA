from common13 import *
import subprocess,time,re
def main():
 # Continue the already-authorized CF013 job; never restart training.
 while True:
  s=json.loads((BASE/'stage_state.json').read_text())
  if s['stage']=='training_failed':return
  if s['stage']=='training_complete_pending_audit':break
  if s['stage'] in ['complete_dynamics_ablation','audited_pending_decision']:return
  child=json.loads((BASE/'runner_child.json').read_text())
  proc=Path('/proc')/str(child['pid'])
  if not proc.exists() and s['stage'] not in ['training_complete']:
   time.sleep(5)
   again=json.loads((BASE/'stage_state.json').read_text())
   if again['stage']==s['stage']:
    stage('continuation_stopped_child_absent',last_stage=s);return
  time.sleep(15)
 with (BASE/'audit.log').open('w') as log:
  p=subprocess.run([sys.executable,str(BASE/'src/audit_results.py')],cwd=BASE,stdout=log,stderr=log)
 if p.returncode:
  lines=(BASE/'audit.log').read_text(errors='replace').splitlines()
  save(BASE/'audit_failure.json',dict(time=now(),returncode=p.returncode,locations=[x.strip() for x in lines if x.startswith('  File ')],error=[x for x in lines if re.match(r'^[A-Za-z]+Error:',x)][-1:]))
  stage('audit_failed');return
 comp=json.loads((BASE/'comparison.json').read_text());when=now()
 agg=comp['aggregate'];means=comp['means']
 by={(r['arm'],r['subset']):r for r in agg}
 ref=by[('cf011_robot','all_targets')]
 chosen=min(CFG['arms'],key=lambda a:by[(a,'all_targets')]['error_mm'])
 better=by[(chosen,'all_targets')]['error_mm']<ref['error_mm']
 wins={}
 for arm in CFG['arms']:
  cells=[r for r in means if r['arm']==arm and r['subset']=='all_targets']
  wins[arm]=sum(r['error_mm']<next(x['error_mm'] for x in means if x['episode']==r['episode'] and x['horizon']==r['horizon'] and x['arm']=='cf011_robot' and x['subset']=='all_targets') for r in cells)
 decision=dict(time=when,mean_error_best_new_arm=chosen,new_arm_improves_mean=better,event_horizon_error_improvements_of12=wins,contact_model_changed=False,manual_ranges_used=False,next_step='Use frozen causal motion representation as a candidate for a separately registered contact-temporal ablation; inspect error and uncertainty per episode first. Dynamics gain alone must not become contact labels.')
 save(BASE/'decision.json',decision)
 text='\n\n### CF013执行结果与下一步\n记录时间：'+when+'。\n'
 text+='24次训练全部完成，每次600步，共14,400次更新；两组参数量均101,352。原逐步计划、初始化复制、训练折隔离、优化器步数、旧数据与结果哈希均核对；新输入与原独立因果恒速估计逐点一致。完整输入/权重/预测/指标保存在远程cf013；人工区间未参与训练或选择。\n\n'
 text+='以下为四episode、三提前量、三seed等权均值；误差是自动对应的三维位移误差，不是接触识别精度。95%覆盖是按坐标轴统计。\n\n'
 text+='|方法|评估点范围|误差mm|NLL|95%覆盖|95%完整宽度mm|\n|---|---|---:|---:|---:|---:|\n'
 for r in agg:
  vals=[f"{r[k]:.4f}" if r[k] is not None else '不适用' for k in ['error_mm','nll','coverage95','width95_mm']]
  text+='|'+r['arm']+'|'+r['subset']+'|'+'|'.join(vals)+'|\n'
 text+='\n在全部原有效目标上，逐episode×提前量12格的平均误差改善数：'+json.dumps(wins,ensure_ascii=False)+'。不能把同一episode的三个时刻当三次独立抓取。\n\n'
 text+='|示范|h4 CF011误差mm|h4直接预测mm|h4残差预测mm|\n|---|---:|---:|---:|\n'
 for ep in CFG['train_episodes']:
  vals=[next(r['error_mm'] for r in means if r['episode']==ep and r['horizon']==4 and r['subset']=='all_targets' and r['arm']==a) for a in ['cf011_robot',*CFG['arms']]]
  text+='|'+ep+'|'+'|'.join(f'{v:.4f}' for v in vals)+'|\n'
 text+='\n本轮新组中平均误差较小的是'+chosen+'；'+('该组平均误差低于CF011。' if better else '新组平均误差未低于CF011。')
 text+='这只决定后续候选表征的研究方向，不替换CF010接触模型。必须同时看NLL、区间覆盖和逐事件结果；未套用CF012方差校准系数，也未按人工接触边界调参。下一步将针对接触时序表征另行在本文件登记对照，仍仅使用示范RGB-D和机器人状态/动作生成自监督或弱监督，不把恒速/随手运动成立直接等同成功接触。\n'
 with (ROOT/'docs/小型验证方案_夹持运动证据与区域表示_v1.md').open('a',encoding='utf8') as f:f.write(text)
 with (ROOT/'docs/STATUS.md').open('a',encoding='utf8') as f:f.write('\n\n'+when+' CF013完成24次局部动力学对照训练与独立数值审计；结果追加唯一小型验证方案MD，CF010接触模型未替换，人工范围仅复核。\n')
 save(BASE/'document_update.json',dict(time=when,document=str(ROOT/'docs/小型验证方案_夹持运动证据与区域表示_v1.md'),appended_section_sha256=hashlib.sha256(text.encode()).hexdigest()))
 verify(json.loads((BASE/'audit_source_locked.json').read_text()));verify(json.loads((BASE/'continuation_source_locked.json').read_text()))
 stage('complete_dynamics_ablation',runs=24,updates=14400,contact_model_changed=False)
 paths=[p for p in BASE.rglob('*') if p.is_file() and p.suffix not in ['.log','.pyc'] and '__pycache__' not in str(p) and p.name not in ['completed_locked.json']]
 save(BASE/'completed_locked.json',lock(sorted(paths)))
 print(json.dumps(decision),flush=True)
if __name__=='__main__':main()
