from pathlib import Path
import json,hashlib,time,re,html
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/region_pose/iter_029'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False))
checks={}
for fn,root in [('source_locked.json',R),('input_locked.json',R),('training_inputs_locked.json',B),('source_addendum_audit_review.json',R)]:
 lock=json.loads((B/fn).read_text())
 for f,h in lock.items():assert sha(Path(f) if Path(f).is_absolute() else root/f)==h,(fn,f)
 checks[fn]=len(lock)
V=B.parent/'reviews/iter_029'
lock=json.loads((V/'immutable_locked.json').read_text())
for f,h in lock.items():assert sha(R/f)==h,f
page=R/'docs/第029轮_真实模型数据扩展对照.html'
refs=re.findall(r'(?:href|src)="([^"]+)"',page.read_text())
imagepaths={html.unescape(x) for x in refs if x.endswith('.png')}
assert len(imagepaths)==132
for f in imagepaths:assert (R/'docs'/f).exists()
dump(B/'audits/report_audit.json',dict(unique_images=len(imagepaths),broken_links=0,review_locked=len(lock),full_inputs=22,model_prediction_images=88,automatic_supervision_images=22))
dump(B/'audits/final_integrity.json',dict(checks=checks,review_locked=len(lock),time=time.time(),unchanged=True,test_read=False,stage2_accepted=False))
dump(B/'stage_state.json',dict(stage='complete',time=time.time(),training_runs=4,updates_per_run=240,validation_events=3,validation_queries=12,review_complete=True,stage2_accepted=False))
files=[p for p in B.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix!='.log' and p.name not in ['completed_locked.json','pipeline.lock']]
dump(B/'completed_locked.json',{str(p.relative_to(R)):sha(p) for p in sorted(files)})
for f,h in json.loads((B/'completed_locked.json').read_text()).items():assert sha(R/f)==h
status='\n\n## 第029轮完成\n四run各240更新，6事件/10事件同新监督数据扩展对照已完成；基础/条件best AP控制0.2111/0.2189，扩展0.4256/0.4472。扩展白黄杯仍0.0485/0.0432且条件loss较差，前20%约54%未知/40%已知负，不能以覆盖称精确区域或成功率。52样本、初始化/计划/optimizer/RNG核对；416次冻结best/last推理，48验证best分数差0。真实页含132图并后台查看22图；结果和完成锁在iter_029。所有产物远程，测试未读，第二步未验收。下一步先另版030训练集损失/曝光诊断，不直接加轮或叠加尾部损失。\n'
with (R/'docs/STATUS.md').open('a') as f:f.write(status)
with (B.parent/'迭代索引.md').open('a') as f:f.write('\n- 029：同监督6/10事件固定240步数据扩展对照完成；4run、132真实预测/输入/监督图，第二步未验收。见iter_029/结果与下一步.md。\n')
print('FINALIZED',len(files),flush=True)
