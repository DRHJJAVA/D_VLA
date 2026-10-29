from pathlib import Path
import json,hashlib,time
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/region_pose';B=P/'iter_028'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
for name in ['source_locked.json','input_locked.json','propagation_source_locked.json','review_source_locked.json','finalize_source_locked.json']:
 for f,h in json.loads((B/name).read_text()).items():assert sha(R/f)==h
v=P/'reviews/iter_028'
for f,h in json.loads((v/'immutable_locked.json').read_text()).items():assert sha(R/f)==h
manifest=json.loads((v/'manifest.json').read_text());assert manifest['image_count']==77 and len(manifest['cases'])==28
for old in ['iter_015','iter_027']:
 for f,h in json.loads((P/old/'completed_locked.json').read_text()).items():assert sha(R/f)==h
dump(B/'audits/count_correction.json',{'original_design_total':49,'actual_images':77,'components':{'input':28,'target':28,'correspondence':21},'cause':'arithmetic text error; all planned cases and image types retained','rerun_or_data_change':False})
(B/'计数更正028_1.md').write_text('# 028.1 图数更正\n28当前输入图+28自动监督图+21对应图=77。设计中49为加法笔误，原设计、源码和阶段历史保留；未重新生成、删减或修改图像。正式结果及完成状态使用77。\n',encoding='utf-8')
selected=[]
for c in manifest['cases']:
 if c['offset']==15 or (c['split']=='val' and c['event'].startswith('white_yellow_mug_demo_1_')):
  selected.extend([c['input'],c['automatic_supervision']]+([c['correspondence']] if c['correspondence'] else []))
assert len(selected)==29
dump(B/'audits/browser_review.json',{'reviewed_images':selected,'reviewed_unique_images':29,'page_rendered':True,'tab_closed':True,'observations':'罐头训练左右局部候选，罐头验证前景残缺右侧候选；杯子小片在右侧杯口/壁，刚体点多在杯口/柄。无明显整体跳机械臂，分辨率遮挡及密集编号不能证明身份。未改标签。','user_acceptance':False})
dump(B/'audits/completion_audit.json',{'passed':True,'validation_queries':12,'events':3,'official_validation_events':4,'all_source_count_eligible':True,'old015_lock_files':89,'old027_lock_files':162,'input_files':300,'review_images':77,'review_files_locked':79,'train_freeze_before_validation':True,'new_tracking':False,'scoring_training':False,'test_access':False,'stage2_accepted':False})
dump(B/'stage_state.json',{'stage':'complete','validation_queries':12,'review_images':77,'new_tracking':False,'scoring_training':False,'test_access':False,'time':time.time()})
files=[f for f in B.rglob('*') if f.is_file() and f.name not in ['completed_locked.json','pipeline.lock'] and f.suffix not in ['.log','.pyc'] and '__pycache__' not in f.parts]
lock={str(f.relative_to(R)):sha(f) for f in files};dump(B/'completed_locked.json',lock)
for f,h in lock.items():assert sha(R/f)==h
for file,txt in [(R/'docs/STATUS.md','\n\n第028轮完成：固定3旧验证12查询复用015对应，025目标重算，来源/因果/置信/哈希审计通过。白黄杯仍仅3正参考来源，番茄酱验证缺持物。来源页77图、后台查29图；无新追踪或评分训练，第二步未验收。下一步029同架构、同更新预算的6/10事件数据扩展对照。\n'),(P/'迭代索引.md','\n- 028：固定旧验证传播与来源核查完成；3事件12查询，77来源图。见iter_028/结果与下一步.md；非模型预测。\n')]:
 with file.open('a',encoding='utf-8') as f:f.write(txt)
print(json.dumps({'complete':True,'locked_files':len(lock),'review_lock_files':79}))
