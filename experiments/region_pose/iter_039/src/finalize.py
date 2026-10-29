from pathlib import Path
import json,hashlib,datetime
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/region_pose/iter_039';V=R/'experiments/region_pose/reviews/iter_039'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def check(p):
 x=json.loads(Path(p).read_text());assert all(sha(R/k)==v for k,v in x.items());return len(x)
assert json.loads((D/'audits/independent_source_display.json').read_text())['passed']
checks={p.name:check(p) for p in [D/'source_locked.json',D/'source_audit_locked.json',D/'source_finalize_locked.json',D/'input_locked.json',R/'experiments/region_pose/iter_038/completed_locked.json',V/'immutable_locked.json']}
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
save(D/'audits/browser_review.json',{'utc':now,'page_checked':True,'all_11_overviews':True,'all_12_nonempty_clean_zooms':True,'full_panels':['45_robot0_eye_in_hand','50_agentview','64_agentview','113_robot0_eye_in_hand'],'unique_images_checked':27,'tab88_closed':True,'findings':'尝试内稀疏来源位于罐体上缘与右夹指邻接处；闭合前腕相机大片来源在罐盖/图像下方，遮挡与低分辨率不能确认逐点身份。零来源不证明失败，目视不生成成功标签。','identity_truth_verified':False,'user_acceptance':False})
(D/'结果与下一步.md').write_text("# 第039轮结果与下一步\n本轮已完成旧验证tomato_sauce_demo_4固定来源查看，不进行网络预测、训练、追踪或新监督构建。所有产物仅远程。\n固定11帧43/45/48/50/62/64/68/70/71/72/113，其中62为相对闭合50实测手z首次严格超过25mm（32.4474mm），不是按图挑帧。22双相机完整四联图、11概览、全部12非空视图无编号固定放大，共45图，来源22JSON；docs/第039轮_旧验证tomato4固定来源查看.html，reviews/iter_039/immutable_locked.json锁46文件。\n全选帧4396个源观测中，闭合前对照4377，尝试内所选8帧仅19（50/62/64/68/70/71/72/113分别1/1/3/4/4/3/3/0），全部主相机；不是独立表面或独立事件。闭合前43主/腕18/284，45为154/3918，48为3/0。不能把大量闭合前来源转为持物支持。\n后台检查页面、全部11概览、全部12无编号放大、4完整四联（45腕、50主、64主、113腕），共27不同图。主相机尝试内少数来源位于罐体上缘右侧与夹指邻接位置，遮挡和低分辨率妨碍逐点身份确认；没有明显整组漂到邻物，但不排除边缘混合。闭合前腕相机43局部罐盖边界，45大条带位于图像下方罐盖附近；不能凭图确认所有像素物体/机器人身份。113可见罐体靠近篮中，但零有限体积来源不是抓取失败事实；目视不新增成功标签。临时标签88关闭。\n完整RGB/深度图按固定0–2m显示，腕相机近距离深度图偏暗，实际浮点深度保留。所有源索引、像素、相机、原窄垫标记、RGB和深度可追溯到038/原同帧派生数据。独立核对22视图真实源和全图原RGB/深度像素、全部放大原RGB逐像素相同；88切片hash读取后保存，非事前已有。重投影最大1.1369e-13像素、深度误差2.2205e-16m。页面68引用均存在，旧038完成锁30文件、7输入和3初始设计/配置/脚本、独立审计附录均核对。\n本轮不改变038最长连续3帧不足5帧的结论，不降低阈值、不补端点凑监督、不将旧验证当盲测。白黄杯6仍仅1稳定正查询来源，分支停止。没有模型AP，网络/损失未变，第二步仍未验收。\n下一步建议独立登记040冻结证据覆盖与数据需求汇总，只读现有训练/旧验证摘要、状态及原24段方向统计，形成原20训练/4旧验证逐事件证据到监督的清单，明确未覆盖、数量不足、跨时不通过与侧向数据缺口；不重读RGB-D/评分或预留测试、不再凭缺失事件盲目扩网。来源真实身份、补侧向数据范围与第二步实际预测审核写远程待人工判断，休息期不提问。\n")
save(D/'待人工判断.json',{'source_identity':'罐体上缘/夹指邻接的稀疏源和闭合前腕相机大片源无法由本次目视确认逐点身份；不恢复监督。','sideways_data_scope':'原24段无明显侧向示范，新增数据范围须人判断。','rest_period':'不提问，不要求决策','stage2_user_review_required':True})
save(D/'stage_state.json',{'stage':'complete','utc':now})
summary=json.loads((D/'summary.json').read_text());summary['stage']='complete';summary['browser_review']='27 images checked';save(D/'summary.json',summary)
save(D/'audits/final_integrity.json',{'utc':now,'locks_checked':checks,'mismatches':0,'training':False,'tracking':False,'new_labels':False,'test_read':False,'stage2_accepted':False})
entry='\n\n## 第039轮完成\n固定旧验证tomato4的11帧/22双相机来源查看完成，45图全部远程；尝试内所选帧仅19源观测，全主相机，主要罐体上缘与夹指邻接，未证实逐点身份。闭合前4377观测不作持物证据。独立来源/显示/哈希检查通过；最长连续3帧不足5的分支继续停止，无新标签或训练。详情experiments/region_pose/iter_039/结果与下一步.md。第二步未验收。\n'
for p in [R/'docs/STATUS.md',R/'experiments/region_pose/迭代索引.md']:
 with p.open('a') as f:f.write(entry)
files=[p for p in D.rglob('*') if p.is_file() and p.suffix not in ['.log','.pyc'] and p.name!='completed_locked.json' and '__pycache__' not in p.parts]
save(D/'completed_locked.json',{str(p.relative_to(R)):sha(p) for p in sorted(files)})
assert check(D/'completed_locked.json')==len(files)
print(json.dumps({'stage':'complete','locked_files':len(files),'review_locked':checks['immutable_locked.json'],'utc':now}))
