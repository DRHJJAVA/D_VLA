from pathlib import Path
import json,hashlib,datetime,collections
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/region_pose/iter_040'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def check(p):
 a=json.loads(Path(p).read_text());assert all(sha(R/k)==v for k,v in a.items());return len(a)
checks={p.name:check(p) for p in [D/'source_locked.json',D/'source_repair_locked.json',D/'source_finalize_locked.json',D/'input_locked.json']}
cfg=json.loads((D/'config.json').read_text());j=lambda k:json.loads(Path(cfg['inputs'][k]).read_text())
rows=json.loads((D/'episode_ledger.json').read_text())['rows'];s=json.loads((D/'summary.json').read_text())
expected_train={x['episode'] for x in j('supervision')['rows'] if x['split']=='train' and x['eligible']}
expected_val={x['episode'] for x in j('supervision')['rows'] if x['split']=='val' and x['eligible']}
assert {x['episode'] for x in rows if x['current_category']=='included_training'}==expected_train
assert {x['episode'] for x in rows if x['current_category']=='included_old_validation'}==expected_val
expected_missing={'alphabet_soup_demo_3','alphabet_soup_demo_6','porcelain_mug_demo_10','tomato_sauce_demo_4'}
assert {x['episode'] for x in rows if x['current_category']=='no_sustained_observed_volume_support'}==expected_missing
assert {x['episode'] for x in rows if x['current_category']=='too_few_initial_points'}=={'porcelain_mug_demo_4','tomato_sauce_demo_3'}
assert {x['episode'] for x in rows if x['current_category']=='too_few_stable_positive_query_sources'}=={'white_yellow_mug_demo_5','white_yellow_mug_demo_6'}
for x in rows:
 assert all((R/p).exists() for p in x['references'])
 assert x['physical_success_label'] is None and not x['source_identity_verified']
for phase in ['train','val']:
 vals=sorted(v for r in j('direction')['rows'] if r['split']==phase for v in r['vertical_axis_angle_deg'])
 median=vals[len(vals)//2] if len(vals)%2 else sum(vals[len(vals)//2-1:len(vals)//2+1])/2
 assert median==s['direction'][phase]['median_deg'] and len(vals)==s['direction'][phase]['query_observations']
save(D/'audits/independent_membership.json',{'passed':True,'stable_supervision_source_sets_equal_actual_membership':True,'missing_and_insufficient_sets_independently_checked':True,'direction_sorted_median_reconstructed':True,'all_references_exist':True,'no_physical_success_or_identity_truth_labels':True})
save(D/'audits/resolved_040_1.json',{'original_failure':'failure.json','cause':'023 uses cases,022 uses rows','repair':'build_ledger_v2.py','threshold_changed':False,'resolved':True})
(D/'结果与下一步.md').write_text("# 第040轮结果与下一步\n本轮已完成冻结证据覆盖与数据需求汇总。只读取16份既有JSON元数据，不读取RGB-D、NPZ、模型权重/分数或预留测试，不新增追踪、标签或网络训练。\n原始单位是24个episode，20训练/4旧验证；另有26闭合尝试（21训练/5旧验证）、78个严格夹前时刻（63/15），均不能冒充独立抓取。逐episode清单见覆盖清单_查看版.md与episode_ledger.json，尝试表见attempt_ledger.json。\n当前可沿用的持物候选元数据涉及17/20训练、3/4旧验证；早期对应数值规则通过12/20训练、3/4旧验证；稳定监督与实际029扩展/031训练清单为10训练事件40查询、3旧验证事件12查询。候选、对应和监督是不同层级，不是成功率。033/038完整有限体积重提仅覆盖5训练+1旧验证，不能称全24段均已重提。\n训练未纳入10段：soup3/6、瓷杯10缺持续有限区域支持；瓷杯4、tomato3初始点不足；soup0、瓷杯1、白黄杯3对应质量未过；白黄杯5/6稳定正查询来源为0/1不足。tomato3首次与后次尝试分开，后次有持物候选，不能称整段失败。旧验证tomato4完整体积交集最长3帧不足5，保持未知。白黄杯6有55/57/60对应和22跨时区域格，仍仅1稳定正查询来源，训练分支继续停止。\n已纳入样本也保留限制：soup5弱8/9持物内点；瓷杯5的58帧高留一敏感；白黄杯训练4和旧验证1仅3正来源；未知表面不负。029/031实际清单的逐事件/查询归属一致，040未重新评价网络性能，未改变best选择或默认方法。\n003方向既有数值重新汇总：训练角度最小/中位/最大1.4966/5.0462/10.2249度，旧验证2.6308/3.4297/8.6674度，均无>30度记录。参考是世界竖直，不是仿真物体坐标或真实侧抓标签；当前数据不足以支持上方/侧向切换结论。\n16输入前后hash一致，其中15与所属旧完成锁一致；003方向审计是后补文件，不在003原完成锁，本轮事前锁定并前后核对，明确不冒称旧锁覆盖。初始脚本805439因023摘要容器cases与022的rows不同而停止，未生成汇总；040.1只修正阶段接口，原失败日志/代码保留，806316接力完成，failure为已解决历史。网络、损失、门槛没有改变。\n下一步所需数据与人工判断见数据需求与待人工判断.md。保持现有方法冻结及健康检查，等待来源身份/新增方向数据范围/阶段2真实预测的人工判断；休息期不提问。没有新增可靠事件时不继续盲目扩网、叠损失、加轮或重复既有诊断。第二步仍未验收。\n")
(D/'数据需求与待人工判断.md').write_text("# 数据需求与待人工判断\n本文件仅列出待审核事项，不授权自动采集真实机器人动作，也不改变现有标签。休息期不发送问题；相关分支停在可审核状态。\n1. 来源身份与有限范围：审核已保存真实输入和来源页，尤其soup5弱对应、瓷杯5高敏感、白黄杯稀疏正来源、tomato4罐体上缘与夹指邻接遮挡。人工查看用于判断证据可靠性与数据范围，不转为人工逐点训练标签。\n2. 方向覆盖：当前原24段世界竖直接近轴均小于11度。需要独立真实示范覆盖不同接近方向、物体可见面和夹持姿态，同一对象应有多个独立抓取事件。具体采集数量、机器人动作范围与场景由用户审核；不从现有近竖直数据制造侧向答案，世界角度>30度只是现有缺口描述，不自动当新成功判定门槛。\n3. 观测覆盖：优先使主/腕相机能观察有限夹持段的物体表面与严格夹前查询表面；保留遮挡、失败尝试和未知样本记录。增加独立事件而非重复同事件帧，完整保留RGB-D、相机内外参、实际gripper_states/ee_pos/ee_ori/joint_states/actions与时间对齐，深度派生来源单列，t0状态未知不补造。\n4. 数据与评估边界：新范围先按episode划分并登记训练/验证规则，保留全部尝试，监督侧未来不能进入查询评分。沿用无名称/文字/仿真物体位姿或接触真值的输入限制。旧已看验证不是盲测；demo40–44继续隔离，方法冻结后另行预注册。\n5. 模型阶段审核：用户审核真正冻结模型预测页（029/031）及完整输入；039仅来源图，不替代模型预测。覆盖率、自动区域AP不等于物理接触精度或抓取成功率，不自动宣布第二步通过。\n当前安全独立工作已完成：逐事件证据表、出处与计数核对。等待上述人判断时只做健康/新输入检查，不重复训练、追踪或既有失败分支。\n")
raw=(D/'覆盖清单.md').read_text();view=raw.replace('\n\n|','\n|');(D/'覆盖清单_查看版.md').write_text(view)
save(D/'audits/display_revision.json',{'original_preserved':'覆盖清单.md','view':'覆盖清单_查看版.md','change':'remove blank lines between Markdown table rows only','data_unchanged':True})
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
save(D/'stage_state.json',{'stage':'complete','utc':now,'next':'frozen_health_checks_pending_human_data_scope_and_review'})
save(D/'audits/final_integrity.json',{'utc':now,'verified_locks':checks,'mismatches':0,'test_access':False,'rgbd_access':False,'network_inference_training':False,'new_labels':False})
entry='\n\n## 第040轮完成\n原20训练/4旧验证逐episode覆盖表已冻结。实际10训练40查询、3旧验证12查询；其余保持未知。世界竖直接近轴统计无>30度，不足以支持侧向结论。只读16元数据，15与旧锁一致；后补003方向文件本轮锁定。详见experiments/region_pose/iter_040/结果与下一步.md及数据需求与待人工判断.md。接下来冻结方法、健康检查，等待来源/新增方向数据范围及第二步人工审核，不盲目继续训练或重复诊断。\n'
for p in [R/'docs/STATUS.md',R/'experiments/region_pose/迭代索引.md']:
 with p.open('a') as f:f.write(entry)
files=[p for p in D.rglob('*') if p.is_file() and p.suffix not in ['.log','.pyc'] and p.name!='completed_locked.json']
save(D/'completed_locked.json',{str(p.relative_to(R)):sha(p) for p in sorted(files)})
check(D/'completed_locked.json')
print(json.dumps({'stage':'complete','locked_files':len(files),'utc':now}))
