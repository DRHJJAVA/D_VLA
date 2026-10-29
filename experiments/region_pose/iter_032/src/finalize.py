from pathlib import Path
import json,hashlib,datetime
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';R=B/'iter_032'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def verify(p):
 d=json.loads(p.read_text())
 for k,v in d.items():assert sha(P/k)==v,k
 return len(d)
checks={n:verify(R/n) for n in ['source_locked.json','input_locked.json','source_addendum_032_2.json','source_addendum_032_3.json','finalize_source_locked.json']}
s=json.loads((R/'summary.json').read_text())
assert len(s['episodes'])==4 and all(not a['accepted'] for ep in s['episodes'] for a in ep['attempts'])
save(R/'audits/resolved_032_2_3.json',{'historical_failures':['failure.json','failure_v2.json'],'resolved_by':'src/audit_continuity_v3.py','t0_gap_unknown_preserved':True,'json_null_npz_nan_equal':True,'no_threshold_changed':True})
save(R/'待人工判断.json',{'stage2_accepted':False,'source_surface_identity':'待独立审核，当前只确认来源计数和规则复现','sideways_demonstration_scope':'现有方向不足，需要另行界定数据范围，休息期不提问','blocked_branch':'将缺失事件直接提升为监督或宣告真实失败','independent_next_work':'先登记033有限体积来源范围对照，不按本轮结果改标签'})
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
save(R/'audits/final_integrity.json',{'utc':now,'locks_verified':checks,'input_scope':'four original training episodes existing derived motion only','validation_read':False,'test_read':False,'new_training':False,'new_labels':False,'stage2_accepted':False})
save(R/'stage_state.json',{'stage':'complete','utc':now})
msg='\n\n## 第032轮完成 '+now+'\n原4无持物训练事件285个尝试帧连续性审计完成：soup3仅42–43两区域帧在闭合45之前；soup6始终0格，瓷杯10最多2格；白黄杯6五个≥3格帧最长连续2，叠加抬升后仅56/58孤立帧，未满足连续5帧。未知不是失败，原003接受帧为空被精确复现。\n关键范围限制：003有限区域格数来自002已筛选的0.3–2mm窄垫面邻域points_world，不是整个有限体积的所有深度点；当前不能断言完整夹持段缺面。逐帧完整heldcloud未保存，重合只核对原字段和3mm代码来源，未伪称全重算。无新训练/标签/验证测试访问。下一步033先登记训练集窄垫区与完整有限体积来源对照；第二步未验收。\n'
with (P/'docs/STATUS.md').open('a') as f:f.write(msg)
with (B/'迭代索引.md').open('a') as f:f.write('\n- 032：4训练事件缺失持物连续性审计完成，285帧；确认原门槛不通过与窄垫区来源限制，无新标签/训练/验证测试。\n')
files={str(p.relative_to(P)):sha(p) for p in R.rglob('*') if p.is_file() and p.name!='completed_locked.json' and p.suffix not in ['.log','.pyc'] and '__pycache__' not in p.parts}
save(R/'completed_locked.json',files);assert verify(R/'completed_locked.json')==len(files)
print(json.dumps({'complete':True,'files':len(files),'locks':checks},ensure_ascii=True))
