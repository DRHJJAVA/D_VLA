from pathlib import Path
import json,hashlib,datetime
P=Path('/mnt/nvme1/drh/D_VLA');R=P/'experiments/region_pose/iter_033'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
assert (R/'audits/independent_reconstruction.json').exists()
assert not (R/'failure_v3.json').exists() and not (R/'audit_failure_v2.json').exists()
alllocks=['source_locked.json','source_addendum_033_1.json','source_addendum_033_2.json','source_audit_addendum.json','source_audit_addendum_033_3.json','source_finalize_addendum.json','input_locked.json']
checks={}
for name in alllocks:
 lock=json.loads((R/name).read_text())
 assert all(sha(Path(k))==v for k,v in lock.items()),name
 checks[name]=len(lock)
save(R/'audits/resolved_failures.json',{'failure.json':'resolved by scoped main AST selection; no data processing before initial failure','failure_v2.json':'resolved by preregistered filter distinction; no completed case before stop','audit_failure.json':'resolved by precise inverse of stored float32 camera rotation; no threshold changed','active_failure':None})
save(R/'待人工判断.json',{'stage2_passed':False,'questions_deferred_until_after_rest':True,'pending':['白黄杯6新增有限体积来源的真实物体身份与遮挡，需要来源查看和独立跨时证据，暂不提升标签','既有数据缺明显侧向示范，另需确定采集/数据范围，不能用重复帧制造方向','旧验证tomato4尚未在033读取，若需同规则审计另版登记'],'paused_branches':['未经来源身份与留一稳定性的新监督提升','缺侧向证据的上方/侧向能力结论']})
save(R/'audits/final_integrity.json',{'locks':checks,'new_labels':False,'new_training':False,'validation_read':False,'test_read':False,'stage2_passed':False})
save(R/'stage_state.json',{'stage':'complete','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_labels':False,'stage2_passed':False})
entry='\n\n033 已完成：训练5段/6尝试窄垫与全有限体积对照。白黄杯6新候选55–82共28连续帧，tomato3首次仍0/后次24；其余缺失三段未恢复。源点复现、独立像素/深度/连续段和哈希核对通过。仅诊断无标签/训练，下一步034训练来源查看与身份可观测性，第二步未验收。\n'
for p in [P/'docs/STATUS.md',P/'experiments/region_pose/迭代索引.md']:
 with p.open('a',encoding='utf-8') as f:f.write(entry)
paths=sorted(p for p in R.rglob('*') if p.is_file() and p.name!='completed_locked.json' and '__pycache__' not in p.parts and p.suffix!='.pyc')
save(R/'completed_locked.json',{str(p.relative_to(P)):sha(p) for p in paths})
print(json.dumps({'complete':True,'locked_files':len(paths),'checks':checks}))
