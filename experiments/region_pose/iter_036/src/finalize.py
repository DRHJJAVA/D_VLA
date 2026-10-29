from pathlib import Path
import json,hashlib,datetime
R=Path('/mnt/nvme1/drh/D_VLA');E=R/'experiments/region_pose';I=E/'iter_036'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
assert json.loads((I/'audits/completion.json').read_text())['passed']
assert json.loads((I/'audits/original024_independent_reconstruction.json').read_text())['passed']
checks={}
for name in ['source_locked.json','input_locked.json','source_addendum_036_1.json','independent_source_locked.json','partial_v1_locked.json','finalize_source_locked.json']:
 d=json.loads((I/name).read_text())
 for k,v in d.items():assert sha(Path(k))==v
 checks[name]=len(d)
for n in ['033','035']:
 d=json.loads((E/f'iter_{n}/completed_locked.json').read_text())
 for k,v in d.items():assert sha(R/k)==v
 checks['old'+n]=len(d)
save(I/'audits/resolved_036_1.json',{'resolved':True,'failure':'failure.json','cause':'eroded seed mask used as full query; diagnostic extraction also omitted original int64 conversion','correction':'original035 causal full query with int64 pixels, exact 4071-point equality','old_partial_loo_unchanged':True,'old_data_modified':False})
save(I/'待人工判断.json',{'stage2_passed':False,'pending':['腕相机杯壁边缘与夹指遮挡附近来源真实身份仍不能由数值稳定证明','当前示范方向不足，侧向数据范围另定'],'rest_no_question':True,'unaffected_next':'037单训练事件稳定监督候选可用性审计，另版登记，不直接训练'})
save(I/'audits/final_integrity.json',{'passed':True,'checks':checks,'no_new_labels':True,'no_training':True,'no_val_test':True,'stage2_approved':False})
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
save(I/'stage_state.json',{'stage':'complete','utc':now})
p=R/'docs/STATUS.md'
with p.open('a') as f:f.write('\n\n第036轮完成（'+now+'）：仅白黄杯6查询36及55/57/60，101有效留一拟合；80实际区域观测、39距查询≤3mm、36跨时支持观测、22真实1mm代表（3主/19腕）。区域最大留一位移≤0.427mm；数值稳定不是身份真值。原024重建差0，输入与旧033/035锁通过。无新目标/追踪/评分训练/验证测试。036.1查询适配错误已解决并保留历史。下一步037先独立登记监督可用性审计，不能直接训练，第二步未验收。\n')
p=E/'迭代索引.md'
with p.open('a') as f:f.write('\n- 036：白黄杯6首次失败前候选的留一与实际区域回映射审计已完成；55/57/60共101有效拟合、22跨时支持真实格，来源身份仍未知，未训练。见iter_036/结果与下一步.md。\n')
files=[p for p in I.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['completed_locked.json','finalize.log']]
lock={str(p.relative_to(R)):sha(p) for p in sorted(files)}
save(I/'completed_locked.json',lock)
assert all(sha(R/k)==v for k,v in lock.items())
print(json.dumps({'complete':True,'locked_files':len(lock),'utc':now}))
