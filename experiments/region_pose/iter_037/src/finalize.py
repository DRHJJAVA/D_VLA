from pathlib import Path
import json,hashlib,datetime,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA');E=R/'experiments/region_pose';I=E/'iter_037'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
assert json.loads((I/'audits/completion.json').read_text())['passed']
s=json.loads((I/'summary.json').read_text());assert not s['full']['quantity_threshold_pass'] and not s['fps']['quantity_threshold_pass']
z=np.load(I/'candidates.npz');rows=[]
for t in [55,57,60]:
 near=z[f'distance_{t}']<=.003;inside=z[f'robust_inside1mm_{t}'];stable=z[f'stable_{t}'];pos=z[f'positive_support_{t}']
 rows.append({'frame':t,'near_supported_region_3mm':int(near.sum()),'inside_all_transforms_1mm':int(inside.sum()),'stable_2mm':int(stable.sum()),'intersection_positive':int(pos.sum()),'near_but_outside_robust':int((near&~inside).sum()),'robust_but_far':int((inside&~near).sum())})
assert rows==json.loads((I/'audits/condition_attrition.json').read_text())['per_frame']
checks={}
for name in ['source_locked.json','input_locked.json','finalize_source_locked.json']:
 d=json.loads((I/name).read_text())
 for k,v in d.items():assert sha(Path(k))==v
 checks[name]=len(d)
d=json.loads((E/'iter_036/completed_locked.json').read_text())
for k,v in d.items():assert sha(R/k)==v
checks['old036']=len(d)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
save(I/'待人工判断.json',{'training_branch':'stopped_insufficient_distinct_positive_query_sources','full_positive':1,'fps_positive':1,'required_positive':3,'not_grasp_failure':True,'not_all_negative_episode':True,'pending':['腕相机边缘来源真实身份','侧向示范数据范围'],'rest_no_question':True,'stage2_approved':False})
save(I/'audits/final_integrity.json',{'passed':True,'checks':checks,'attrition_independent_reconstruction':True,'training_added':False,'val_test_access':False})
save(I/'pre_validation_freeze.json',{'utc':now,'purpose':'freeze this training-only diagnostic before any separately registered old-validation source audit','files':{str(p.relative_to(R)):sha(p) for p in [I/'设计方案.md',I/'config.json',I/'src/audit_candidates.py',I/'summary.json',I/'candidates.npz',I/'audits/completion.json']}})
save(I/'stage_state.json',{'stage':'complete','utc':now})
with (R/'docs/STATUS.md').open('a') as f:f.write('\n\n第037轮完成（'+now+'）：白黄杯6查询36及55/57/60，完整4071点仅1正/3486负/584未知，FPS2048为1/1726/321。未达到3正来源门槛，训练分支停止，未知不当全负。025原AST与036留一独立复算target/known相同、置信差≤2.98e−8。无训练/追踪/验证测试。下一步038先独立登记对缺失旧验证tomato4同冻结033来源范围诊断，第二步未验收。\n')
with (E/'迭代索引.md').open('a') as f:f.write('\n- 037：白黄杯6稳定监督可用性审计完成；完整/FPS都仅1稳定正查询来源，未过3点门槛，不加训练、不作全负事件；见iter_037/结果与下一步.md。\n')
files=[p for p in I.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['completed_locked.json','finalize.log']]
lock={str(p.relative_to(R)):sha(p) for p in sorted(files)};save(I/'completed_locked.json',lock)
assert all(sha(R/k)==v for k,v in lock.items())
print(json.dumps({'complete':True,'locked_files':len(lock)}))
