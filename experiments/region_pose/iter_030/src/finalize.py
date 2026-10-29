from pathlib import Path
import json,hashlib,datetime
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/region_pose/iter_030'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2))
assert not (P/'completed_locked.json').exists()
src=P/'src/finalize.py'
dump(P/'finalize_source_locked.json',{str(src.relative_to(P)):sha(src)})
counts={}
for name,root in [('input_locked.json',R),('input_addendum_030_2.json',R),('source_locked.json',P),('source_addendum_030_1.json',P),('source_addendum_030_2.json',P),('finalize_source_locked.json',P)]:
 d=json.loads((P/name).read_text())
 assert all(sha(root/k)==v for k,v in d.items()),name
 counts[name]=len(d)
assert len(list((P/'points').rglob('*.npz')))==320
audit=json.loads((P/'audits/completion_audit.json').read_text())
assert audit['original029_metric_max_difference']==0 and not audit['training']
dump(P/'audits/final_integrity.json',{'locks':counts,'all_unchanged':True,'point_files':320,'source_bodies_exported':False,'no_new_prediction_page':True,'test_access':False,'stage2_accepted':False})
dump(P/'stage_state.json',{'stage':'complete','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training':False,'review_required_for_stage2':True})
note='\n\n## 030 训练集损失与固定预算曝光审计完成\n40训练查询×8冻结快照共320组逐点贡献；原Torch损失/解析梯度最大差2.45123e-7/1.45509e-8，未知梯度0，029原指标差0。扩展best tomato1正BCE占各查询头loss约97.6%–98.4%；白黄杯4两best各4时刻倒序负点均未入rank64，但完整负BCE仍有梯度。240步控制/扩展每offset曝光10/6，历史无逐事件loss，不能伪造同曝光中间评估。030.1修正短云arange与027FPS排列的审计假设，训练输入未改。全部远程、未训练/未新验证测试/未验收。下一步031先登记仅rank负子集top64单因素、完整BCE与240步预算不变，预检后执行。见experiments/region_pose/iter_030/结果与下一步.md。\n'
with (R/'docs/STATUS.md').open('a',encoding='utf-8') as f:f.write(note)
with (R/'experiments/region_pose/迭代索引.md').open('a',encoding='utf-8') as f:f.write('\n- 030 已完成：训练冻结损失/梯度/负尾部与曝光审计，10事件40查询320快照，无新训练；[结果](iter_030/结果与下一步.md)。下一步031单因素预登记，第二步未验收。\n')
files=[p for p in P.rglob('*') if p.is_file() and 'logs' not in p.relative_to(P).parts and '__pycache__' not in p.relative_to(P).parts and p.name!='completed_locked.json']
locked={str(p.relative_to(R)):sha(p) for p in files};dump(P/'completed_locked.json',locked)
assert all(sha(R/k)==v for k,v in locked.items())
print(json.dumps({'complete':True,'locked_files':len(locked),'lock_counts':counts}))
