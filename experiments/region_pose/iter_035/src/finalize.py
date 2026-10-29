from pathlib import Path
import json,hashlib,subprocess,traceback
from datetime import datetime,timezone
B=Path('/mnt/nvme1/drh/D_VLA');E=B/'experiments/region_pose';R=E/'iter_035';V=E/'reviews/iter_035'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for a in iter(lambda:f.read(8388608),b''):h.update(a)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def check(p,root=B):
 d=json.loads(p.read_text())
 for s,h in d.items():
  q=Path(s);q=q if q.is_absolute() else root/q
  assert sha(q)==h,'lock mismatch '+str(p)
 return len(d)
try:
 assert not (R/'completed_locked.json').exists()
 locks=['source_locked.json','input_locked.json','seed_locked.json','source_runner_addendum.json','source_audit_addendum.json','source_review_addendum.json','source_finalize_addendum.json']
 counts={n:check(R/n) for n in locks}
 for n in ['iter_022','iter_033','iter_034']:counts[n]=check(E/n/'completed_locked.json')
 counts['train_locked']=check(R/'train/locked.json',R/'train')
 counts['review_locked']=check(V/'immutable_locked.json')
 assert json.loads((R/'audits/completion.json').read_text())['supported_endpoints']==[55,57,60]
 assert json.loads((R/'audits/method_identity.json').read_text())['whole_script_ast_equal_after_only_registered_8_assignment_adapters']
 assert json.loads((R/'audits/browser_review.json').read_text())['all_six_images_checked']
 assert len(list((V/'images').glob('*.png')))==6
 now=datetime.now(timezone.utc).isoformat()
 save(R/'audits/final_integrity.json',{'utc':now,'locks_checked':counts,'mismatches':0,'no_new_labels':True,'no_scoring_network_training':True,'no_validation_or_test_access':True,'source_and_input_hashes_unchanged':True,'finalize_log_excluded':True})
 save(R/'stage_state.json',{'stage':'complete','utc':now})
 with (B/'docs/STATUS.md').open('a',encoding='utf-8') as f:f.write('\n\n## 第035轮完成 '+now+'\n白黄杯6严格夹前36→55/57/60的冻结对应通过；65比例27/46=0.587未过0.6，首次失败截断，82虽单独通过只作诊断。91采样点，1训练事件，不是成功率。没有新标签或评分训练。6图来源页：第035轮_白黄杯6冻结跨时对应.html；报告experiments/region_pose/iter_035/结果与下一步.md。下一步先登记036，仅55/57/60留一稳定性与实际有限区域回映射。第二步未验收。\n')
 with (E/'迭代索引.md').open('a',encoding='utf-8') as f:f.write('\n- 035：白黄杯6冻结跨时对应诊断完成。55/57/60通过、65失败截断、82仅诊断；无新标签/训练，详见iter_035/结果与下一步.md。\n')
 files={str(p.relative_to(B)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.name not in ['completed_locked.json','finalize.log'] and '__pycache__' not in p.parts}
 save(R/'completed_locked.json',files);check(R/'completed_locked.json')
 print(json.dumps({'stage':'complete','locked_files':len(files),'lock_mismatches':0}))
except Exception as e:
 (R/'finalize_traceback.txt').write_text(traceback.format_exc());save(R/'finalize_failure.json',{'type':type(e).__name__,'message':str(e)});print('finalization failed; safe details retained remote');raise SystemExit(1)
