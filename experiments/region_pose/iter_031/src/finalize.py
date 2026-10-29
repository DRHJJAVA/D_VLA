from pathlib import Path
import json, hashlib, datetime, traceback, numpy as np
P=Path('/mnt/nvme1/drh/D_VLA'); R=P/'experiments/region_pose/iter_031'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def verify(p,base):
 x=json.loads(p.read_text())
 for k,v in x.items():
  q=base/k
  assert q.is_file(), 'missing locked file'
  assert sha(q)==v,'locked file mismatch: '+k
 return len(x)
try:
 checks={}
 for name in ['source_locked.json','input_locked.json','preflight_locked.json','baseline_input_locked.json','baseline_source_locked.json','evaluation_source_locked.json','review_source_locked.json','finalize_source_locked.json']:
  checks[name]=verify(R/name,P)
 checks['training_inputs_locked.json']=verify(R/'training_inputs_locked.json',R)
 V=P/'experiments/region_pose/reviews/iter_031'
 checks['review_immutable']=verify(V/'immutable_locked.json',P)
 old=json.loads((P/'experiments/region_pose/iter_029/completed_locked.json').read_text())
 target=P/'experiments/region_pose/iter_029/audits/review_image_slices.json'
 candidates=[str(target.relative_to(P)),str(target.relative_to(P/'experiments/region_pose/iter_029'))]
 found=[k for k in candidates if k in old]
 assert len(found)==1 and sha(target)==old[found[0]],'review metadata source mismatch'
 save(R/'review_metadata_input_addendum.json',{str(target.relative_to(P)):sha(target)})
 # Never descend sample/graph symlinks; inputs remain parent immutable products.
 x=json.loads((R/'evaluation/summary.json').read_text())
 report=(R/'结果与下一步.md').read_text()
 report+='\n## 全事件与提前量冻结指标明细\n以下各行同一事件内查询等权；不是独立抓取重复。train/val全量逐点预算详见evaluation。\n'
 for a in x:
  report+='\n### '+a['run']+' / '+a['checkpoint']+'\n\n|集合|事件|AP|原头loss|候选头loss|覆盖10/20/30|20%未知|20%已知负|\n|---|---|---:|---:|---:|---|---:|---:|\n'
  for split,agg in [('train',a['all_training_diagnostic']),('val',a['validation'])]:
   rows=agg['rows']
   for ep in sorted(set(t['episode'] for t in rows)):
    rr=[t for t in rows if t['episode']==ep]
    mean=lambda k:float(np.mean([t[k] for t in rr]))
    report+='|'+split+'|'+ep+'|'+f"{mean('ap'):.6f}|{mean('loss'):.6f}|{mean('candidate_head_loss'):.6f}|"+'/'.join(f"{mean('coverage'+str(k)):.6f}" for k in [10,20,30])+f"|{mean('all_unknown_fraction20'):.6f}|{mean('all_known_negative_fraction20'):.6f}|\n"
   report+='\n'+split+'提前量表：\n\n|提前量|AP|原头loss|候选头loss|\n|---:|---:|---:|---:|\n'
   for off in [15,10,5,1]:
    rr=[t for t in rows if t['offset']==off]
    report+='|'+str(off)+'|'+'|'.join(f"{np.mean([t[k] for t in rr]):.6f}" for k in ['ap','loss','candidate_head_loss'])+'|\n'
 (R/'结果与下一步.md').write_text(report)
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 save(R/'audits/final_integrity.json',{'utc':now,'checks':checks,'old_029_review_slice_metadata_unchanged':True,'samples_graphs_readonly_parent_symlinks':True,'new_inferences':416,'test_read':False,'stage2_accepted':False})
 save(R/'stage_state.json',{'stage':'complete','utc':now,'training_finished':True,'audit_finished':True,'prediction_review_finished':True,'stage2_accepted':False})
 msg='\n\n## 第031轮完成 '+now+'\n两个新run各24轮/240步，排序负子集唯一变化为最高logit64已知负；best基础/条件旧验证AP0.422488/0.381453，029扩展对照0.425613/0.447240，无整体收益，不自动替换默认方法。last条件白黄杯训练2/4 AP均1，旧验证仍弱，不称泛化。真实预测页第031轮_真实模型排序负子集对照.html共132图，后台18图检查；第二步待用户审核。\n031.1更正029/030当前几何基线短云点序：训练AP0.119275→0.219665，验证0.206372→0.265695；模型/训练/best不受影响，旧锁保留，详见iter_031/audits/baseline_index_correction.json。下一步032先审计原4无持物训练事件连续支持交集，不扩网/加轮/放宽门槛，休息期判断事项远程记录。\n'
 with (P/'docs/STATUS.md').open('a') as f:f.write(msg)
 with (P/'experiments/region_pose/迭代索引.md').open('a') as f:f.write('\n- 031：排序负子集单因素对照完成；2×240更新，无整体旧验证收益；几何基线点序独立更正，真预测132图，第二步未验收。\n')
 files={}
 for p in R.rglob('*'):
  if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix not in ['.log','.pyc'] and p.name not in ['completed_locked.json','pipeline.lock']:
   files[str(p.relative_to(P))]=sha(p)
 save(R/'completed_locked.json',files)
 assert verify(R/'completed_locked.json',P)==len(files)
 print(json.dumps({'complete':True,'locked_files':len(files),'checks':checks},ensure_ascii=True))
except Exception as e:
 save(R/'audits/finalize_failure.json',{'type':type(e).__name__,'message':str(e)})
 (R/'audits/finalize_traceback_remote.txt').write_text(traceback.format_exc())
 print(json.dumps({'complete':False,'error_type':type(e).__name__,'message':str(e)},ensure_ascii=True))
 raise SystemExit(1)
