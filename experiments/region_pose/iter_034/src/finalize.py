import json,hashlib,traceback
from pathlib import Path
from datetime import datetime,timezone
B=Path('/mnt/nvme1/drh/D_VLA'); R=B/'experiments/region_pose/iter_034'; V=B/'experiments/region_pose/reviews/iter_034'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def verify(p):
 d=json.loads(p.read_text());bad=[]
 for s,h in d.items():
  q=Path(s);q=q if q.is_absolute() else B/q
  if not q.is_file() or sha(q)!=h:bad.append(s)
 assert not bad, 'lock mismatch '+p.name
 return len(d)
def main():
 assert not (R/'completed_locked.json').exists()
 locks=[R/'input_locked.json',R/'source_locked.json',R/'source_clean_addendum.json',R/'source_finalize_addendum.json',B/'experiments/region_pose/iter_033/completed_locked.json',V/'immutable_locked.json',V/'display_locked.json',V/'clean_zoom_locked.json']
 checks={str(p.relative_to(B)):verify(p) for p in locks}
 m=json.loads((R/'manifest.json').read_text()); assert len(m)==23
 assert len(list((R/'sources').glob('*.json')))==46
 assert len(list((V/'images').glob('*.png')))==101
 assert len(json.loads((R/'input_slices_locked.json').read_text()))==184
 a=json.loads((R/'audits/completion_audit.json').read_text())
 assert a['selection_independent_equal'] and a['source_pixel_rgb_depth_equal']
 assert not a['validation_read'] and not a['test_read']
 browser=json.loads((R/'audits/browser_review.json').read_text());assert browser['unique_images_checked']==47
 now=datetime.now(timezone.utc).isoformat()
 summary={'iteration':34,'episodes':5,'attempts':6,'selected_frames':23,'camera_records':46,'initial_images':85,'additional_clean_zoom_images':16,'total_images':101,'slice_hashes':184,'browser_checked_unique_images':47,'max_pixel_error':a['max_pixel_error'],'max_depth_error_m':a['max_depth_error_m'],'new_labels':False,'training':False,'tracking':False,'validation_read':False,'test_read':False,'stage2_approved':False,'finding':'白黄杯新增候选依赖腕相机边界附近细条；逐点身份未知，不能直接训练。空候选不是整段失败。','next':'独立登记035白黄杯6冻结跨时对应可用性诊断。'}
 save(R/'summary.json',summary)
 save(R/'audits/final_integrity.json',{'utc':now,'locks_checked':checks,'mismatches':0,'counts':summary,'source_and_inputs_unchanged':True,'original_bad_link_page_preserved':True,'working_display_page':'docs/第034轮_训练有限体积来源查看_查看版.html','clean_page':'docs/第034轮_来源无编号放大.html','finalize_log_excluded':True})
 save(R/'stage_state.json',{'stage':'complete','utc':now})
 status='\n\n## 第034轮完成 '+now+'\n训练来源查看完成：5训练episode/6尝试/23固定帧/46相机来源记录，101图。白黄杯6新窗口主要依赖腕相机边界附近细条，身份与真实区域尚未确认；空候选不等于抓取失败。没有新标签、追踪、训练或验证测试访问。报告 experiments/region_pose/iter_034/结果与下一步.md；工作页 第034轮_训练有限体积来源查看_查看版.html、无编号放大页。第二步未验收。下一步先登记035白黄杯6冻结跨时对应诊断，不直接训练28帧。\n'
 with (B/'docs/STATUS.md').open('a',encoding='utf-8') as f:f.write(status)
 with (B/'experiments/region_pose/迭代索引.md').open('a',encoding='utf-8') as f:f.write('\n- 034：训练有限体积来源查看完成。23固定帧、101图，新增边界来源身份仍未知，无新监督/训练；见iter_034/结果与下一步.md。\n')
 completed={str(p.relative_to(B)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.name not in ['completed_locked.json','finalize.log'] and '__pycache__' not in p.parts}
 save(R/'completed_locked.json',completed)
 assert verify(R/'completed_locked.json')==len(completed)
 print(json.dumps({'stage':'complete','locked_files':len(completed),'images':101,'lock_mismatches':0},ensure_ascii=False))
try:main()
except Exception as e:
 (R/'finalize_traceback.txt').write_text(traceback.format_exc())
 save(R/'finalize_failure.json',{'type':type(e).__name__,'message':str(e)})
 print(json.dumps({'error_type':type(e).__name__,'detail':'remote finalize_failure.json'}))
 raise SystemExit(1)
