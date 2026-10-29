from pathlib import Path
import json,numpy as np,hashlib,datetime
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/contact_frames/cf005';C=B.parent/'cf003';D=B.parent/'cf004'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
oldpage=R/'docs/CF003_接触网络优化对照.html'
s=oldpage.read_text();i=s.index('const clips=')+len('const clips=');j=s.index(",labels=['接触候选'",i);clips=json.loads(s[i:j])
cfg=json.loads((B/'config.json').read_text())
verification=[]
for c in clips:
 ep=c['id']
 if ep=='white_yellow_mug_demo_5':c['human_negative']=[44,52]
 for base,arm,title in [(D,'state_dropout','CF004 · 状态分支随机屏蔽'),(B,'all_train_frames','CF005 · 完整训练帧覆盖')]:
  d=np.load(base/('evaluation_'+arm)/ep/'artifacts/dataset.npz')
  a=np.load(base/('evaluation_'+arm)/ep/'artifacts/frame_predictions.npz')
  rows=[]
  for t in range(c['n']):
   pr=a['probabilities'][:,t];state=0 if a['contact_candidate'][t] else 1 if a['observed_separation_candidate'][t] else 2
   known=float(d['weight'][t])>0;target=float(d['target'][t]) if known else None
   if state==0:reason='三个模型均高分，几何估计可用。'
   elif state==1:reason='三个模型均低分，并有可用的几何分离证据。'
   elif not bool(d['state_valid'][t]):reason='本帧状态不可用，保留未知。'
   elif not bool(d['geometry_reliable'][t]):reason='几何证据不足，保留未知；高分不替代证据。'
   else:reason='模型分数或种子一致性不足，保留未知。'
   if state==0 and target is not None and target<=.2:reason='模型高分，但与低几何弱目标冲突（弱目标非真值）。'
   rows.append({'frame':t,'time':t/c['fps'],'mean':float(pr.mean()),'scores':pr.astype(float).tolist(),'state':state,'reason':reason,'weak_target':target})
  counts=[sum(r['state']==k for r in rows) for k in range(3)]
  c['models'].append({'name':title,'rows':rows,'counts':counts})
  verification.append({'episode':ep,'arm':arm,'frames':len(rows),'counts':counts,'prediction_hash':sha(base/('evaluation_'+arm)/ep/'artifacts/frame_predictions.npz')})
template=(B/'src/comparison_template.html').read_text()
page=template.replace('__CLIPS__',json.dumps(clips,ensure_ascii=False,separators=(',',':'))).replace('__RESULTS__',json.dumps(json.loads((B/'summary.json').read_text()),ensure_ascii=False,separators=(',',':')))
out=R/'docs/CF005_接近与接触预测优化对照.html';out.write_text(page)
save(B/'audits/display.json',{'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),'source_page_hash':sha(oldpage),'output_page_hash':sha(out),'rows':verification,'videos':'unchanged remote data-URI bytes reused from CF003, no download','review_negative':[44,52],'review_is_not_training':True})
save(B/'display_locked.json',{str(out):sha(out),str(B/'src/comparison_template.html'):sha(B/'src/comparison_template.html'),str(B/'src/build_comparison.py'):sha(B/'src/build_comparison.py')})
print('comparison_page_ready')
