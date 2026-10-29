from pathlib import Path
import ast,json,re,numpy as np,hashlib
B=Path('/mnt/nvme1/drh/D_VLA');R=B/'experiments/contact_frames/cf009'
tree=ast.parse((R.parent/'cf008/src/build_review.py').read_text())
env={'ROOT':B,'json':json,'re':re}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get'],type_ignores=[]),'prior_payload_reader','exec'),env)
clips=env['get']('CF008_接近与有效作用证据对照.html')
notes={
'tomato_sauce_demo_3':'失败49–74：CF008误候选2帧，A/B均回升到18帧（49–66）。本轮未解决抓空；第二次尝试候选1→5。范围仅复核，不作训练标签。',
'alphabet_soup_demo_1':'两尝试边界仅导航。第一次候选0→3；第二次18→35(A)/30(B)，无逐帧真值，不能称准确率。',
'white_yellow_mug_demo_5':'用户确认44–52未接触：CF008零误候选，A/B均误报49、51帧。退化。',
'alphabet_soup_demo_0':'CF008零候选，A恢复46帧，B恢复32帧；CF005为35帧。没有整段逐帧真值，数量恢复不等于准确率提高。',
'porcelain_mug_demo_9':'确认接触56–95：CF008候选17，A/B均19；仍21帧未知。',
'tomato_sauce_demo_7':'训练片，不能当泛化。确认52–132：CF008候选32，A49，B50；未用用户范围训练。'}
for c in clips:
 ep=c['id'];c['summary_note']=notes[ep]
 for arm,title in [('positive_evidence','CF009 A · 未支持不作负'),('persistent_evidence','CF009 B · 延续正证据')]:
  a=R/arm/'evaluation'/ep/'artifacts'
  p=np.load(a/'frame_predictions.npz');d=np.load(a/'dataset.npz')
  rows=[]
  for t in range(c['n']):
   st=0 if p['contact_candidate'][t] else 1 if p['observed_separation_candidate'][t] else 2
   reason=['三种子均高分且通过既有观测质量检查；仍是候选，不是接触真值。','低接近分数且有独立可见分离证据。','分数分歧或观测不足，保留未知，不标失败。'][st]
   rows.append({'frame':t,'time':t/c['fps'],'mean':float(p['mean'][t]),'scores':p['probabilities'][:,t].tolist(),'state':st,'reason':reason,'weak_target':float(d['target'][t]) if d['weight'][t]>0 else None,'proximity':float(p['proximity_probabilities'][:,t].mean())})
  counts=[sum(z['state']==i for z in rows) for i in range(3)]
  c['models'].append({'name':title,'rows':rows,'counts':counts})
headline='本轮两组共完成6次600步训练：demo0恢复候选（0→46/32），但demo3失败49–74误候选2→18，白黄杯未接触段也重现误报。没有同时改善误报和漏检，不替换默认模型；A/B全部结果保留。'
out=B/'docs/CF009_接触证据延续与抓空对照.html'
template=(R/'src/viewer_template.html').read_text()
out.write_text(template.replace('__CLIPS__',json.dumps(clips,ensure_ascii=False)).replace('__HEADLINE__',json.dumps(headline,ensure_ascii=False)))
audit={'page':str(out),'clips':len(clips),'frames':sum(c['n'] for c in clips),'models':4,'saved_predictions_used':True,'old_video_bytes_reused':True,'human_ranges_display_only':True,'page_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
(R/'audits/display.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps({'page':str(out),'clips':len(clips)}))
