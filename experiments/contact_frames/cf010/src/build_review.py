from pathlib import Path
import json,ast,re,numpy as np,hashlib
B=Path('/mnt/nvme1/drh/D_VLA');BASE=B/'experiments/contact_frames';R=BASE/'cf010'
t=ast.parse((BASE/'cf008/src/build_review.py').read_text());env={'ROOT':B,'json':json,'re':re}
exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='get'],type_ignores=[]),'get_old_media','exec'),env)
clips=env['get']('CF008_接近与有效作用证据对照.html')
ranges=json.loads((R/'review_ranges.json').read_text())
comp=json.loads((R/'comparison.json').read_text())['rows']
models=[('CF005','CF005 · 9帧接近'),('CF008','CF008 · 9帧作用证据'),('CF009B','CF009 B · 延续正证据'),('CF010','CF010 · 整段历史')]
for c in clips:
 ep=c['id'];c['human']=ranges['positive'][ep];c['human_negative']=ranges['negative_explicit_only'].get(ep)
 positive=next(z for z in comp if z['episode']==ep and z['region']=='user_contact')
 c['summary_note']='用户确认 '+str(c['human'][0])+'–'+str(c['human'][1])+'，仅复核。区间内预测接触（CF005/008/009B/010）：'+'/'.join(str(positive['results'][k]['predicted_contact']) for k,_ in models)+'；总'+str(positive['results']['CF010']['frames'])+'帧。'
 if c['human_negative']:
  neg=next(z for z in comp if z['episode']==ep and z['region']=='user_noncontact_or_failed')
  c['summary_note']+=' 明确失败/未接触'+str(c['human_negative'])+'误报：'+'/'.join(str(neg['results'][k]['predicted_contact']) for k,_ in models)+'。'
 if ep=='tomato_sauce_demo_7':c['summary_note']+=' 本片参与训练，不能称泛化。'
 c['models']=[]
 for k,title in models:
  if k=='CF005':root=BASE/'cf005_frozen_review_003' if ep in ['tomato_sauce_demo_3','alphabet_soup_demo_1'] else BASE/'cf005/evaluation_all_train_frames'
  elif k=='CF008':root=BASE/'cf008/evaluation'
  elif k=='CF009B':root=BASE/'cf009/persistent_evidence/evaluation'
  else:root=R/'evaluation'
  a=root/ep/'artifacts';p=np.load(a/'frame_predictions.npz');d=np.load(a/'dataset.npz')
  reliable=d['geometry_reliable']&d['state_valid'];sep=reliable&(d['distance']-2*d['sigma']>.006)
  pred=(p['probabilities'].min(0)>=.8)&~sep
  rows=[]
  for frame in range(c['n']):
   state=0 if pred[frame] else 1 if p['observed_separation_candidate'][frame] else 2
   obs='当前可见证据：可靠' if reliable[frame] else '当前可见证据：不足（不直接抹掉模型的持续接触预测）'
   if sep[frame]:obs='当前有明确分离证据；若高分则为冲突未知'
   reason=obs+'。原严格候选：'+('是' if p['contact_candidate'][frame] else '否')+'。预测接触不等于物理真值。'
   rows.append({'frame':frame,'time':frame/c['fps'],'mean':float(p['probabilities'][:,frame].mean()),'scores':p['probabilities'][:,frame].tolist(),'state':state,'reason':reason,'proximity':float(p['proximity_probabilities'][:,frame].mean()) if 'proximity_probabilities' in p.files else None})
  c['models'].append({'name':title,'rows':rows,'counts':[sum(z['state']==v for z in rows) for v in range(3)]})
template=(BASE/'cf009/src/viewer_template.html').read_text()
template=template.replace('CF009 · 接触证据延续对照','CF010 · 整段历史与观测证据分离')
template=template.replace('两组同网络对照，各3个种子、每种子600步。A：不把“缺乏运动支持”当负例；B：另用可靠相对位姿延续已有正证据。用户范围只用于复核，不参与训练。分数不是校准的物理接触概率。','CF010实际完成3种子各600步，监督完全复用CF008，只把GRU改为整段连续历史。六组最新人工范围仅复核。所有模型统一区分预测状态与可见证据；不能把取消观测门槛当网络提升。')
template=template.replace("['接触候选','观测分离候选','未知']","['预测接触','观测分离','未知']")
template=template.replace('白线：所选模型三种子均分；黄：候选，蓝：分离，灰：未知。','白线：所选模型三种子均分；黄：预测接触，蓝：观测分离，灰：未知。')
template=template.replace('CF005 候选/分离/未知','CF005 预测/分离/未知').replace('<th>CF009 A</th><th>CF009 B</th>','<th>CF009 B</th><th>CF010</th>')
template=template.replace('候选变多不等于更准确。同时检查失败尝试误报与有效接触漏检；未知没有转成失败负例。','以下四列统一使用模型预测状态。逐帧右栏单列当前可见证据及原严格候选；不是旧页面严格候选的同一统计口径。人工正段外不自动当负例。')
template=template.replace('同一训练片的局部证据延续不等于新增独立接触事件，也不能保证滑动、静态用力或关节物体全部识别。网络使用9帧窗口；追踪预处理使用整段视频，属于离线分析。所有模型、视频和报告保留远程。','CF010的GRU连续使用当前帧及全部历史；已通过相同长度未来扰动不改变此前输出的检查。原RGB-D追踪预处理仍使用整段视频，所以系统仍是离线分析。旧模型只读9帧。缺可见证据不等于不接触；预测高分也不等于接触真值。全部产物远程。')
template=template.replace('全视频：候选','全视频：预测接触')
headline='整段历史没有带来整体改善：同一预测口径，CF008→CF010，demo0正段0→6，demo1正段19→9，demo3失败段误报2→4。六组人工范围未参与训练。本页分开展示预测状态与可见证据，避免遮挡被误读成模型认为不接触；不替换默认模型。'
out=B/'docs/CF010_整段历史与接触状态对照.html'
out.write_text(template.replace('__CLIPS__',json.dumps(clips,ensure_ascii=False)).replace('__HEADLINE__',json.dumps(headline,ensure_ascii=False)))
(R/'audits/display.json').write_text(json.dumps({'page':str(out),'clips':6,'models':4,'latest_human_ranges_display_only':True,'same_prediction_policy_all_models':True,'separate_observation_support':True,'media_unchanged':True,'page_sha256':hashlib.sha256(out.read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
print(json.dumps({'page':str(out),'clips':6}))
