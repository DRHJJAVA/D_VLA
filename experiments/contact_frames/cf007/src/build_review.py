import pathlib,json,copy,numpy as np,hashlib
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA');RUN=ROOT/'experiments/contact_frames/cf007'
s=(ROOT/'docs/CF005_多次尝试视频接触预测查看.html').read_text()
a=s.index('const clips=')+len('const clips=');clips,n=json.JSONDecoder().raw_decode(s[a:])
for c in clips:
 old=c['models'][0];old['name']='CF005 世界运动辅助（原模型）'
 new=copy.deepcopy(old);new['name']='CF007 相对运动辅助（新训练）'
 z=np.load(RUN/'evaluation_all_train_frames'/c['id']/'artifacts/frame_predictions.npz');d=np.load(RUN/'evaluation_all_train_frames'/c['id']/'artifacts/dataset.npz')
 for t,row in enumerate(new['rows']):
  row.update(mean=float(z['mean'][t]),scores=z['probabilities'][:,t].tolist(),state=0 if z['contact_candidate'][t] else 1 if z['observed_separation_candidate'][t] else 2)
  if row['state']==0:row['reason']='三种子均高分且几何估计可用；仍可能把近距离/抓空计入，不能视为有效接触真值。'
  elif row['state']==1:row['reason']='三种子均低分并有原几何分离证据。'
  elif not d['geometry_reliable'][t]:row['reason']='几何证据不足，保留未知。'
  else:row['reason']='模型输出尚未满足原候选/分离条件，保留未知。'
 new['counts']=[int(z['contact_candidate'].sum()),int(z['observed_separation_candidate'].sum()),int(z['unknown'].sum())]
 c['models']=[old,new]
 if c['id']=='tomato_sauce_demo_3':c['note']+=' 本次针对抓空的网络对照未改善：第一次52–69，原/新均18候选，均分0.981→0.986；第二次94–153仍5候选55未知。不按本视频改阈值，不采用CF007替换原模型。'
s=s[:a]+json.dumps(clips,ensure_ascii=False,separators=(',',':'))+s[a+n:]
s=s.replace('CF005 · 多次尝试视频接触预测','CF007 · 抓空问题网络优化对照')
s=s.replace('两段未参与接触网络训练的完整视频，均有两次闭合尝试。直接使用冻结CF005预测，没有重新训练、调阈值或按旧尝试标签改分。','实际完成3种子×600步对照：只把运动辅助目标从世界位移改成物体相对夹爪位移。数据、主监督、初始化、步数与网络结构不变。结果：未解决demo3抓空误报，CF007不替换CF005。')
s=s.replace('本页没有新增人工接触标签；这些案例在早期区域工程中看过，不是盲测。','用户指出的抓空仅作复核，没有用来训练。两个案例未参与接触网络训练，但属于已看过的开发复核，不是盲测。')
s=s.replace('白线：CF005三个种子的平均分','白线：所选模型三个种子的平均分')
s=s.replace('模型高分与低弱目标冲突会单独提示。','CF007主比较保持原候选规则，不混入CF006的冲突修正。')
comparison=json.loads((RUN/'comparison.json').read_text())['rows']
table='<section style="margin:24px 0;padding:20px;border:1px solid #344155;border-radius:14px"><h2>固定复核结果：本轮未解决目标问题</h2><table><tr><th>视频</th><th>CF005 候选/分离/未知</th><th>CF007 候选/分离/未知</th></tr>'
for row in comparison:
 if not row['training'] or row['episode']=='tomato_sauce_demo_7':
  def counts(a):return '/'.join(str(a[k]) for k in ['candidate','separation','unknown'])
  table+='<tr><td>'+row['episode']+('（训练案例）' if row['training'] else '')+'</td><td>'+counts(row['cf005'])+'</td><td>'+counts(row['cf007'])+'</td></tr>'
table+='</table><p>候选数量变化不是准确率。demo3第一次抓空仍18/18候选；瓷杯demo9复核区间仍19候选21未知；白黄杯demo5的44–52仍1候选1分离7未知。新模型不作默认替换。</p><p><a href="CF006_抓空误报与证据冲突对照.html">另看CF006：同一模型的分离冲突修正（不是网络训练收益）</a></p></section>'
s=s.replace('</body>',table+'</body>')
out=ROOT/'docs/CF007_抓空问题网络优化对照.html';out.write_text(s)
for c in clips:
 z=np.load(RUN/'evaluation_all_train_frames'/c['id']/'artifacts/frame_predictions.npz')
 assert np.allclose([r['scores'] for r in c['models'][1]['rows']],z['probabilities'].T,atol=0,rtol=0)
 assert [r['state'] for r in c['models'][1]['rows']]==np.where(z['contact_candidate'],0,np.where(z['observed_separation_candidate'],1,2)).tolist()
(RUN/'audits/display.json').write_text(json.dumps({'page':str(out),'predicted_rows':sum(c['n'] for c in clips),'same_video_bytes':True,'scores_match':True,'states_match':True,'claim':'targeted trial unsuccessful; not adopted'},ensure_ascii=False,indent=2))
print(json.dumps({'page':str(out),'size':out.stat().st_size}))
