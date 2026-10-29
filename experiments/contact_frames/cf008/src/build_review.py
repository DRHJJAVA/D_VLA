import pathlib,json,copy,numpy as np,hashlib
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA');R=ROOT/'experiments/contact_frames/cf008'
def get(name):
 s=(ROOT/'docs'/name).read_text();i=s.index('const clips=')+len('const clips=');return json.JSONDecoder().raw_decode(s[i:])[0]
clips=get('CF005_多次尝试视频接触预测查看.html')
more=get('CF005_接近与接触预测优化对照.html')
for c in more:c['models']=[c['models'][-1]]
clips+=more
notes={'tomato_sauce_demo_3':'用户确认49–74为失败场景，仅用于复核。旧24候选，新2候选（65、66仍误报）、24未知；不是26帧都已识别为失败。第二次94–153候选5→1，未知55→59，仍有漏检代价。','alphabet_soup_demo_1':'两次闭合尝试仅导航。第一次46–64候选3→0，第二次80–158候选11→18；未有逐帧人工真值，不当准确率。','white_yellow_mug_demo_5':'用户指出44–52尚未接触，仅用于复核。原1候选1分离7未知，新0候选1分离8未知。','alphabet_soup_demo_0':'明显退化：候选35→0，更多成为未知；不能只报告demo3误报减少。','porcelain_mug_demo_9':'用户确认56–95接触范围，仅用于复核。原19候选21未知，新17候选23未知，未标分离但漏检增加。','tomato_sauce_demo_7':'本片参与训练。用户确认52–132为接触，仅用于复核。原50候选31未知，新32候选49未知，不能称泛化；未用用户范围训练。'}
for c in clips:
 ep=c['id'];old=c['models'][0];old['name']='CF005 原接近兼容分数';new=copy.deepcopy(old);new['name']='CF008 有效作用支持分数'
 z=np.load(R/'evaluation'/ep/'artifacts/frame_predictions.npz')
 for t,row in enumerate(new['rows']):
  row.update(mean=float(z['mean'][t]),scores=z['probabilities'][:,t].tolist(),proximity=float(z['proximity_probabilities'][:,t].mean()),state=0 if z['contact_candidate'][t] else 1 if z['observed_separation_candidate'][t] else 2)
  row['reason']=['三种子作用支持均高且几何可用；仍是弱证据候选，不是已确认成功。','接近辅助头均低且有明确分离几何证据。','作用支持或观测不足，保留未知；低支持不等于已证实失败。'][row['state']]
 new['counts']=[int(z['contact_candidate'].sum()),int(z['observed_separation_candidate'].sum()),int(z['unknown'].sum())]
 c['models']=[old,new];c['note']=notes[ep];c['summary_note']=notes[ep]
 if ep=='tomato_sauce_demo_3':c['human_negative']=[49,74]
 if 'attempts' not in c:
  p=ROOT/'experiments/region_pose/iter_003/evidence/motion'/(ep+'.json')
  c['attempts']=[[a['start_frame'],a['end_frame']] for a in json.loads(p.read_text())['attempts']]
 for t,row in enumerate(new['rows']):
  assert row['scores']==z['probabilities'][:,t].tolist()
out=ROOT/'docs/CF008_接近与有效作用证据对照.html'
out.write_text((R/'src/viewer_template.html').read_text().replace('__CLIPS__',json.dumps(clips,ensure_ascii=False,separators=(',',':'))))
(R/'audits/display.json').write_text(json.dumps({'page':str(out),'clips':len(clips),'frames':sum(c['n'] for c in clips),'scores_and_states_from_saved_predictions':True,'video_bytes_reused':True,'human_ranges_display_only':True},ensure_ascii=False,indent=2))
print({'page':str(out),'clips':len(clips)})
