import pathlib,json,copy,hashlib,numpy as np
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA');RUN=ROOT/'experiments/contact_frames/cf006'
p=ROOT/'docs/CF005_多次尝试视频接触预测查看.html';s=p.read_text()
a=s.index('const clips=')+len('const clips=');clips,n=json.JSONDecoder().raw_decode(s[a:])
for c in clips:
 old=c['models'][0];old['name']='CF005 原候选（冻结模型）'
 new=copy.deepcopy(old);new['name']='CF006 冲突修正（同一模型）'
 z=np.load(RUN/c['id']/'artifacts/frame_predictions.npz');d=np.load(RUN/c['id']/'artifacts/dataset.npz')
 for t,row in enumerate(new['rows']):
  if z['evidence_conflict'][t]:
   row['state']=2;row['reason']='高模型分数与分离证据冲突，保留未知。距离 %.2f mm；不确定尺度 %.2f mm；原始分数未修改。'%(d['distance'][t]*1000,d['sigma'][t]*1000)
  elif row['state']==0:row['reason']+=' 仅是候选，仍可能包含贴近夹指但未有效操作的情况。'
 new['counts']=[int(z['contact_candidate'].sum()),int(z['observed_separation_candidate'].sum()),int(z['unknown'].sum())]
 c['models']=[old,new]
 if c['id']=='tomato_sauce_demo_3':c['note']+=' 本轮仅把67–70、72–73共6帧改为冲突/未知；49–66的近距离误报仍未解决。第一次尝试52–69中仅67–69三帧因此改变，不能称已识别整次空夹。'
s=s[:a]+json.dumps(clips,ensure_ascii=False,separators=(',',':'))+s[a+n:]
s=s.replace('CF005 · 多次尝试视频接触预测','CF006 · 抓空误报与证据冲突对照')
s=s.replace('两段未参与接触网络训练的完整视频，均有两次闭合尝试。直接使用冻结CF005预测，没有重新训练、调阈值或按旧尝试标签改分。','同一冻结CF005模型，比较原候选和证据冲突修正。没有重训，原始分数完全一致。右侧选择CF006查看修正；明确分离却高分的帧保留为未知。')
s=s.replace('本页没有新增人工接触标签；这些案例在早期区域工程中看过，不是盲测。','用户指出tomato3抓空，本轮反馈只作复核。尚未解决49–66帧的近距离误报；这些案例属于开发复核，不是盲测。')
s=s.replace('冻结模型输出','原始模型分数（CF006未改分）')
out=ROOT/'docs/CF006_抓空误报与证据冲突对照.html';out.write_text(s)
assert len(clips)==2 and all(len(c['models'])==2 for c in clips)
for c in clips:
 for t,(a,b) in enumerate(zip(c['models'][0]['rows'],c['models'][1]['rows'])):
  assert a['scores']==b['scores']
  z=np.load(RUN/c['id']/'artifacts/frame_predictions.npz')
  assert b['state']==(0 if z['contact_candidate'][t] else 1 if z['observed_separation_candidate'][t] else 2)
(RUN/'audits/display.json').write_text(json.dumps({'page':str(out),'rows':sum(c['n'] for c in clips),'models':2,'raw_scores_equal':True,'states_match_saved_predictions':True,'video_bytes_reused':True},ensure_ascii=False,indent=2))
print(json.dumps({'page':str(out),'bytes':out.stat().st_size}))
