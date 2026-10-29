from pathlib import Path
import json,re,html,hashlib,datetime,zoneinfo
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018')
DOC=Path('/mnt/nvme1/drh/D_VLA/docs')
OUT=R/'normalized_responses';OUT.mkdir(exist_ok=True)
ARMS={'A':'仅 RGB','B':'RGB＋任务指令','C':'RGB＋传感器摘要','D':'RGB＋指令＋传感器'}
def integer(x):
    if isinstance(x,bool):return None
    try:
        y=int(x)
        return y if str(x).strip()==str(y) else None
    except:return None
def normalize_interval(z,ids,notes):
    n=len(ids);chosen=set(ids);a=integer(z.get('start_frame'));b=integer(z.get('end_frame'))
    ev=[integer(x) for x in z.get('evidence_frame_ids',[]) if integer(x) is not None]
    if a is None or b is None:return {'status':'invalid','reason':'missing boundary','raw':z}
    vals=[a,b]+ev
    if any(x<0 for x in vals):return {'status':'invalid','reason':'negative frame','raw':z}
    abs_hint=any(x>=n for x in vals)
    local_hint=any(x not in chosen and x<n for x in vals)
    if abs_hint and local_hint:
        return {'status':'ambiguous','reason':'mixed original and local indices','raw':z}
    if abs_hint:unit='original'
    elif local_hint:unit='sample_index'
    else:
        return {'status':'ambiguous','reason':'both index units possible','raw':z}
    if unit=='sample_index':
        if a>=n or b>=n:return {'status':'invalid','reason':'sample index outside input','raw':z}
        aa,bb=ids[a],ids[b]
        ee=[ids[x] for x in ev if x<n]
    else:
        aa,bb=a,b
        ee=ev
    if aa>bb or bb>ids[-1]:
        return {'status':'invalid','reason':'boundary outside video','raw':z}
    if str(z.get('object','')).strip().lower() in {'gripper','robot gripper','robot arm','fingers','left finger','right finger'}:
        return {'status':'self_object','reason':'model named the robot as the contacted object','raw':z,'mapped':[aa,bb],'unit':unit}
    low=notes.lower()
    contradiction=any(s in low for s in ['gripper is not in contact with any object','gripper does not make physical contact','contact is not visually confirmed'])
    if contradiction:
        return {'status':'contradiction','reason':'response text contradicts claimed contact','raw':z,'mapped':[aa,bb],'unit':unit}
    return {'status':'mapped','start_frame':aa,'end_frame':bb,'unit':unit,'object':str(z.get('object','unknown')),'evidence_frame_ids':ee,'raw':z}
def normal(d):
    q=d.get('parsed') or {};ids=d.get('frame_ids') or [];notes=str(q.get('notes',''))
    if not ids or d.get('parse_status')!='ok':
        return {'status':'unparsed','contact':[],'unknown':[],'issues':['parse failed']}
    out=[normalize_interval(z,ids,notes) for z in q.get('contact_intervals',[]) if isinstance(z,dict)]
    unk=[normalize_interval(z,ids,'') for z in q.get('unknown_intervals',[]) if isinstance(z,dict)]
    issues=[x['reason'] for x in out+unk if x['status']!='mapped']
    return {'status':'ok' if not issues else 'needs_review','contact':out,'unknown':unk,'issues':issues}
manifest=json.loads((R/'episode_manifest.json').read_text())['cases']
seen={}
for p in sorted((R/'responses').glob('*.json')):
    try:d=json.loads(p.read_text())
    except:continue
    x=normal(d);o={'case':d['case'],'arm':d['arm'],'kind':d['kind'],'start':d.get('start'),'frame_ids':d['frame_ids'],'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'normalization':x}
    (OUT/p.name).write_text(json.dumps(o,ensure_ascii=False,indent=2))
    seen[(d['case'],d['arm'],d['kind'],d.get('start'))]=(d,x)
def esc(x):return html.escape(str(x),quote=True)
parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>CF018 接触时段预测</title><style>body{font:15px system-ui;background:#101924;color:#e7f0f5;max-width:1200px;margin:auto;padding:24px}h1,h2{color:#f0f8ff}section{background:#192736;border:1px solid #385067;border-radius:12px;padding:20px;margin:24px 0}video{width:100%;max-width:896px}.track{position:relative;height:24px;background:#34495a;border-radius:6px;margin:5px 0}.seg{position:absolute;top:2px;height:20px;border-radius:5px;background:#66d9a9}.bad{color:#ffb670}.muted{color:#aebfcc}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #385067;padding:8px;text-align:left;vertical-align:top}details{margin:6px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}.legend{padding:12px;background:#263e51;border-radius:8px}</style><h1>CF018｜冻结预训练 VLM 接触时段预测</h1><p class="legend">当前为全局 2 fps 粗定位。绿色区间仅表示模型报告、且帧号单位可以按实际输入确定性换算；橙色提示表示单位歧义或回答自相矛盾，不画成确定接触。固定窗口与原帧率边界细查仍在运行。无人工区间训练或选窗。</p>']
for c in manifest:
 name=c['case'];n=c['frames']
 group='cf014_contact' if name in ['alphabet_soup_demo_0','alphabet_soup_demo_1','porcelain_mug_demo_9','white_yellow_mug_demo_5','tomato_sauce_demo_3'] else 'cf017_contact'
 src='图示/'+group+'/'+name+'_clean.mp4'
 parts += ['<section><h2>'+esc(name)+'</h2><p class="muted">原视频 '+str(n)+' 帧，20 fps。任务指令只输入 B/D 组。</p><video controls preload="metadata" src="'+esc(src)+'"></video><table><tr><th>输入</th><th>模型区间（原帧）</th><th>时间轴</th><th>核对状态</th></tr>']
 for arm in ARMS:
  item=seen.get((name,arm,'global',None))
  if item is None:
   parts+=['<tr><td>'+arm+' '+ARMS[arm]+'</td><td colspan="3">尚未输出</td></tr>'];continue
  d,x=item
  mapped=[z for z in x['contact'] if z['status']=='mapped']
  other=[z for z in x['contact'] if z['status']!='mapped']
  spans=', '.join(esc(z['object'])+' '+str(z['start_frame'])+'–'+str(z['end_frame']) for z in mapped) if mapped else '无确定区间'
  bar='<div class="track">'
  for z in mapped:
   left=100*z['start_frame']/max(n-1,1);width=max(.4,100*(z['end_frame']-z['start_frame']+1)/max(n,1))
   bar+='<span class="seg" style="left:'+f'{left:.2f}'+'%;width:'+f'{width:.2f}'+'%" title="'+esc(z['object'])+' '+str(z['start_frame'])+'–'+str(z['end_frame'])+'"></span>'
  bar+='</div>'
  issue='；'.join(x['issues']) if x['issues'] else '可换算'
  if other:issue+='；'+str(len(other))+' 个区间未画'
  parts+=['<tr><td>'+arm+' '+ARMS[arm]+'</td><td>'+spans+'</td><td>'+bar+'</td><td class="'+('bad' if x['issues'] else 'muted')+'">'+esc(issue)+'</td></tr>']
  parts+=['<tr><td colspan="4"><details><summary>原始模型回答与输入帧号（'+arm+'）</summary><pre>'+esc(d.get('raw_response',''))+'</pre><pre>'+esc(d.get('frame_ids',[]))+'</pre></details></td></tr>']
 parts+=['</table></section>']
parts+=['<p class="muted">所有区间都是外部预训练 VLM 的预测，不是物理接触真值，也不是训练集人工标签。若画面遮挡或证据不足，应保留未知。</p></html>']
page=DOC/'CF018_冻结VLM_LIBERO接触时段预测.html';page.write_text(''.join(parts))
stats={'timestamp_beijing':datetime.datetime.now(zoneinfo.ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),'raw_responses':len(seen),'global_responses':sum(k[2]=='global' for k in seen),'mapped_global_intervals':sum(sum(z['status']=='mapped' for z in x['contact']) for (d,x) in seen.values() if d['kind']=='global'),'ambiguous_or_contradictory_global_intervals':sum(sum(z['status']!='mapped' for z in x['contact']) for (d,x) in seen.values() if d['kind']=='global'),'page':str(page)}
(R/'frame_unit_audit.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
print(json.dumps(stats,ensure_ascii=False))
