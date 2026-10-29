from pathlib import Path
import json,html,hashlib,argparse
R=Path('/mnt/nvme1/drh/D_VLA')
E=R/'experiments/contact_frames/cf018'
cases=json.loads((E/'episode_manifest.json').read_text())['cases']
label={'A':'仅RGB','B':'RGB＋任务指令','C':'RGB＋传感器','D':'全部输入'}
colors={'A':'#38bdf8','B':'#a78bfa','C':'#fb923c','D':'#34d399'}
def clamp_interval(x,n):
 try:
  a=int(x['start_frame']);b=int(x['end_frame'])
  if a<0 or b<a or b>=n:return None
  return a,b
 except:return None
def bars(x,n,color,kind):
 out=[]
 for z in x:
  ab=clamp_interval(z,n)
  if ab is None:continue
  a,b=ab
  obj=html.escape(str(z.get('object','物体')))
  title=html.escape(f"{obj} · {a}–{b} · {kind} · "+str(z.get('evidence_mode','')))
  out.append(f'<button class="bar" data-frame="{a}" title="{title}" style="left:{100*a/n:.3f}%;width:{max(0.35,100*(b-a+1)/n):.3f}%;background:{color}"></button>')
 return ''.join(out)
def get(name,arm,kind):
 pattern=f'{name}__{arm}__{kind}__*.json'
 return [json.loads(p.read_text()) for p in sorted((E/'responses_for_fine').glob(pattern))]
def row(name,arm,n):
 allout=get(name,arm,'global')+get(name,arm,'window')
 globalout=[x for x in allout if x['kind']=='global']
 windowout=[x for x in allout if x['kind']=='window']
 def fill(group,col):
  contacts=[];unknown=[];bad=0;details=[]
  for x in group:
   p=x.get('parsed') if isinstance(x.get('parsed'),dict) else {}
   for z in p.get('contact_intervals',[]) if isinstance(p.get('contact_intervals',[]),list) else []:
    if isinstance(z,dict):contacts.append(z)
   for z in p.get('unknown_intervals',[]) if isinstance(p.get('unknown_intervals',[]),list) else []:
    if isinstance(z,dict):unknown.append(z)
   bad+=int(x.get('parse_status')=='unparsed')
   txt=html.escape(x.get('raw_response',''))
   span=html.escape(str(x.get('frame_ids',[])[0]))+'–'+html.escape(str(x.get('frame_ids',[])[-1]))
   details.append(f'<details><summary>{col} 窗 {span} · {html.escape(x.get("parse_status",""))}</summary><pre>{txt}</pre></details>')
  return contacts,unknown,bad,''.join(details)
 cg,ug,bg,dg=fill(globalout,'全局')
 cw,uw,bw,dw=fill(windowout,'定窗')
 color=colors[arm]
 return f'''<div class="arm"><div class="label">{label[arm]} · 全局{len(globalout)}窗 / 细查{len(windowout)}窗 · 解析失败{bg+bw}</div>
 <div class="track"><span>全局</span>{bars(cg,n,color,'全局')}{bars(ug,n,'#64748b','未知')}</div>
 <div class="track"><span>固定窗</span>{bars(cw,n,color,'固定窗')}{bars(uw,n,'#64748b','未知')}</div>
 <div class="small">彩色=模型报告接触；灰色=模型报告未知。重叠条保留各窗原判断，未强行合并。</div>{dg}{dw}</div>'''
def build():
 sections=[]
 for c in cases:
  name=c['case'];n=c['frames']
  folder='cf017_contact' if name in ['two_objects_demo_0','drawer_demo_0','stove_demo_0'] else 'cf014_contact'
  video='/图示/'+folder+'/'+name+'_clean.mp4'
  rows=''.join(row(name,a,n) for a in ['A','B','C','D'])
  sections.append(f'''<section id="{name}"><h2>{html.escape(name)} <small>{n}帧</small></h2>
   <video controls preload="metadata" src="{video}"></video><div class="time">当前约第 <b>0</b> 帧 · 播放器20fps</div>
   <div class="ruler"><span>0</span><span>{n//4}</span><span>{n//2}</span><span>{3*n//4}</span><span>{n-1}</span></div>
   {rows}</section>''')
 count=sum(1 for _ in (E/'responses_for_fine').glob('*.json')) if (E/'responses_for_fine').exists() else 0
 css='''body{font:15px system-ui;background:#101722;color:#e5ebf3;max-width:1200px;margin:0 auto;padding:24px}h1{font-size:28px}section{background:#192434;padding:20px;margin:24px 0;border-radius:14px}video{width:100%;max-width:896px;display:block;margin:12px auto}small,.small{color:#a9b8ca}.arm{border-top:1px solid #334155;padding:12px 0}.label{font-weight:650}.track{height:32px;background:#26354a;position:relative;margin:7px 0;border-radius:5px;overflow:hidden}.track span{position:absolute;z-index:2;left:5px;top:7px;font-size:12px;color:white;text-shadow:0 1px 3px #000}.bar{position:absolute;top:0;height:100%;border:0;opacity:.8;cursor:pointer}.bar:hover{opacity:1;outline:2px solid white}.ruler{display:flex;justify-content:space-between}details{margin:5px 0;color:#abb9ca}pre{white-space:pre-wrap;word-break:break-word;background:#0d1420;padding:12px;border-radius:6px;max-height:280px;overflow:auto}.time{padding:5px}a{color:#7dd3fc}'''
 js='''document.querySelectorAll("section").forEach(s=>{let v=s.querySelector("video"),b=s.querySelector(".time b");v.addEventListener("timeupdate",()=>b.textContent=Math.floor(v.currentTime*20));s.querySelectorAll(".bar").forEach(x=>x.addEventListener("click",()=>{v.currentTime=Number(x.dataset.frame)/20;v.play()}))})'''
 page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CF018 冻结VLM接触时段预测</title><style>{css}</style><h1>CF018：冻结VLM接触时段预测</h1><p>当前已保存 {count}/184 份正式输出。四组使用同一批LIBERO帧；时间条是模型报告，不是物理接触真值。播放器可直接点击时间条定位。完整结果冻结前，本页不载入人工范围。</p><p><a href="/预训练VLM_LIBERO接触时段验证方案.md">查看方案与实施记录</a></p>{''.join(sections)}<script>{js}</script></html>'''
 out=R/'docs/CF018_冻结VLM_LIBERO接触时段预测.html'
 out.write_text(page,encoding='utf-8')
 print(json.dumps({'page':str(out),'response_files':count,'bytes':out.stat().st_size},ensure_ascii=False))
if __name__=='__main__':build()
