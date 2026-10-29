from pathlib import Path
import json, hashlib, html
from datetime import datetime
from zoneinfo import ZoneInfo

root=Path('/mnt/nvme1/drh/D_VLA')
exp=root/'experiments/contact_frames/cf018'
docs=root/'docs'
cases=['alphabet_soup_demo_0','alphabet_soup_demo_1','porcelain_mug_demo_9','white_yellow_mug_demo_5','tomato_sauce_demo_3','two_objects_demo_0','drawer_demo_0','stove_demo_0']
names={'alphabet_soup_demo_0':'罐头 demo0','alphabet_soup_demo_1':'罐头 demo1','porcelain_mug_demo_9':'瓷杯 demo9','white_yellow_mug_demo_5':'白黄杯 demo5','tomato_sauce_demo_3':'番茄酱 demo3','two_objects_demo_0':'双物体 demo0','drawer_demo_0':'抽屉 demo0','stove_demo_0':'炉灶开关 demo0'}
videos={c:('/图示/'+('cf017_contact' if c in cases[5:] else 'cf014_contact')+'/'+c+'_clean.mp4') for c in cases}
meta=[]
for c in cases:
    files=sorted((exp/'prepared_pair_fine'/c).glob('*.jpg'))
    assert files and all(p.name==f'{i:04d}.jpg' for i,p in enumerate(files))
    meta.append({'id':c,'name':names[c],'frames':len(files),'video':videos[c]})
records=[]
for p in sorted((exp/'responses').glob('*.json')):
    d=json.loads(p.read_text())
    npath=exp/'normalized_responses'/p.name
    n=json.loads(npath.read_text())['normalization'] if npath.exists() else None
    intervals=[]
    if n:
        for key in ('contact','unknown'):
            for item in n.get(key,[]):
                if item.get('status')=='mapped':
                    a,b=int(item['start_frame']),int(item['end_frame'])
                    intervals.append({'kind':key,'start':a,'end':b,'object':str(item.get('object',''))})
    frames=[int(x) for x in d['frame_ids']]
    assert frames and frames==sorted(set(frames))
    records.append({'id':p.stem,'case':d['case'],'arm':d['arm'],'kind':d['kind'],'start':d['start'],'frame_ids':frames,'parse':d['parse_status'],'normal_status':n.get('status') if n else '未整理','intervals':intervals,'issues':n.get('issues',[]) if n else []})
counts={'raw':len(records),'normalized':sum(r['normal_status']!='未整理' for r in records),'planned':184}
assert len(meta)==8 and counts['raw']==105
payload=json.dumps({'cases':meta,'records':records,'counts':counts},ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
page=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CF018 原帧逐帧核对</title>
<style>
:root{font:16px/1.5 system-ui,sans-serif;color:#eaf0f7;background:#0e1520}*{box-sizing:border-box}
body{max-width:1200px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 8px}.lead{color:#b9c7d7;margin:0 0 18px}
.panel{background:#192637;border:1px solid #35506c;border-radius:12px;padding:18px;margin:12px 0}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:10px 0}
button,select,input{font:inherit;border-radius:7px;border:1px solid #69819b;background:#263a50;color:#fff;padding:6px 10px}
button{cursor:pointer}button:hover{background:#355474}select{max-width:100%}
input[type=range]{flex:1;min-width:260px}input[type=number]{width:95px}
#frame{width:100%;height:auto;display:block;background:#0a0d13;border-radius:7px;object-fit:contain}
.big{font-size:22px;font-weight:700} .minor{color:#adbed0}.bad{color:#ffb4a8}.ok{color:#9fe0ae}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:14px}.cards>*{min-width:0}
@media(max-width:800px){.cards{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #3b5066;padding:5px 7px;text-align:left}
tbody tr{cursor:pointer}tbody tr:hover{background:#30445b}
#mapwrap{max-height:320px;overflow:auto}.overview{background:#23364c;border:1px solid #5b7896;border-radius:9px;padding:12px;margin:10px 0}.lane{display:flex;align-items:center;gap:9px;margin:5px 0}.lane strong{width:26px}.track{position:relative;flex:1;height:18px;background:#344558;border-radius:5px;overflow:hidden}.seg{position:absolute;top:2px;height:14px;min-width:3px;padding:0;border:0;border-radius:3px;background:#ffc34d}.chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:8px}.chips button{font-size:14px;background:#6c5529;border-color:#b58c40}.pill{border-radius:5px;padding:2px 6px;background:#3d536d}
.contact{background:#815f28}.unknown{background:#555d68}
a{color:#9dd0ff}details{margin-top:10px}
</style></head><body>
<h1>CF018｜原视频帧逐帧核对</h1>
<p class="lead">本页用原视频逐帧图片播放，避开黑屏播放器。逐帧播放是便于查看的约 11 fps，原视频帧号与时间仍按 20 fps 计算；可用上一帧、下一帧精确核对。图片左侧为主相机，右侧为腕相机。仅展示已保存的冻结 VLM 输出；未继续推理。模型报告不等于真实接触，也没有逐帧概率分数。</p>
<div class="panel">
<div class="row"><label>示范 <select id="case"></select></label><label>输入组合 <select id="arm"><option value="A">A 仅 RGB</option><option value="B">B RGB＋指令</option><option value="C">C RGB＋状态</option><option value="D">D RGB＋指令＋状态</option></select></label></div>
<div class="overview"><strong>候选帧在哪里？</strong><div id="candidateStatus" class="minor"></div><div id="candidateTracks"></div><div id="candidateChips" class="chips"></div><details id="extraCandidates" hidden><summary>显示其余候选</summary><div id="extraChips" class="chips"></div></details><p class="minor">黄色块是 VLM 报告、且能映射到原帧的候选；点击区间跳到起始帧。每行代表一种输入组合，可能互相矛盾。没有黄色块不等于没有接触。</p></div>
<div class="row"><button id="prev">上一帧</button><button id="play">逐帧播放</button><button id="next">下一帧</button><input id="slider" type="range"><label>跳到原帧 <input id="jump" type="number"></label></div>
<div class="row big"><span id="frameNo"></span><span id="time"></span></div>
<img id="frame" alt="原视频的主相机和腕相机逐帧画面">
<div id="load" class="minor"></div>
<p class="minor">帧号 t 从 0 开始；画面时间=t÷20 秒。回放 RGB-D 第 t 帧对应官方 obs／动作记录 t−1（t≥1）；t=0 的手部状态未知。这里的 t 始终是原视频帧号，不是模型抽帧后的序号。</p>
<details><summary>原 MP4 链接（逐帧查看不依赖它）</summary><a id="mp4" target="_blank" rel="noopener">打开原视频文件</a></details>
</div>
<div class="cards">
<div class="panel">
<h2>模型看过哪些帧</h2>
<p class="minor">全局输入约每 10 个原帧取 1 张（2 fps）；固定窗每 2 个原帧取 1 张（10 fps）。k 是送给模型的第 k 张图，t 是原帧号。二者必须按下表对应，不能把 k 当作 t。</p>
<label>查看一条输出 <select id="record"></select></label>
<p id="recinfo"></p><div id="mapwrap"><table><thead><tr><th>模型图序号 k</th><th>原帧 t</th><th>时间</th></tr></thead><tbody id="mapping"></tbody></table></div>
</div>
<div class="panel">
<h2>模型报告</h2>
<p id="count" class="minor"></p>
<div id="report"></div>
<p class="minor">黄色是该条输出报告的接触候选；灰色是报告未知。空白只表示这条输出未报告候选，不能判为“未接触”。未整理或解析有疑点的输出直接标出，不涂成确定结果。不同窗口不强行合并。</p>
</div></div>
<script id="payload" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('payload').textContent);
const $=id=>document.getElementById(id);
const cs=$('case'),ar=$('arm'),rs=$('record'),im=$('frame'),sl=$('slider'),jump=$('jump');
let t=0,timer=null,seq=0;
const fmt=n=>String(n).padStart(4,'0');
const sec=n=>(n/20).toFixed(2)+' 秒';
function currentCase(){return D.cases.find(x=>x.id===cs.value)}
function currentRecord(){return D.records.find(x=>x.id===rs.value)}
function stop(){if(timer)clearTimeout(timer);timer=null;$('play').textContent='逐帧播放'}
function show(n){
 const c=currentCase();if(!c)return;
 t=Math.max(0,Math.min(c.frames-1,Math.round(Number(n)||0)));
 sl.value=t;jump.value=t;$('frameNo').textContent='原帧 t='+t+' / '+(c.frames-1);
 $('time').textContent='时间 '+sec(t);
 let token=++seq;
 im.onload=()=>{if(token===seq)$('load').textContent='画面已加载：原帧 '+t+'；左主相机／右腕相机'};
 im.onerror=()=>{if(token===seq)$('load').textContent='图片加载失败：'+im.src};
 im.src='/图示/cf018_frames/'+c.id+'/'+fmt(t)+'.jpg';
}
function playNext(){
 if(!timer)return;
 const c=currentCase();if(t>=c.frames-1){stop();return}
 show(t+1);timer=setTimeout(playNext,90);
}
function drawOverview(){
 const c=currentCase(), all=[];
 for(const r of D.records.filter(x=>x.case===c.id))for(const z of r.intervals)if(z.kind==='contact')all.push({r,z});
 all.sort((a,b)=>a.z.start-b.z.start||a.r.arm.localeCompare(b.r.arm));
 $('candidateStatus').textContent='当前 '+ar.value+' 组合：'+(all.some(x=>x.r.arm===ar.value)?'有已保存的候选；下方显示全部输入组合。':'没有可映射候选；下方仍显示其他组合。')+' 本视频共 '+all.length+' 条候选报告（不同窗口可能重复），不是接触真值。';
 $('candidateTracks').replaceChildren();$('candidateChips').replaceChildren();$('extraChips').replaceChildren();
 const jumpTo=x=>{ar.value=x.r.arm;records();rs.value=x.r.id;drawRecord();show(x.z.start);drawOverview()};
 for(const arm of ['A','B','C','D']){
  const lane=document.createElement('div');lane.className='lane';
  const label=document.createElement('strong');label.textContent=arm;lane.append(label);
  const track=document.createElement('div');track.className='track';
  for(const x of all.filter(y=>y.r.arm===arm)){
   const b=document.createElement('button');b.className='seg';
   b.style.left=(100*x.z.start/c.frames)+'%';
   b.style.width=(100*(x.z.end-x.z.start+1)/c.frames)+'%';
   b.title=arm+' '+(x.r.kind==='global'?'全局':'固定窗')+' 原帧 '+x.z.start+'–'+x.z.end;
   b.setAttribute('aria-label',b.title);b.onclick=()=>jumpTo(x);track.append(b);
  }
  lane.append(track);$('candidateTracks').append(lane);
 }
 if(!all.length){const p=document.createElement('p');p.textContent='已保存输出中没有可映射的候选。';$('candidateChips').append(p)}
 all.forEach((x,i)=>{
  const b=document.createElement('button');
  b.textContent=x.r.arm+(x.r.kind==='global'?' 全局':' 固定窗')+'｜原帧 '+x.z.start+'–'+x.z.end+(x.z.object?'｜'+x.z.object:'');
  b.onclick=()=>jumpTo(x);
  (i<8?$('candidateChips'):$('extraChips')).append(b);
 });
 $('extraCandidates').hidden=all.length<=8;
 if(all.length>8)$('extraCandidates').querySelector('summary').textContent='显示其余 '+(all.length-8)+' 条候选';
}
function records(){
 const c=currentCase();const list=D.records.filter(x=>x.case===c.id&&x.arm===ar.value)
 .sort((a,b)=>(a.kind==='global'?-1:1)-(b.kind==='global'?-1:1)||a.frame_ids[0]-b.frame_ids[0]);
 rs.replaceChildren();
 for(const r of list){let o=document.createElement('option');o.value=r.id;o.textContent=(r.kind==='global'?'全局':'固定窗')+'｜原帧 '+r.frame_ids[0]+'–'+r.frame_ids.at(-1)+(r.normal_status==='未整理'?'｜待整理':'');rs.append(o)}
 if(!list.length){let o=document.createElement('option');o.textContent='没有已保存的输出';rs.append(o)}
 drawRecord();
}
function drawRecord(){
 const r=currentRecord();$('mapping').replaceChildren();$('report').replaceChildren();
 if(!r){$('recinfo').textContent='尚无已保存的模型输出';return}
 $('recinfo').textContent='本条模型实际看了 '+r.frame_ids.length+' 张图；原视频范围 '+r.frame_ids[0]+'–'+r.frame_ids.at(-1)+'；解析 '+r.parse+'；映射 '+r.normal_status+'。';
 r.frame_ids.forEach((v,k)=>{let tr=document.createElement('tr');for(const s of [k,v,sec(v)]){let td=document.createElement('td');td.textContent=s;tr.append(td)}tr.onclick=()=>show(v);$('mapping').append(tr)});
 let out=$('report');
 if(r.normal_status==='未整理'){let p=document.createElement('p');p.className='bad';p.textContent='这条原始输出尚未完成帧号整理，不展示接触区间。';out.append(p)}
 else if(!r.intervals.length){let p=document.createElement('p');p.textContent='这条输出没有可安全映射到原帧的区间。';out.append(p)}
 for(const z of r.intervals){let p=document.createElement('p');let b=document.createElement('button');b.className=z.kind;b.textContent=(z.kind==='contact'?'候选接触':'未知')+'：原帧 '+z.start+'–'+z.end+'（'+sec(z.start)+'–'+sec(z.end)+'）'+(z.object?'｜'+z.object:'');b.onclick=()=>show(z.start);p.append(b);out.append(p)}
 if(r.issues.length){let p=document.createElement('p');p.className='bad';p.textContent='映射疑点：'+r.issues.join('；');out.append(p)}
}
function initCase(){
 stop();const c=currentCase();sl.max=c.frames-1;jump.max=c.frames-1;$('mp4').href=c.video;
 records();drawOverview();show(0);
}
D.cases.forEach(c=>{let o=document.createElement('option');o.value=c.id;o.textContent=c.name+'（'+c.frames+' 原帧）';cs.append(o)});
$('count').textContent='已保存 '+D.counts.raw+' / '+D.counts.planned+' 条模型输出；其中 '+D.counts.normalized+' 条完成自动帧号整理。其余保持未整理。不是新的推理结果。';
cs.onchange=initCase;ar.onchange=()=>{records();drawOverview()};rs.onchange=drawRecord;
$('prev').onclick=()=>{stop();show(t-1)};$('next').onclick=()=>{stop();show(t+1)};
sl.oninput=()=>{stop();show(sl.value)};jump.oninput=()=>{stop();show(jump.value)};
$('play').onclick=()=>{if(timer){stop();return}timer=setTimeout(playNext,90);$('play').textContent='暂停'};
initCase();
</script></body></html>'''
out=docs/'CF018_冻结VLM_LIBERO接触时段预测.html'
backup=docs/'CF018_冻结VLM_LIBERO接触时段预测_旧版备份.html'
if not backup.exists(): backup.write_bytes(out.read_bytes())
out.write_text(page.replace('__DATA__',payload),encoding='utf-8')
now=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %Z')
plan=docs/'预训练VLM_LIBERO接触时段验证方案.md'
with plan.open('a',encoding='utf-8') as f:
 f.write('\n\n### '+now+'：CF018 展示修复\n\n')
 f.write('仅修复页面，未运行模型或新增推理。旧版页面单独保留。原视频播放器在应用内出现黑画面，改用已保存的 20 fps 双相机逐帧 JPEG 作为主查看器；页面逐帧显示原始帧 t、t/20 秒和 RGB-D t 对应官方 obs/动作 t−1，t0 状态未知。模型采样序号 k 与原帧 t 用实际 frame_ids 一对一表格显示；全局 2 fps、固定窗 10 fps，点击表格或区间可跳原帧。候选仅显示自动安全映射区间，未整理/矛盾输出保持未知，不把空白当无接触。所有图、代码与页面仅在远程。\n')
 f.write('页面生成代码 SHA256：'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+'；页面 SHA256：'+hashlib.sha256(out.read_bytes()).hexdigest()+'。\n')
print(json.dumps({'page':str(out),'bytes':out.stat().st_size,'backup':str(backup),'raw':counts['raw'],'normalized':counts['normalized'],'cases':len(meta),'time':now,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},ensure_ascii=False))
