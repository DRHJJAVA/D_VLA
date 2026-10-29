from pathlib import Path
import json,hashlib,datetime
from zoneinfo import ZoneInfo
ROOT=Path('/mnt/nvme1/drh/D_VLA')
E=ROOT/'experiments/contact_frames/cf019'
N=148
pred=[]
for t in range(N):
 p=E/'predictions'/f'{t:04d}.json'
 if p.exists():
  d=json.loads(p.read_text())
  assert d['frame']==t
  pred.append({'t':t,'state':d['state'],'object':d.get('object'),'reason':d.get('reason',''),'history':d['history_frame_ids'],'parse':d['parse_status']})
 else:pred.append({'t':t,'state':'pending','object':None,'reason':'尚未推理','history':[],'parse':'pending'})
data=json.dumps({'pred':pred,'count':sum(x['state']!='pending' for x in pred)},ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
template=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CF019 单视频逐帧接触判断</title>
<style>
:root{font:16px/1.5 system-ui,sans-serif;background:#101722;color:#eaf2fa}*{box-sizing:border-box}body{max-width:1200px;margin:auto;padding:24px}
h1{font-size:26px;margin:0 0 8px}h2{font-size:18px;margin:0 0 8px}.warning{border:1px solid #b78648;background:#3b3023;border-radius:8px;padding:10px;margin:12px 0}.muted{color:#b8c9d8}
.box{background:#1b2a3c;border:1px solid #3e5b79;border-radius:12px;padding:16px;margin:12px 0}
button,input,select{font:inherit;color:#fff;background:#29405a;border:1px solid #6689aa;border-radius:7px;padding:6px 10px}
button{cursor:pointer}button:hover{background:#3c5d82}
.flex{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:12px 0}input[type=range]{flex:1;min-width:280px}input[type=number]{width:95px}
#viewer{display:block;width:100%;height:auto;border-radius:8px;background:#070a0f}
#timeline{display:flex;height:30px;gap:0;width:100%;border-radius:6px;overflow:hidden}
#timeline button{flex:1;min-width:0;border:0;border-radius:0;padding:0;border-right:1px solid #1117}
.contact{background:#28b971!important}.no_contact{background:#557cac!important}.unknown{background:#adb6c0!important}.pending{background:#4f4b50!important}
#timeline button.active{outline:3px solid #fff;outline-offset:-3px;z-index:1}
.legend{display:flex;gap:15px;flex-wrap:wrap;margin:8px 0}.swatch{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px}
.state{font-size:21px;font-weight:700}.card{padding:13px;border-radius:9px;background:#253a50}
details{margin-top:10px}a{color:#a2d4ff}
</style></head><body>
<h1>CF019｜一个视频，逐原帧判断当前是否接触</h1>
<p class="muted">罐头 demo0，共 148 帧（20 fps）。冻结预训练 VLM；没有用人工区间训练或提示。每次只给当前帧与更早的帧、双相机 RGB-D、夹爪／末端／关节／动作和相机标定得到的 3D 连续量。输出是模型对当前帧的判断，不是物理接触真值，也不是概率。</p>
<p class="warning">复核提示：你此前指出 47–134 帧在接触；本轮 88 帧中只报接触 1 帧，漏报严重。以下保留模型原样输出，不用这个结果自动生成接触监督。</p>
<div class="box">
<h2>逐帧结果｜<span id="counts"></span></h2>
<div id="timeline" aria-label="每一原始帧的接触判断"></div>
<div class="legend"><span><i class="swatch contact"></i>模型报接触</span><span><i class="swatch no_contact"></i>模型报未接触</span><span><i class="swatch unknown"></i>模型不确定</span><span><i class="swatch pending"></i>尚未推理</span></div>
<div class="flex"><button id="prev">上一帧</button><button id="play">逐帧播放</button><button id="next">下一帧</button><input id="slide" type="range" min="0" max="147" value="0"><label>原帧 <input id="jump" type="number" min="0" max="147" value="0"></label></div>
<div class="state" id="current"></div><div class="card" id="answer"></div>
<img id="viewer" alt="当前帧双相机RGB及米制深度">
<div id="loading" class="muted"></div>
</div>
<div class="box">
<h2>输入与只读复核</h2>
<p class="muted" id="history"></p>
<p class="muted">原视频帧号 t 从 0 开始，时间=t÷20 秒。回放 RGB-D 帧 t 对应官方 obs／动作 t−1；t=0 无对应状态。图中上排主相机／腕相机 RGB，下排为相同帧的深度伪彩；不同相机的深度色标在图上标明。3D 数值是深度反投影的几何提示，包含可能的机器人／背景点，不是接触测量。</p>
<details><summary>只读复核：此前用户给出的区间</summary><p>用户曾指出本视频 47–134 帧有接触；这段信息没有进入模型输入、推理或逐帧结果生成。此处只供人工比较，区间外也不自动当作负例。</p><p id="posthoc"></p></details>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),P=D.pred,N=P.length;
const $=id=>document.getElementById(id);let t=0,timer=null,seq=0;
const names={contact:'接触',no_contact:'未接触',unknown:'不确定',pending:'尚未推理'};
const counts={contact:0,no_contact:0,unknown:0,pending:0};P.forEach(p=>counts[p.state]++);
$('counts').textContent='已完成 '+D.count+'/'+N+' 帧；接触 '+counts.contact+'，未接触 '+counts.no_contact+'，不确定 '+counts.unknown+(counts.pending?'，待推理 '+counts.pending:'');
const list=[];
P.forEach((p,i)=>{const b=document.createElement('button');b.className=p.state;b.title='原帧 '+i+' / '+(i/20).toFixed(2)+'秒：'+names[p.state];b.setAttribute('aria-label',b.title);b.onclick=()=>{stop();show(i)};$('timeline').append(b);list.push(b)});
function stop(){if(timer)clearTimeout(timer);timer=null;$('play').textContent='逐帧播放'}
function show(x){
 t=Math.max(0,Math.min(N-1,Math.round(Number(x)||0)));const p=P[t];
 $('slide').value=t;$('jump').value=t;$('current').textContent='原帧 t='+t+' / '+(N-1)+'｜'+(t/20).toFixed(2)+' 秒｜模型：'+names[p.state];
 $('answer').textContent='状态：'+names[p.state]+(p.object?'；模型所称对象：'+p.object:'')+'。依据：'+p.reason+(p.parse!=='ok'?'；解析状态：'+p.parse:'');
 $('history').textContent='本次实际输入的原帧：'+(p.history.length?p.history.join(' → '):'未完成')+'（最后一张是当前帧；之前均为历史帧）';
 list.forEach((b,i)=>b.classList.toggle('active',i===t));
 const token=++seq;let im=$('viewer');
 im.onload=()=>{if(token===seq)$('loading').textContent='RGB-D 图已加载：原帧 '+t};
 im.onerror=()=>{if(token===seq)$('loading').textContent='图像未能加载'};
 im.src='/图示/cf019_inputs/'+String(t).padStart(4,'0')+'.jpg';
}
function tick(){if(!timer)return;if(t>=N-1){stop();return}show(t+1);timer=setTimeout(tick,90)}
$('prev').onclick=()=>{stop();show(t-1)};$('next').onclick=()=>{stop();show(t+1)};
$('slide').oninput=()=>{stop();show($('slide').value)};$('jump').oninput=()=>{stop();show($('jump').value)};
$('play').onclick=()=>{if(timer){stop();return}timer=setTimeout(tick,90);$('play').textContent='暂停'};
let known=P.slice(47,135);let kc=known.filter(x=>x.state==='contact').length,ku=known.filter(x=>x.state==='unknown').length,kn=known.filter(x=>x.state==='no_contact').length;
$('posthoc').textContent='在用户已指出的 47–134 共 '+known.length+' 帧中，模型报接触 '+kc+'、不确定 '+ku+'、未接触 '+kn+'。这是只读复核计数，不用于调本轮结果。';
show(0);
</script></body></html>'''
page=template.replace('__DATA__',data)
out=ROOT/'docs/CF019_单视频逐帧接触预测.html'
out.write_text(page,encoding='utf-8')
print(json.dumps({'page':str(out),'bytes':out.stat().st_size,'predicted':D if False else sum(x['state']!='pending' for x in pred),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},ensure_ascii=False))
