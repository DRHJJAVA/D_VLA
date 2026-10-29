from pathlib import Path
import json,html
from PIL import Image,ImageDraw,ImageFont
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/lcjp003/r2/v0_v1/v001';A=R/'docs/LCJP003_R2_V01_assets'
d=json.loads((P/'v1_result.json').read_text())
for k in range(10):
 cs=d['cases'][k*10:k*10+10];im=Image.new('RGB',(1280,1120),'#f0f4f8');draw=ImageDraw.Draw(im)
 for j,c in enumerate(cs):
  x=(j%5)*256;y=(j//5)*560
  for v in range(2):
   src=Image.open(A/c['images'][v]['overlay']).resize((256,256))
   im.paste(src,(x,y+34+v*256))
  draw.text((x+5,y+5),f"case {c['index']} {c['demo']} frame {c['frame']}",fill='black')
  draw.text((x+5,y+20),f"target {c['target']} top #{c['top_target']}",fill='black')
 im.save(A/f'task_{k:02d}_contact.jpg',quality=92)
v0=json.loads((P/'v0_result.json').read_text()) if (P/'v0_result.json').exists() else {'status':'running'}
(A/'v0_results.json').write_text(json.dumps(v0,ensure_ascii=False))
page=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LCJP003 R2 · V0 / V1 检查</title>
<style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#172b3e;background:#edf2f5}body{margin:0}header{padding:32px max(24px,calc((100vw - 1240px)/2));background:#102c3a;color:#fff}header small{color:#81d5c0;letter-spacing:2px}h1{font-size:28px;margin:12px 0}header p{color:#c9d6df;line-height:1.7;max-width:1000px}main{max-width:1240px;margin:24px auto;padding:0 20px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card,section{background:white;border:1px solid #dbe3e8;border-radius:12px;padding:20px;margin-bottom:16px}.card b{font-size:25px;display:block;margin:8px 0}.muted{color:#617484;font-size:14px;line-height:1.6}.warn{background:#fff2de;border-left:4px solid #c78527;padding:16px;line-height:1.7}h2{font-size:20px}label{margin-right:20px}select,button{font:inherit;padding:8px;border:1px solid #b6c5ce;border-radius:6px;background:white}button{cursor:pointer}.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}.views img{width:100%;image-rendering:auto}.viewtitle{font-weight:600;padding:12px 0}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #e3eaee}th{color:#506675}.tag{color:#b14c19;font-weight:600}details{margin-top:16px}#audit{width:100%}code{word-break:break-all}.check{display:grid;grid-template-columns:1fr 1fr;gap:8px}a{color:#126b77}@media(max-width:750px){.cards{grid-template-columns:1fr 1fr}.views,.check{grid-template-columns:1fr}}
</style>
<header><small>LCJP003 · R2 / V001</small><h1>先核对输入，再开始学习</h1><p>V0 检查真实预训练权重、动作接口和梯度；V1 检查分割候选与冻结语义先验。此次 optimizer 更新数为 0。这里的颜色与分数均不代表已经学会目标选择或抓取部位评分。</p></header>
<main><div class="cards"><div class="card">开发样本<b>100 帧</b><span class="muted">10 个任务 · 20 段示范</span></div><div class="card">视角<b>200 张</b><span class="muted">外部相机 + 腕部相机</span></div><div class="card">候选<b>835 个</b><span class="muted">全部保留，可逐帧核对</span></div><div class="card">训练更新<b>0</b><span class="muted">旧训练及闭环保持停止</span></div></div>
<section><h2>V0 · 接口检查</h2><div id="v0">正在读取检查结果…</div><p class="muted">核心接口检查只证明能传递数据和梯度；不证明随机初始化的目标门与评分已经有效。V2 的共享解释教师、价值权重和连续动作执行仍需后续实现与验证。</p></section>
<div class="warn"><strong>V1 需要注意：</strong>红色轮廓表示冻结语义先验排名第一的候选，可能选错。候选中也可能存在桌面、机械臂碎片及重复区域。语义值是 Yes / No 两个词的相对似然，不是识别正确率；没有独立标注，不能报告目标覆盖率或识别准确率。</div>
<section><h2>V1 · 逐帧双视角检查</h2><p><label>任务 <select id="task"></select></label><label>示范 / 帧 <select id="case"></select></label><button id="prev">上一帧</button> <button id="next">下一帧</button></p><p><label><input type="checkbox" id="raw"> 只看原图</label><span id="caption"></span></p><div class="views"><div><div class="viewtitle">外部相机</div><img id="cam0"></div><div><div class="viewtitle">腕部相机</div><img id="cam1"></div></div><p class="muted">边界框及掩膜来自已有分割缓存；#编号是候选 ID。遮挡或单视角候选保留，不补造不可见物体。来源：开发 demo45 / demo46，未读取留出 demo40–44。</p><table><thead><tr><th>候选</th><th>真实点数</th><th>可见视角</th><th>目标先验</th><th>篮子先验</th><th>目标排名</th></tr></thead><tbody id="rows"></tbody></table><details><summary>当前任务全部 10 帧双视角总览</summary><img id="audit"></details></section>
<section><h2>先验是否真的看区域？</h2><p>固定 20 个首帧样本，将最高先验候选的区域擦除，再使用同一问题推理。目标先验绝对变化均值 <b>0.03950</b>，最大 <b>0.10457</b>。这是输入响应检查，不能据此认定选对目标；完整图像的上下文仍可能影响判断。</p><p class="muted">抽样锁在推理前写入；无人工接触标签、无仿真物体真值、无强制识别正确率门槛。前置检查中观察到的错误会保留用于定位输入问题，不自动转成监督标签。</p></section></main>
<script>
const root='LCJP003_R2_V01_assets/';let D,cs=[],n=0;
const $=x=>document.getElementById(x);
async function init(){D=await(await fetch(root+'v1_results.json')).json();let ts=[...new Set(D.cases.map(x=>x.target))];ts.forEach((t,i)=>$('task').add(new Option(t,i)));$('task').onchange=task;$('case').onchange=()=>{n=+$('case').value;show()};$('raw').onchange=show;$('prev').onclick=()=>{n=Math.max(0,n-1);show()};$('next').onclick=()=>{n=Math.min(cs.length-1,n+1);show()};task();
let v=await(await fetch(root+'v0_results.json')).json();if(v.status==='core_interface_smoke_passed'){$('v0').innerHTML='<div class="check"><div>✓ 490 / 490 主干张量精确加载</div><div>✓ LIBERO 归一化往返误差 '+v.action_normalization_roundtrip.toExponential(2)+'</div><div>✓ 动作损失→评分梯度 '+v.action_loss_score_gradient_norm.toExponential(3)+'</div><div>✓ 动作损失→目标门梯度 '+v.action_loss_role_gradient_norm.toExponential(3)+'</div><div>✓ 无效填充扰动差 '+v.padding_perturbation_max+'</div><div>✓ 参数更新 0，前后 hash 相同</div></div><p class="muted">真实 '+v.candidates+' 个候选，16 帧历史；一次前反向 '+v.seconds_one_forward_backward.toFixed(2)+' 秒，峰值显存 '+v.gpu_peak_allocated_gib.toFixed(2)+' GiB。单次 smoke 耗时不是训练 ETA。</p>'}else{$('v0').textContent='检查状态：'+v.status+'；尚未标记通过。'}}
function task(){cs=D.cases.slice(+$('task').value*10,+$('task').value*10+10);$('case').innerHTML='';cs.forEach((c,i)=>$('case').add(new Option(c.demo+' · '+c.frame+'/'+(c.frames-1),i)));n=0;$('audit').src=root+'task_'+String(+$('task').value).padStart(2,'0')+'_contact.jpg';show()}
function show(){let c=cs[n];$('case').value=n;$('caption').textContent='目标：'+c.target+' | 最高先验 #'+c.top_target+' | 目标覆盖：待独立核对';for(let v=0;v<2;v++)$('cam'+v).src=root+c.images[v][$('raw').checked?'raw':'overlay'];let rank=[...c.candidates].sort((a,b)=>b.target_prior-a.target_prior);$('rows').innerHTML=rank.map((a,i)=>'<tr><td>#'+a.id+'</td><td>'+a.points+'</td><td>'+a.views.map(v=>v.camera===0?'外部':'腕部').join('、')+'</td><td>'+a.target_prior.toFixed(4)+'</td><td>'+a.goal_prior.toFixed(4)+'</td><td class="'+(i===0?'tag':'')+'">'+(i+1)+'</td></tr>').join('')}
init().catch(e=>{$('v0').textContent='结果读取失败：'+e.message});
</script></html>'''
(R/'docs/LCJP003_R2_V0_V1检查.html').write_text(page)
print(json.dumps({'page':'LCJP003_R2_V0_V1检查.html','cases':100,'contact_sheets':10}))
