import json,hashlib,traceback
from pathlib import Path
import numpy as np,h5py
from PIL import Image
R=Path("/mnt/nvme1/drh/D_VLA");O=R/"experiments/contact_preference/stage3_pilot_v1"
def apply(p,T):return p@T[:3,:3].T+T[:3,3]
def main():
 S=json.loads((O/"summary.json").read_text());(O/"webdata").mkdir(exist_ok=True)
 for ci,c in enumerate(S["cases"]):
  with h5py.File(R/"data/v3/trainval_rgbd_448"/c["asset"]/(c["demo"]+".h5"),"r") as h:
   q=c["rows"][0]["query_frame"];Image.fromarray(h["agentview/rgb"][q]).save(O/"images"/("c"+str(ci)+"_query.jpg"),quality=93)
   for row in c["rows"]:
    t=row["frame"];z=np.load(O/row["data"]);name="c"+str(ci)+"_f"+str(t)
    projections=[]
    for cam in ["agentview","robot0_eye_in_hand"]:
     g=h[cam];Image.fromarray(g["rgb"][t]).save(O/"images"/(name+"_"+cam+".jpg"),quality=93)
     K=np.asarray(g["K"][t] if g["K"].ndim==3 else g["K"]);T=np.asarray(g["camera_to_world"][t] if g["camera_to_world"].ndim==3 else g["camera_to_world"])
     p=apply(z["mapped_world_xyz"],np.linalg.inv(T.astype(np.float64)));v=p@K.T;uv=v[:,:2]/v[:,2:3];uv[~np.isfinite(uv)]=0
     projections.append(np.round(uv,2).tolist())
    item=dict(points=np.round(z["mapped_hand_xyz"]*1000,3).tolist(),source_pixel=z["query_pixel"].tolist(),
     supports=np.round(np.column_stack([z["positive"],z["separation"],z["unknown"]]),6).tolist(),
     category=z["display_class"].tolist(),visible=z["currently_depth_consistent"].astype(int).tolist(),
     inside=z["penetrated"].astype(int).tolist(),distance_mm=np.round(z["distance_m"]*1000,3).tolist(),
     sigma_mm=np.round(z["sigma_base_m"]*1000,3).tolist(),sigma_proxy_mm=np.round(z["sigma_proxy_m"]*1000,3).tolist(),
     raw_lower=np.round(z["raw_lower"],6).tolist(),raw_upper=np.round(z["raw_upper"],6).tolist(),
     projection=projections,robot=np.round(z["gripper_vertices"]*1000,3).tolist(),faces=z["gripper_faces"].tolist())
    (O/"webdata"/(name+".json")).write_text(json.dumps(item,separators=(",",":")))
    row["webstem"]=name
 (O/"webdata"/"summary.json").write_text(json.dumps(S,ensure_ascii=False,separators=(",",":")))
 docs=R/"docs";link=docs/"图示/阶段3支持"
 if not link.exists():link.symlink_to(O,target_is_directory=True)
 html=HTML.replace("__SUMMARY__",json.dumps(S,ensure_ascii=False,separators=(",",":")))
 (docs/"接触偏好_阶段3_正负支持与未知.html").write_text(html)
 print(json.dumps(dict(page="stage3",images=len(list((O/"images").glob("*.jpg"))),frame_data=len(list((O/"webdata").glob("c*.json"))))))
HTML=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>阶段3 · 正支持、分离支持与未知</title>
<style>
:root{--ink:#17323a;--muted:#60757b;--line:#d4e0df;--green:#16805d;--red:#bc544b;--blue:#557db5}
*{box-sizing:border-box}body{margin:0;background:#f1f5f3;color:var(--ink);font-family:system-ui,"Microsoft YaHei",sans-serif;font-size:14px}
header{padding:28px max(24px,calc((100vw - 1360px)/2));background:#173d42;color:white;border-bottom:5px solid #d9b768}
header small{letter-spacing:.16em;color:#bad2d1}h1{font-size:27px;margin:8px 0 10px;font-weight:650}header p{margin:0;line-height:1.8;color:#dfebea}
main{max-width:1408px;margin:auto;padding:22px 24px 60px}.notice{padding:14px 18px;background:#fff7e8;border:1px solid #dfca99;border-radius:8px;line-height:1.8;margin-bottom:18px}
.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:12px 0}button,select{font:inherit;border:1px solid #b7cbca;background:#fff;color:var(--ink);padding:9px 14px;border-radius:6px;cursor:pointer}button.active{background:#204f55;color:white;border-color:#204f55}button:hover{border-color:#267b78}label{color:var(--muted)}
.frames button{padding:7px 14px}.card{background:white;border:1px solid var(--line);border-radius:9px;overflow:hidden}.title{padding:13px 16px;border-bottom:1px solid var(--line);font-weight:600}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:16px 0}.viewbox{position:relative;aspect-ratio:1}.viewbox img{width:100%;height:100%;object-fit:contain;display:block}.cross{position:absolute;transform:translate(-50%,-50%);width:20px;height:20px;border:2px solid #fff22c;border-radius:50%;box-shadow:0 0 0 1px #202020;pointer-events:none;display:none}.cross:before,.cross:after{content:"";position:absolute;background:#fff22c}.cross:before{width:28px;height:1px;left:-6px;top:7px}.cross:after{height:28px;width:1px;top:-6px;left:7px}.caption{padding:10px 14px;font-size:12px;color:var(--muted);line-height:1.6}
.grid{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(300px,1fr);gap:16px}.canvaswrap{height:490px;position:relative;background:#f8faf9}canvas{width:100%;height:100%;display:block;touch-action:none}.canvashelp{position:absolute;bottom:12px;left:14px;color:#647c81;font-size:12px;pointer-events:none}.content{padding:16px}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;padding:12px 16px;border-top:1px solid var(--line)}i.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0}.stat{background:#f4f7f6;padding:12px;border-radius:5px}.stat b{display:block;font-size:24px;margin:5px 0}.stat small{color:var(--muted)}p{line-height:1.8}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:9px 6px;border-bottom:1px solid var(--line)}th{font-weight:500;color:var(--muted)}.bar{height:11px;display:flex;border-radius:6px;overflow:hidden;background:#eee}.bar span{height:100%}details{margin-top:18px;border:1px solid var(--line);border-radius:8px;background:white;padding:14px 18px}summary{cursor:pointer;font-weight:600}.small{font-size:12px;color:var(--muted);line-height:1.8}.metric{font-variant-numeric:tabular-nums}.tag{display:inline-block;padding:3px 8px;border-radius:3px;background:#edf3f3;font-size:12px;color:#4e6b72}a{color:#1c6e75}.empty{color:var(--muted)}
@media(max-width:850px){.grid{grid-template-columns:1fr}.views{gap:6px}.title{padding:8px;font-size:12px}.caption{padding:8px;font-size:11px}.canvaswrap{height:420px}main{padding:14px}header{padding:22px 16px}}
</style>
<header><small>CONTACT PREFERENCE / STAGE 03</small><h1>哪些点有支持，哪些还不知道？</h1><p>3 段已有示范 · 12 个运输时刻 · 真实表面映射与证据小试 · 未训练评分网络</p></header>
<main>
<div class="notice"><b>这里的颜色不是最终模型评分，也不是已确认接触。</b> 绿色表示保守正支持，红色表示当前姿态下的分离支持，蓝色表示未知。接近夹爪的点可以仍然是未知；不把遮挡、缺观测或旧候选直接当成正例。</div>
<div id="caseTabs" class="toolbar"></div><div class="toolbar frames"><label>原始回放帧</label><div id="frameTabs" class="toolbar" style="margin:0"></div><span id="qinfo" class="tag"></span></div>
<div class="views">
 <div class="card"><div class="title">① 当前帧 · 主相机</div><div class="viewbox"><img id="agent" alt="当前主相机原始RGB"><span class="cross" id="mark0"></span></div><div class="caption">黄色圈只定位选中点的投影；遮挡时可能落在夹爪上，不表示夹爪像素是物体。</div></div>
 <div class="card"><div class="title">② 同一帧 · 腕相机</div><div class="viewbox"><img id="wrist" alt="当前腕相机原始RGB"><span class="cross" id="mark1"></span></div><div class="caption">保留完整原图。没有把画面中全部邻近物体或机器人都列为目标。</div></div>
 <div class="card"><div class="title">③ 夹前真实来源 · 主相机</div><div class="viewbox"><img id="query" alt="夹前真实来源RGB"><span class="cross" id="mark2"></span></div><div class="caption">点云每个点都来自此帧实际深度。点击点云可追溯原始像素；没有生成完整补全面。</div></div>
</div>
<div class="grid">
<div class="card"><div class="title">映射到当前夹爪坐标的物体表面</div>
<div class="toolbar" style="padding:0 14px"><select id="mode" aria-label="点云颜色"><option value="class">三类概览 · 最大支持项</option><option value="0">正支持 · 0—1 固定色标</option><option value="1">分离支持 · 0—1 固定色标</option><option value="2">未知 · 0—1 固定色标</option></select><select id="visibility" aria-label="可见性"><option value="all">全部真实来源</option><option value="visible">当前深度一致可见</option><option value="hidden">当前遮挡 / 深度不一致</option></select><button id="reset">复位视角</button></div>
<div class="canvaswrap"><canvas id="cloud"></canvas><div class="canvashelp">拖动旋转 · 滚轮缩放 · 单击选点　/　灰色网格：夹爪几何代理</div></div>
<div class="legend"><span><i class="dot" style="background:var(--green)"></i>正支持</span><span><i class="dot" style="background:var(--red)"></i>分离支持</span><span><i class="dot" style="background:var(--blue)"></i>未知</span><span id="renderCount"></span></div>
</div>
<div class="card"><div class="title">这一帧的结果</div><div class="content">
<div id="stats" class="stats"></div><div id="summaryText"></div><hr style="border:0;border-top:1px solid var(--line);margin:18px 0">
<b id="selectedTitle">点的证据</b><p class="small">类别仅用于浏览：连续正支持、分离支持、未知之和为 1。未知不是负例；分离也不代表这个位置永远不能接触。</p>
<div id="pointInfo"></div>
</div></div>
</div>
<details open><summary>为什么没有绿色占优的点？这次验证说明了什么</summary><p>本轮没有正支持占优的点，不能作为已完成的正标签生成器。原因不只是案例少：先验0.1与“局部误差尺度内正间隙”这一组解释，使非穿透点的正支持下界理论上最多约0.115；取整个模型族最小值时，这个上限始终存在。即使距离为0也不能产生强正支持。三段视频的运动对照也不足以消除该歧义。<b>距离近、运动一致仍不够证明物理接触。</b> 所以不能把候选片直接涂成强正例，也不能因本轮无强正点而断言实际没有接触。后续应改进可识别的接触证据模型；本轮不事后删掉不利先验或缩窄间隙来制造绿色。</p><div id="motionTable"></div><p class="small">M0：用物体过去位移预测下一位移；M1：再加入实测夹爪诱导位移。误差单位为 mm/回放帧，显示三维各分量均方误差的平方根。按整段示范留出，正则强度只在其余训练示范内部选取。错时/倒序是关联对照，不是失败真值。优势很小不等于已确认接触；白黄杯只有一个转移块，对照无法识别。</p></details>
<details><summary>支持如何生成、哪些结果仍未验收</summary><p>夹前真实点 → 已有跨帧映射与留一敏感性 → 到夹爪实际碰撞代理表面的距离 → 接触零间隙与分离正间隙两类解释 → 多组噪声、先验和几何差异的包络 [L,U]。正支持=L，分离支持=1−U，未知=U−L。缺运动支持令 L=0；落入实体代理内部令 [L,U]=[0,1]。这些数值是指定模型族内的敏感性支持，<b>没有校准为真实接触概率</b>。</p><p>当前深度一致来自既有可见性检查；不一致可由遮挡或配准问题造成。完整真实查询表面参与计算，不只处理旧有限夹指区域候选。几何仅使用机器人模型，不读取仿真物体姿态、物体mesh或接触真值。</p><p>这次没有训练评分网络，没有使用人工区间，也没有访问预留测试。没有生成时长、多视频偏好目标或接触面积；仅验证它们之前的证据支持是否可用。抽屉、开关、双物体来源身份未通过的区域仍保持未知，不由这三段运输案例替代验收。</p><p><a href="夹爪物体接触偏好学习_完整方案.md">查看同一份完整方案与本轮结果</a> · <a href="接触偏好_夹持后运输多帧点云.html">原运输候选页</a></p></details>
<p class="small">页面生成自远程保存的逐点支持数组；不按人工区间选阈值、不用当前预测重造标签。原始回放帧号保留，不能把不同示范同帧号对齐。当前试验仅支持有限范围工程验收。</p>
</main>
<script>
const S=__SUMMARY__,base='图示/阶段3支持/',colors=[[22,128,93],[188,84,75],[85,125,181]],names=['正支持','分离支持','未知'];
let ci=0,fi=0,D=null,row=null,sel=-1,mode='class',vis='all',yaw=-.55,pitch=.28,zoom=1,screen=[],drag=null,moved=false,serial=0;
const $=id=>document.getElementById(id),cv=$('cloud'),ctx=cv.getContext('2d');
function button(text,active,fn){let b=document.createElement('button');b.textContent=text;if(active)b.className='active';b.onclick=fn;return b}
function tabs(){$('caseTabs').replaceChildren(...S.cases.map((c,i)=>button(c.title+' · '+c.demo,i===ci,()=>{ci=i;fi=0;tabs();load()})));$('frameTabs').replaceChildren(...S.cases[ci].rows.map((r,i)=>button('第 '+r.frame+' 帧',i===fi,()=>{fi=i;tabs();load()})))}
function marker(id,p){let e=$(id);if(p&&p[0]>=0&&p[1]>=0&&p[0]<448&&p[1]<448){e.style.display='block';e.style.left=p[0]/448*100+'%';e.style.top=p[1]/448*100+'%'}else e.style.display='none'}
async function load(){let seq=++serial;let c=S.cases[ci];row=c.rows[fi];D=null;sel=-1;$('stats').innerHTML='';$('summaryText').textContent='正在读取当前帧…';$('selectedTitle').textContent='点的证据';$('renderCount').textContent='';for(let id of ['mark0','mark1','mark2'])marker(id,null);draw();$('pointInfo').textContent='读取逐点证据…';$('agent').src=base+'images/'+row.webstem+'_agentview.jpg';$('wrist').src=base+'images/'+row.webstem+'_robot0_eye_in_hand.jpg';$('query').src=base+'images/c'+ci+'_query.jpg';$('qinfo').textContent='来源第 '+row.query_frame+' 帧 → 当前第 '+row.frame+' 帧';let data=await fetch(base+'webdata/'+row.webstem+'.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()});if(seq!==serial)return;D=data;
$('stats').innerHTML=['positive','separation','unknown'].map((k,j)=>'<div class="stat"><small>'+names[j]+'占优</small><b style="color:rgb('+colors[j]+')">'+row.dominant_count[k]+'</b><small>个来源点</small></div>').join('');
$('summaryText').innerHTML='<p>共 <b>'+row.n+'</b> 个夹前真实来源点；当前深度一致 '+row.current_depth_consistent+' 个，其余 '+row.occluded_or_depth_inconsistent+' 个保持遮挡/对应不确定标记。</p><p>本帧正支持最大值 <b>'+row.positive_max.toFixed(3)+'</b>。'+(row.motion_positive_basis?'有弱运动关联支持，仍不等于接触真值。':'独立运动对照不足，正支持下界保留 0。')+'</p><p class="small">穿入碰撞代理的 '+row.inside_proxy_unknown+' 点置未知，未把穿透当成强接触。</p>';
sel=D.supports.reduce((best,v,i)=>v[0]>D.supports[best][0]?i:best,0);
if(row.positive_max===0)sel=D.distance_mm.indexOf(Math.min(...D.distance_mm));
showPoint();draw()}
function showPoint(){if(!D||sel<0)return;let a=D.supports[sel];$('selectedTitle').textContent='来源点 #'+sel+' · '+names[D.category[sel]]+'占优';$('pointInfo').innerHTML='<div class="bar">'+a.map((v,j)=>'<span style="width:'+v*100+'%;background:rgb('+colors[j]+')"></span>').join('')+'</div><table>'+a.map((v,j)=>'<tr><th>'+names[j]+'</th><td class="metric">'+v.toFixed(4)+'</td></tr>').join('')+'<tr><th>到夹爪代理表面</th><td>'+D.distance_mm[sel].toFixed(2)+' mm</td></tr><tr><th>误差敏感性尺度</th><td>'+D.sigma_mm[sel].toFixed(2)+'—'+D.sigma_proxy_mm[sel].toFixed(2)+' mm</td></tr><tr><th>当前深度关系</th><td>'+(D.visible[sel]?'一致可见（自动证据）':'遮挡 / 深度不一致')+'</td></tr><tr><th>碰撞实体代理内部</th><td>'+(D.inside[sel]?'是 → 未知':'否')+'</td></tr><tr><th>来源像素 (u,v)</th><td>'+D.source_pixel[sel].join(', ')+'</td></tr></table>';
marker('mark0',D.projection[0][sel]);marker('mark1',D.projection[1][sel]);marker('mark2',D.source_pixel[sel])}
function draw(){const rect=cv.getBoundingClientRect(),dpr=devicePixelRatio||1;cv.width=rect.width*dpr;cv.height=rect.height*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);let w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);if(!D)return;
let pts=D.points,center=[0,0,0];for(let p of pts)for(let j=0;j<3;j++)center[j]+=p[j]/pts.length;let radius=Math.max(...pts.map(p=>Math.hypot(...p.map((v,j)=>v-center[j])))),scale=Math.min(w,h)*.36/Math.max(radius,25)*zoom;
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],xx=x*Math.cos(yaw)+y*Math.sin(yaw),yy=-x*Math.sin(yaw)+y*Math.cos(yaw);let zz=z*Math.cos(pitch)-yy*Math.sin(pitch),depth=z*Math.sin(pitch)+yy*Math.cos(pitch);return [w/2+xx*scale,h/2-zz*scale,depth]}
ctx.strokeStyle='#aebdbd66';ctx.lineWidth=.6;let rob=D.robot.map(project);for(let f of D.faces){ctx.beginPath();for(let j=0;j<4;j++){let p=rob[f[j%3]];if(j===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1])}ctx.stroke()}
screen=[];for(let i=0;i<pts.length;i++){if(vis==='visible'&&!D.visible[i]||vis==='hidden'&&D.visible[i])continue;screen.push({i,p:project(pts[i])})}screen.sort((a,b)=>a.p[2]-b.p[2]);
for(let o of screen){let i=o.i,c;if(mode==='class')c=colors[D.category[i]];else{let j=+mode,v=D.supports[i][j];c=colors[j].map(x=>Math.round(232*(1-v)+x*v))}ctx.fillStyle='rgb('+c+')';ctx.globalAlpha=D.visible[i]?1:.7;ctx.beginPath();ctx.arc(o.p[0],o.p[1],D.visible[i]?2.2:1.7,0,Math.PI*2);ctx.fill()}ctx.globalAlpha=1;
if(sel>=0){let p=project(pts[sel]);ctx.strokeStyle='#e5a11c';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p[0],p[1],7,0,Math.PI*2);ctx.stroke()}
$('renderCount').textContent='当前显示 '+screen.length+' 点';}
cv.onpointerdown=e=>{drag=[e.clientX,e.clientY];moved=false;cv.setPointerCapture(e.pointerId)};
cv.onpointermove=e=>{if(!drag)return;let dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(Math.abs(dx)+Math.abs(dy)>2)moved=true;yaw+=dx*.008;pitch+=dy*.008;drag=[e.clientX,e.clientY];draw()};
cv.onpointerup=e=>{drag=null;if(moved||!D)return;let r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,best=null,dist=100;for(let p of screen){let d=(p.p[0]-x)**2+(p.p[1]-y)**2;if(d<dist){dist=d;best=p.i}}if(best!==null){sel=best;showPoint();draw()}};
cv.onwheel=e=>{e.preventDefault();zoom=Math.max(.4,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()};
$('mode').onchange=e=>{mode=e.target.value;draw()};$('visibility').onchange=e=>{vis=e.target.value;draw()};$('reset').onclick=()=>{yaw=-.55;pitch=.28;zoom=1;draw()};window.addEventListener('resize',draw);
$('motionTable').innerHTML='<table><thead><tr><th>示范</th><th>时间转移块</th><th>M0</th><th>M1</th><th>错时</th><th>倒序</th></tr></thead><tbody>'+S.motion.map((m,i)=>'<tr><td>'+S.cases[i].title+'</td><td>'+m.n_temporal_blocks+'</td>'+['M0','M1','roll','reverse'].map(k=>'<td>'+(Math.sqrt(m.mse_m2_per_frame2[k])*1000).toFixed(3)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
tabs();load().catch(e=>{$('pointInfo').textContent='页面数据加载失败：'+e.message});
</script></html>'''
if __name__=="__main__":
 try:main()
 except Exception as e:
  (O/"gallery_failure.log").write_text(traceback.format_exc());print(json.dumps({"error":type(e).__name__}))
  raise SystemExit(1)
