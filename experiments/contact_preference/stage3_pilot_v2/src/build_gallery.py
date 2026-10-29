from pathlib import Path
import json,importlib.util,traceback
import numpy as np
R=Path("/mnt/nvme1/drh/D_VLA");O=R/"experiments/contact_preference/stage3_pilot_v2";OLD=R/"experiments/contact_preference/stage3_pilot_v1"
def main():
 S=json.loads((O/"evidence_summary.json").read_text())
 for ci,ca in enumerate(S["cases"]):
  for row in ca["rows"]:
   t=row["frame"];z=np.load(O/row["data"]);stem="c"+str(ci)+"_f"+str(t);old=json.loads((OLD/"webdata"/(stem+".json")).read_text())
   old.update(points=np.round(z["mapped_hand_xyz"]*1000,3).tolist(),supports=np.round(np.column_stack([z["positive"],z["separation"],z["unknown"]]),6).tolist(),
    category=z["display_class"].tolist(),distance_mm=np.round(z["distance_m"]*1000,3).tolist(),signed_mm=np.round(z["signed_distance_m"]*1000,3).tolist(),
    sigma_mm=np.round(z["sigma_base_m"]*1000,3).tolist(),sigma_proxy_mm=np.round(z["sigma_proxy_m"]*1000,3).tolist(),
    mounting=z["excluded_mounting_nearest"].astype(int).tolist(),deep=z["deep_penetration"].astype(int).tolist(),
    opposing=z["opposing_normals"].astype(int).tolist(),dot12=np.round(z["normal_dot_k12"],4).tolist(),dot24=np.round(z["normal_dot_k24"],4).tolist(),
    geom=z["nearest_gripper_geom"].tolist(),robot=np.round(z["gripper_vertices"]*1000,3).tolist(),faces=z["gripper_faces"].tolist())
   (O/"webdata"/(stem+".json")).write_text(json.dumps(old,separators=(",",":")));row["webstem"]=stem
 (O/"webdata/summary.json").write_text(json.dumps(S,ensure_ascii=False,separators=(",",":")))
 spec=importlib.util.spec_from_file_location("oldgallery",OLD/"src/build_gallery.py");g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 html=g.HTML[:g.HTML.index("<script>")]
 html=html.replace("阶段3 · 正支持、分离支持与未知","阶段3 · 夹爪专属弱支持对照")
 html=html.replace("哪些点有支持，哪些还不知道？","只查看夹爪与目标物体的接触支持")
 html=html.replace("3 段已有示范 · 12 个运输时刻 · 真实表面映射与证据小试 · 未训练评分网络","同样 3 个案例 / 12 帧 · 正支持机制与接触范围对照 · 未训练评分网络")
 start=html.index('<div class="notice">');end=html.index('</div>',start)+6
 html=html[:start]+'''<div class="notice"><b>绿色现在是清楚的实心点。</b> 只突出“夹指/指垫—目标物体”的正支持占优点；灰色未知、安装壳体最近点都不是接触候选。<br><b>绿色仍是自动弱支持，不是确认的物理接触：</b> 当前约毫米级误差内的微小分离仍可能混淆。没有通过变浅/变深颜色掩盖数量，也没有训练评分网络。</div>'''+html[end:]
 html=html.replace('三类概览 · 最大支持项','三类支持与范围排除')
 html=html.replace('<option value="class">','<option value="positive">只突出夹爪正支持 · 其余灰色</option><option value="class">',1)
 html=html.replace('未知 · 0—1 固定色标','未知强度 · 灰色 0—1')
 html=html.replace('<button id="reset">复位视角</button>','<button id="reset">复位视角</button><label><input id="overlayToggle" type="checkbox" checked> 显示绿色点的图像投影</label>')
 for idx,name in enumerate(["agent","wrist","query"]):
  html=html.replace('<span class="cross" id="mark'+str(idx)+'"></span>','<canvas class="overlay" id="ov'+str(idx)+'"></canvas><span class="cross" id="mark'+str(idx)+'"></span>')
 html=html.replace('</style>','.viewbox .overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.cross{z-index:2}:root{--green:#009b64;--blue:#8e969b}.stats .stat b{font-size:24px}</style>')
 html=html.replace('黄色圈只定位选中点的投影；遮挡时可能落在夹爪上，不表示夹爪像素是物体。','绿点是正支持的当前投影，可能被夹指遮挡；不是把机器人像素判为物体。黄色圈定位选中点。')
 html=html.replace('保留完整原图。没有把画面中全部邻近物体或机器人都列为目标。','同一批目标物体来源点投影；不把其它物体、桌面或机械臂的接触并入。')
 html=html.replace('点云每个点都来自此帧实际深度。点击点云可追溯原始像素；没有生成完整补全面。','绿色来源像素来自夹前真实深度，不是补全点；当前被挡住的表面依靠已有对应映射。')
 html=html.replace('类别仅用于浏览：连续正支持、分离支持、未知之和为 1。未知不是负例；分离也不代表这个位置永远不能接触。','连续量是两种测量解释的弱证据，不是接触概率。未知与范围排除均不作为正例；分离仅指当前夹爪关系。')
 start=html.index('<details open>')
 html=html[:start]+'''<details open><summary>旧版与新版：绿色点数、范围与失败记录</summary><div id="comparison"></div><p>旧版三类概览确实没有绿色，不只是颜色浅。固定保守先验将正支持压在0.115以下。随后“全物体表面拟合接触比例”的对照也退化到几乎全分离，已作为失败记录保留。</p><p>当前展示改用<b>接触与分离解释各自多出来的似然证据</b>，共有部分归未知。没有把整个物体上接触面积稀少的先验乘到每个点上。未知用灰色，不再用容易被理解为接触区域的蓝色。</p></details>
<details><summary>只允许哪两个对象贡献正支持？</summary><p>一方是既有来源核查的目标物体真实表面；另一方只能是两根夹指的内侧、外侧表面及指垫。没有限定抬升或动作类型。安装壳体/腕/机械臂、桌面、其他物体不贡献正支持。</p><p>最近安装壳体的点单列“范围排除”，这是最近部件判定，不是断言它实际碰到了壳体。点云灰色的杯底、杯壁等位置也不表示与别处接触。夹指外侧仍允许贡献支持，不仅限于两指之间。</p></details>
<details open><summary>新支持怎么生成？还不能说明什么？</summary><p>同一夹前真实来源在多个时刻的有符号间距只汇总一次；误差保留时序相关性，不按帧数缩小。接触解释为0间隙加测量噪声；分离解释为从另外两段训练示范拟合的非负间隙分布加相同噪声。使用两种分离分布、误差模型及episode重采样检查敏感性。</p><p>令 p=fC/(fC+fS)，正支持=max(2p−1,0)，分离支持=max(1−2p,0)，共有解释2min(p,1−p)归未知。各模型及当前/汇总时刻之间取较弱支持；局部真实表面与夹指法向不对向时取消正支持，安装壳体最近或明显穿入时排除。<b>这改变了监督量的定义，不是校准后的物理接触概率。</b></p><p>合成检查中，3倍测量误差的分离间隙不再产生正支持；但约1倍误差内的微小分离仍可能有正支持。绿色只说明在当前测量分辨率下更符合接触解释，不能保证每点真实接触；阶段3的物理可靠性尚未验收，暂不进入评分训练。</p><div id="tests"></div><p><a href="夹爪物体接触偏好学习_完整方案.md">同一份完整方案与迭代记录</a> · <a href="接触偏好_阶段3_正负支持与未知.html">保留的上一版</a></p></details>
<p class="small">全部产物远程保存。沿用相同3个训练案例，不使用人工区间、物体仿真位姿/接触真值、保留测试；没有新增评分网络训练。</p></main>'''
 html+=JS.replace("__SUMMARY__",json.dumps(S,ensure_ascii=False,separators=(",",":")))
 link=R/"docs/图示/阶段3夹爪支持"
 if not link.exists():link.symlink_to(O,target_is_directory=True)
 (R/"docs/接触偏好_阶段3_夹爪专属正负支持对照.html").write_text(html)
 print("gallery_ready")
JS=r'''<script>
const S=__SUMMARY__,base='图示/阶段3夹爪支持/',imbase='图示/阶段3支持/images/',colors=[[0,155,100],[186,91,82],[142,150,155],[193,196,198],[135,140,143]],names=['夹爪正支持','分离支持','未知','最近安装壳体 · 范围排除','几何冲突 · 排除'];
let ci=0,fi=0,D=null,row=null,sel=-1,mode='positive',vis='all',yaw=-.55,pitch=.28,zoom=1,screen=[],drag=null,moved=false,serial=0;
const $=id=>document.getElementById(id),cv=$('cloud'),ctx=cv.getContext('2d');
function button(text,active,fn){let b=document.createElement('button');b.textContent=text;if(active)b.className='active';b.onclick=fn;return b}
function tabs(){$('caseTabs').replaceChildren(...S.cases.map((c,i)=>button(c.title+' · '+c.demo,i===ci,()=>{ci=i;fi=0;tabs();load()})));$('frameTabs').replaceChildren(...S.cases[ci].rows.map((r,i)=>button('第 '+r.frame+' 帧',i===fi,()=>{fi=i;tabs();load()})))}
function marker(id,p){let e=$(id);if(p&&p[0]>=0&&p[1]>=0&&p[0]<448&&p[1]<448){e.style.display='block';e.style.left=p[0]/448*100+'%';e.style.top=p[1]/448*100+'%'}else e.style.display='none'}
function overlays(){for(let j=0;j<3;j++){let el=$('ov'+j);el.width=448;el.height=448;let c=el.getContext('2d');c.clearRect(0,0,448,448);if(!D||!$('overlayToggle').checked)continue;for(let i=0;i<D.points.length;i++){if(D.category[i]!==0)continue;let p=j===2?D.source_pixel[i]:D.projection[j][i];if(p[0]<0||p[1]<0||p[0]>=448||p[1]>=448)continue;c.fillStyle='#00c67b';c.strokeStyle='#004f36';c.lineWidth=.55;c.beginPath();c.arc(p[0],p[1],1.8,0,Math.PI*2);c.fill();c.stroke()}}}
async function load(){let seq=++serial;let ca=S.cases[ci];row=ca.rows[fi];D=null;sel=-1;$('stats').innerHTML='';$('summaryText').textContent='读取当前帧…';$('selectedTitle').textContent='点的证据';$('pointInfo').textContent='读取中…';$('renderCount').textContent='';for(let j=0;j<3;j++)marker('mark'+j,null);draw();overlays();
$('agent').src=imbase+row.webstem+'_agentview.jpg';$('wrist').src=imbase+row.webstem+'_robot0_eye_in_hand.jpg';$('query').src=imbase+'c'+ci+'_query.jpg';$('qinfo').textContent='夹前来源 '+row.query_frame+' → 当前 '+row.frame;
try{let d=await fetch(base+'webdata/'+row.webstem+'.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()});if(seq!==serial)return;D=d;
$('stats').innerHTML=[0,1,2].map(j=>'<div class="stat"><small>'+names[j]+'占优</small><b style="color:rgb('+colors[j]+')">'+row.count[j]+'</b><small>个来源点</small></div>').join('');
$('summaryText').innerHTML='<p><b>'+row.count[0]+' 个绿色点</b>，当前深度一致 '+row.green_visible+' 个，当前遮挡/不一致 '+row.green_occluded+' 个。绿色是弱支持，非确认接触。</p><p>安装壳体最近 '+row.count[3]+' 点、几何冲突 '+row.count[4]+' 点均灰色排除，<b>不是接触候选</b>。</p><p class="small">正支持最大 '+row.positive_max.toFixed(3)+'；上一版最大 '+row.old_positive_max.toFixed(3)+'。当前量是似然证据，不能把数值增加当准确率提高。</p>';
sel=D.supports.reduce((best,v,i)=>v[0]>D.supports[best][0]?i:best,0);showPoint();draw();overlays();
}catch(e){if(seq===serial)$('pointInfo').textContent='加载失败：'+e.message}}
function showPoint(){if(!D||sel<0)return;let a=D.supports[sel];$('selectedTitle').textContent='来源点 #'+sel+' · '+names[D.category[sel]];$('pointInfo').innerHTML='<div class="bar">'+a.map((v,j)=>'<span style="width:'+v*100+'%;background:rgb('+colors[j]+')"></span>').join('')+'</div><table>'+a.map((v,j)=>'<tr><th>'+names[j]+'</th><td>'+v.toFixed(4)+'</td></tr>').join('')+
'<tr><th>最近夹指部件</th><td>'+({3:'左夹指表面',4:'左指垫',6:'右夹指表面',7:'右指垫'}[D.geom[sel]])+'</td></tr>'+
'<tr><th>安装壳体是否更近</th><td>'+(D.mounting[sel]?'是 → 排除':'否')+'</td></tr>'+
'<tr><th>有符号表面间距</th><td>'+D.signed_mm[sel].toFixed(2)+' mm</td></tr>'+
'<tr><th>误差尺度</th><td>'+D.sigma_mm[sel].toFixed(2)+'—'+D.sigma_proxy_mm[sel].toFixed(2)+' mm</td></tr>'+
'<tr><th>两尺度法向对向</th><td>'+(D.opposing[sel]?'一致':'不一致 → 不给正支持')+'</td></tr>'+
'<tr><th>当前深度关系</th><td>'+(D.visible[sel]?'一致可见':'遮挡 / 不一致')+'</td></tr>'+
'<tr><th>来源像素 (u,v)</th><td>'+D.source_pixel[sel].join(', ')+'</td></tr></table>';
marker('mark0',D.projection[0][sel]);marker('mark1',D.projection[1][sel]);marker('mark2',D.source_pixel[sel])}
function draw(){let rect=cv.getBoundingClientRect(),dpr=devicePixelRatio||1;cv.width=rect.width*dpr;cv.height=rect.height*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);let w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);if(!D)return;let pts=D.points,center=[0,0,0];for(let p of pts)for(let j=0;j<3;j++)center[j]+=p[j]/pts.length;let radius=Math.max(...pts.map(p=>Math.hypot(...p.map((v,j)=>v-center[j])))),scale=Math.min(w,h)*.36/Math.max(radius,25)*zoom;
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],xx=x*Math.cos(yaw)+y*Math.sin(yaw),yy=-x*Math.sin(yaw)+y*Math.cos(yaw),zz=z*Math.cos(pitch)-yy*Math.sin(pitch),dep=z*Math.sin(pitch)+yy*Math.cos(pitch);return [w/2+xx*scale,h/2-zz*scale,dep]}
ctx.strokeStyle='#607a7a88';ctx.lineWidth=.7;let rob=D.robot.map(project);for(let f of D.faces){ctx.beginPath();for(let j=0;j<4;j++){let p=rob[f[j%3]];if(j===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1])}ctx.stroke()}
screen=[];for(let i=0;i<pts.length;i++){if(vis==='visible'&&!D.visible[i]||vis==='hidden'&&D.visible[i])continue;screen.push({i,p:project(pts[i])})}screen.sort((a,b)=>a.p[2]-b.p[2]);
function point(o,top){let i=o.i,c,rad=1.6;if(mode==='positive'){let green=D.category[i]===0;if(green!==top)return;c=green?colors[0]:[196,201,202];rad=green?3:1.45;ctx.globalAlpha=green?1:.45}
else {if(top)return;ctx.globalAlpha=1;if(mode==='class'){c=colors[D.category[i]];rad=D.category[i]===0?2.8:1.7}else {let j=+mode,v=D.supports[i][j];c=colors[j].map(x=>Math.round(228*(1-v)+x*v));rad=2}}
ctx.fillStyle='rgb('+c+')';ctx.beginPath();ctx.arc(o.p[0],o.p[1],rad,0,Math.PI*2);ctx.fill()}
for(let o of screen)point(o,false);if(mode==='positive')for(let o of screen)point(o,true);ctx.globalAlpha=1;if(sel>=0){let p=project(pts[sel]);ctx.strokeStyle='#e0a016';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p[0],p[1],7,0,Math.PI*2);ctx.stroke()}
$('renderCount').textContent='显示 '+screen.length+' 点，其中绿色 '+screen.filter(o=>D.category[o.i]===0).length;}
cv.onpointerdown=e=>{drag=[e.clientX,e.clientY];moved=false;cv.setPointerCapture(e.pointerId)};
cv.onpointermove=e=>{if(!drag)return;let dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(Math.abs(dx)+Math.abs(dy)>2)moved=true;yaw+=dx*.008;pitch+=dy*.008;drag=[e.clientX,e.clientY];draw()};
cv.onpointerup=e=>{drag=null;if(moved||!D)return;let r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,best=null,dist=100;for(let p of screen){let d=(p.p[0]-x)**2+(p.p[1]-y)**2;if(d<dist){dist=d;best=p.i}}if(best!==null){sel=best;showPoint();draw()}};
cv.onwheel=e=>{e.preventDefault();zoom=Math.max(.4,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()};
$('mode').onchange=e=>{mode=e.target.value;draw()};$('visibility').onchange=e=>{vis=e.target.value;draw()};$('overlayToggle').onchange=overlays;$('reset').onclick=()=>{yaw=-.55;pitch=.28;zoom=1;draw()};window.addEventListener('resize',draw);
$('comparison').innerHTML='<table><tr><th>案例</th><th>旧版绿色</th><th>新版绿色 / 帧</th><th>未知 / 帧</th><th>范围排除 / 帧</th></tr>'+S.cases.map(c=>{function range(j){let a=c.rows.map(r=>r.count[j]);return Math.min(...a)+'—'+Math.max(...a)}return '<tr><td>'+c.title+' '+c.demo+'</td><td>0</td><td>'+range(0)+'</td><td>'+range(2)+'</td><td>'+range(3)+'</td></tr>'}).join('')+'</table>';
$('tests').innerHTML='<table><tr><th>合成间隙</th><th>瓷杯模型正支持</th><th>罐头模型正支持</th><th>白黄杯模型正支持</th></tr>'+[0,1,2,3].map(i=>'<tr><td>'+[0,1,3,10][i]+' × 测量误差</td>'+S.synthetic_tests.map(t=>'<td>'+t.positive[i].toFixed(3)+'</td>').join('')+'</tr>').join('')+'</table>';
tabs();load();
</script></html>'''
if __name__=="__main__":
 try:main()
 except Exception as e:
  (O/"gallery_failure.log").write_text(traceback.format_exc());print(json.dumps(dict(error=type(e).__name__,message=str(e))));raise SystemExit(1)
