from pathlib import Path
import json
from PIL import Image
from data import B,P,CAMS
PAGE='LCJP001_LIBERO_Object联合训练逐点评分.html'
def build_review(row,prediction_path,render_path,summary):
 import h5py,numpy as np
 directory=B/'review/v001';directory.mkdir(parents=True,exist_ok=True)
 frames=[]
 with h5py.File(prediction_path,'r') as pred,h5py.File(render_path,'r') as h:
  for t in range(row['frames']):
   images=[]
   for ci,cam in enumerate(CAMS):
    name='frame_%04d_cam%d.jpg'%(t,ci);Image.fromarray(h[cam+'/rgb'][t]).save(directory/name,quality=93);images.append(name)
   points=[]
   camera_data=[(h[c+'/K'][t],h[c+'/camera_to_world'][t],h[c+'/depth_m'][t]) for c in CAMS]
   for name,g in pred[str(t)].items():
    xyz=g['xyz'][:];score=g['score'][:];low=g['lower'][:];high=g['upper'][:];selected=np.zeros(len(xyz),bool)
    selected[np.argsort(-score,kind='stable')[:int(np.ceil(.8*len(score)))]]=True
    for i in range(len(xyz)):
     visible=[]
     for K,C,depth in camera_data:
      v=(xyz[i]-C[:3,3])@C[:3,:3];pix=np.rint(v[:2]/max(float(v[2]),1e-8)*[K[0,0],K[1,1]]+K[:2,2]).astype(int)
      good=v[2]>0 and (pix>=0).all() and (pix<256).all()
      visible.append(int(good and abs(float(v[2])-float(depth[pix[1],pix[0]]))<.008) if good else 0)
     points.append([*xyz[i].tolist(),float(score[i]),float(low[i]),float(high[i]),int(selected[i]),int(name),*visible])
   frames.append({'frame':t,'time':t/20,'images':images,'points':points,'cameras':[{'K':h[c+'/K'][t].tolist(),'C':h[c+'/camera_to_world'][t].tolist()} for c in CAMS]})
 (directory/'frames.json').write_text(json.dumps(frames,separators=(',',':')))
 (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 doc=P/'docs';link=doc/'lcjp001_assets'
 if not link.exists():link.symlink_to(directory,target_is_directory=True)
 html="""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LCJP001 实际模型逐点评分</title>
<style>body{font-family:system-ui,"Microsoft Yahei",sans-serif;background:#f5f6f8;color:#152238;max-width:1300px;margin:24px auto;padding:0 20px}h1{font-size:25px}p{line-height:1.65}.card{background:white;padding:18px;border-radius:12px;margin:14px 0;border:1px solid #dbe0e8}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}canvas{width:100%;background:#e8edf2;border-radius:8px}button,select,input{font-size:15px;padding:7px;margin:5px}input[type=range]{width:70%}.swatch{display:inline-block;width:22px;height:14px;margin:0 6px 0 14px}small{color:#58677c}#status{font-weight:600}#warning{background:#fff3cc;padding:12px}a{color:#1458aa}</style>
<h1>LCJP001 · LIBERO-Object 实际模型逐点评分</h1>
<p>冻结模型的当前帧预测。分数表示示范接触偏好估计，不是接触真值、当前接触概率或抓取成功率。每个当前物体提议保留排名前80%的点；颜色使用所有帧一致的绝对分数区间。</p>
<div id="warning">正在加载实际预测文件……</div>
<div class="card"><button id="play">播放</button><button id="prev">上一帧</button><button id="next">下一帧</button><label>帧号 <input id="frame" type="number" value="0" min="0" style="width:80px"></label><span id="status"></span><br><input id="slider" type="range" value="0" min="0">
<label>相机<select id="camera"><option value="0">主相机 agentview</option><option value="1">腕相机 robot0_eye_in_hand</option></select></label>
<label>物体提议<select id="object"><option value="-1">全部当前提议</option></select></label><br>
<label><input type="checkbox" id="top" checked>只强调前80%</label>
<label><input type="checkbox" id="uncertain" checked>宽不确定区间显示灰色</label>
<label><input type="checkbox" id="base" checked>显示完整点云背景</label>
<label>播放速度<select id="speed"><option value="200">0.25×</option><option value="100">0.5×</option><option value="50">1×（20帧/秒）</option></select></label>
</div>
<div class="card" id="legend"></div>
<div class="grid"><div class="card"><h3>当前 RGB 与真实点投影</h3><canvas id="rgb" width="640" height="640"></canvas><p>位置按本帧真实相机标定投影，并用当前深度排除该视角不可见点（8mm显示容差，不作为接触判断）。彩点是模型评分；没有自动监督或人工目标叠加。</p></div>
<div class="card"><h3>同帧点云评分</h3><canvas id="cloud" width="640" height="640"></canvas><p>默认与所选相机同方向，完整点云作浅灰背景。拖动旋转，双击恢复相机视角。</p></div></div>
<div class="card"><h3>本帧统计</h3><pre id="stats"></pre><small>“宽区间”只是展示规则：预测上下界宽度≥0.8；它不是经过接触真值校准的置信区间。可以关闭灰色显示以检查原始中心分数，但这不会使未知变可靠。前80%只是显示预算，低分和相同分数也可能入选。</small></div>
<div class="card"><h3>实际训练与测试记录</h3><pre id="summary" style="white-space:pre-wrap"></pre><a href="lcjp001_assets/summary.json">完整运行摘要</a></div>
<script>
const root='lcjp001_assets/',colors=['#313695','#74add1','#fee090','#f46d43','#a50026'];let data,summary,index=0,playing=false,timer=null,angleX=0,angleY=0,drag=null,token=0;
const el=id=>document.getElementById(id);
el('legend').innerHTML=colors.map((c,i)=>'<span class="swatch" style="background:'+c+'"></span>'+(.2*i).toFixed(1)+'–'+(.2*(i+1)).toFixed(1)).join('')+'<span class="swatch" style="background:#777"></span>未知/宽区间 <span class="swatch" style="background:#d0d6df"></span>未强调背景';
function color(p){return el('uncertain').checked&&p[5]-p[4]>=.8?'#777777':colors[Math.min(4,Math.floor(Math.max(0,p[3])*5))]}
function project(p,cam){let C=cam.C,q=[p[0]-C[0][3],p[1]-C[1][3],p[2]-C[2][3]],x=C[0][0]*q[0]+C[1][0]*q[1]+C[2][0]*q[2],y=C[0][1]*q[0]+C[1][1]*q[1]+C[2][1]*q[2],z=C[0][2]*q[0]+C[1][2]*q[1]+C[2][2]*q[2];return [x,y,z]}
function drawPoints(ctx,pts,cam,rotate){let pp=pts.map(p=>({p,q:project(p,cam)}));let center=[0,0,0];if(pp.length)for(let a of pp)for(let k=0;k<3;k++)center[k]+=a.q[k]/pp.length;
 for(let a of pp){let q=a.q;if(rotate){let x=q[0]-center[0],y=q[1]-center[1],z=q[2]-center[2];let x2=Math.cos(angleY)*x+Math.sin(angleY)*z,z2=-Math.sin(angleY)*x+Math.cos(angleY)*z;q=[x2+center[0],Math.cos(angleX)*y-Math.sin(angleX)*z2+center[1],Math.sin(angleX)*y+Math.cos(angleX)*z2+center[2]]}
 let K=cam.K;a.pixel=[(q[0]/q[2]*K[0][0]+K[0][2])*2.5,(q[1]/q[2]*K[1][1]+K[1][2])*2.5,q[2]]}
 pp.sort((a,b)=>b.pixel[2]-a.pixel[2]);for(let a of pp){let p=a.p,[u,v,z]=a.pixel;if(!rotate&&!p[8+(+el('camera').value)])continue;if(z<=0||u<0||u>=640||v<0||v>=640)continue;let high=!el('top').checked||p[6];if(!high&&!el('base').checked)continue;ctx.fillStyle=high?color(p):'#d0d6df';ctx.globalAlpha=high?.95:.45;ctx.beginPath();ctx.arc(u,v,high?2.5:1.7,0,2*Math.PI);ctx.fill()}ctx.globalAlpha=1}
async function show(){if(!data)return;let local=++token;index=Math.max(0,Math.min(data.length-1,index));let f=data[index],ci=+el('camera').value,old=el('object').value,ids=[...new Set(f.points.map(p=>p[7]))];el('object').innerHTML='<option value="-1">全部当前提议</option>'+ids.map(id=>'<option value="'+id+'">提议 '+id+'</option>').join('');el('object').value=ids.includes(+old)?old:'-1';
 let filter=+el('object').value,pts=f.points.filter(p=>filter<0||p[7]===filter);el('frame').value=index;el('slider').value=index;el('status').textContent='第 '+index+' / '+(data.length-1)+' 帧 · '+f.time.toFixed(2)+' 秒';
 let cv=el('cloud'),ct=cv.getContext('2d');ct.fillStyle='#edf1f5';ct.fillRect(0,0,640,640);drawPoints(ct,pts,f.cameras[ci],true);
 let im=new Image();im.src=root+f.images[ci];await im.decode();if(local!==token)return;let cr=el('rgb').getContext('2d');cr.drawImage(im,0,0,640,640);drawPoints(cr,pts,f.cameras[ci],false);
 let counts=[0,0,0,0,0],unc=0;for(let p of pts){counts[Math.min(4,Math.floor(p[3]*5))]++;if(p[5]-p[4]>=.8)unc++}
 el('stats').textContent=JSON.stringify({真实来源点:pts.length,前80百分比入选点:pts.filter(p=>p[6]).length,各绝对分数带点数:counts,宽不确定区间点数:unc,最小分数:pts.length?pts.reduce((a,p)=>Math.min(a,p[3]),Infinity):null,最大分数:pts.length?pts.reduce((a,p)=>Math.max(a,p[3]),-Infinity):null},null,2)}
el('frame').onchange=()=>{index=+el('frame').value;show()};el('slider').oninput=()=>{index=+el('slider').value;show()};el('prev').onclick=()=>{index--;show()};el('next').onclick=()=>{index++;show()};
for(let id of ['camera','object','top','uncertain','base'])el(id).onchange=()=>show();
function tick(){if(!playing)return;index=(index+1)%data.length;show();timer=setTimeout(tick,+el('speed').value)}
el('play').onclick=()=>{playing=!playing;el('play').textContent=playing?'暂停':'播放';if(playing)tick();else clearTimeout(timer)};
el('cloud').onpointerdown=e=>{drag=[e.clientX,e.clientY];el('cloud').setPointerCapture(e.pointerId)};el('cloud').onpointermove=e=>{if(!drag)return;angleY+=(e.clientX-drag[0])*.008;angleX+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];show()};el('cloud').onpointerup=()=>drag=null;el('cloud').ondblclick=()=>{angleX=angleY=0;show()};
Promise.all([fetch(root+'frames.json').then(r=>r.json()),fetch(root+'summary.json').then(r=>r.json())]).then(([d,s])=>{data=d;summary=s;el('slider').max=el('frame').max=d.length-1;el('summary').textContent=JSON.stringify(s,null,2);el('warning').textContent='实际模型权重：'+s.best_sha256+'。测试视频：'+s.case+'。是否学到有效偏好仍需结合完整分布和不确定性复核，不能仅凭彩色点判断。';show()}).catch(e=>el('warning').textContent='加载失败：'+e.message)
</script></html>"""
 (P/'docs'/PAGE).write_text(html,encoding='utf-8')
 return {'page':str(P/'docs'/PAGE),'frames':len(frames),'images':2*len(frames),'prediction_file':str(prediction_path)}
