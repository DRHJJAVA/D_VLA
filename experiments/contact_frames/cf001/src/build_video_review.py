from common import *
import numpy as np,torch,h5py,subprocess,re
from PIL import Image,ImageDraw,ImageFont
from model import ContactFrameNet
import imageio_ffmpeg
def main():
 cfg=json.loads((RUN/"config.json").read_text());z=np.load(RUN/"artifacts/dataset.npz");old=np.load(RUN/"artifacts/frame_predictions.npz")
 dest=RUN/"review_video";dest.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.08)
 images=torch.tensor(z["images"],device="cuda");points=torch.tensor(z["points"],device="cuda");ix=torch.tensor(old["context_indices"],device="cuda")
 preds=[];hashes={}
 for seed in cfg["ensemble_seeds"]:
  path=RUN/"artifacts"/("seed_"+str(seed))/"last.pt";hashes[str(seed)]=sha(path)
  ck=torch.load(path,map_location="cpu",weights_only=False);net=ContactFrameNet(ck["state_dim"]).cuda();net.load_state_dict(ck["model"]);net.eval()
  st=torch.tensor((z["states"]-ck["state_mean"])/ck["state_std"],device="cuda");vals=[]
  with torch.inference_mode():
   for start in range(0,len(st),16):
    j=ix[start:start+16];out,_=net(images[j],points[j],st[j]);vals.append(out.sigmoid().cpu().numpy())
  preds.append(np.concatenate(vals));del net;torch.cuda.empty_cache()
 pred=np.stack(preds);diff=float(np.max(np.abs(pred-old["probabilities"])));assert diff<1e-6
 np.savez_compressed(dest/"reinference.npz",probabilities=pred)
 scores=pred.mean(0);lo=pred.min(0);hi=pred.max(0);n=len(scores)
 rel=z["geometry_reliable"]&z["state_valid"]
 contact=(lo>=.8)&rel;sep=(hi<=.2)&rel&(z["distance"]-2*z["sigma"]>.006);unknown=~(contact|sep)
 assert np.array_equal(contact,old["contact_candidate"]) and np.array_equal(unknown,old["unknown"])
 conflict=contact&(z["target"]<=.2)&(z["weight"]>0)
 times=z["frame_time"];dt=np.median(np.diff(times));assert np.max(np.abs(np.diff(times)-dt))<1e-5
 fps=float(1/dt);video_path=ROOT/"docs/图示/CF001_接触时段预测.mp4"
 poster_path=ROOT/"docs/图示/CF001_接触时段预测_示例.jpg"
 assert not video_path.exists()
 fontpath="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
 font=ImageFont.truetype(fontpath,19);small=ImageFont.truetype(fontpath,16);big=ImageFont.truetype(fontpath,24)
 ff=imageio_ffmpeg.get_ffmpeg_exe()
 cmd=[ff,"-y","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","rgb24","-s","896x656","-r",str(fps),"-i","-","-an","-c:v","libx264","-crf","20","-preset","fast","-pix_fmt","yuv420p","-movflags","+faststart",str(video_path)]
 colors={"contact_candidate":(239,183,63),"observed_separation_candidate":(95,168,222),"unknown":(150,160,174)}
 states=["contact_candidate" if contact[t] else "observed_separation_candidate" if sep[t] else "unknown" for t in range(n)]
 names={"contact_candidate":"接触候选（不是成功真值）","observed_separation_candidate":"已见表面分离候选","unknown":"未知：证据不足或模型分歧"}
 rows=[]
 with (dest/"encode.log").open("w") as log, h5py.File(ROOT/cfg["rgbd"],"r") as f:
  process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=log)
  for t in range(n):
   im=Image.new("RGB",(896,656),(18,24,34));im.paste(Image.fromarray(f["agentview/rgb"][t]),(0,64));im.paste(Image.fromarray(f["robot0_eye_in_hand/rgb"][t]),(448,64));draw=ImageDraw.Draw(im)
   draw.text((12,4),"CF001  视频接触时段预测",font=big,fill="white")
   draw.text((12,36),"主相机：模型视觉输入",font=small,fill=(205,217,231));draw.text((460,36),"腕相机：仅辅助查看",font=small,fill=(205,217,231))
   draw.text((12,519),f"帧 {t:03d}/{n-1}  原记录时间 {times[t]:.3f}s  |  "+names[states[t]],font=font,fill=colors[states[t]])
   draw.text((12,551),f"模型均分 {scores[t]:.3f}   三个初始化范围 [{lo[t]:.3f}, {hi[t]:.3f}]   "+("训练帧" if z["train"][t] else "同视频留出帧" if z["audit"][t] else "手状态未知"),font=font,fill="white")
   note="与低几何兼容弱目标冲突；不是已确认误报" if conflict[t] else "几何弱目标只作对照，不是接触真值；灰色不可当作失败"
   draw.text((12,582),note,font=small,fill=(255,121,121) if conflict[t] else (183,195,212))
   for j in range(n):
    x0=12+j*872/n;x1=12+(j+1)*872/n
    draw.rectangle([x0,618,x1,636],fill=colors[states[j]])
    if conflict[j]:draw.rectangle([x0,638,x1,642],fill=(243,77,91))
   xx=12+(t+.5)*872/n;draw.line([xx,611,xx,646],fill="white",width=2)
   process.stdin.write(np.asarray(im).tobytes())
   if t==95:im.save(poster_path,quality=92)
   def finite(v):return float(v) if np.isfinite(v) else None
   rows.append({"frame":t,"seconds":float(times[t]),"score":float(scores[t]),"low":float(lo[t]),"high":float(hi[t]),"state":states[t],"state_label":names[states[t]],"conflict":bool(conflict[t]),"weak_target":float(z["target"][t]) if z["weight"][t]>0 else None,"distance_mm":finite(z["distance"][t]*1000),"uncertainty_mm":finite(z["sigma"][t]*1000),"split":"训练帧" if z["train"][t] else "同视频留出帧" if z["audit"][t] else "手状态未知"})
  process.stdin.close();ret=process.wait(timeout=120);assert ret==0
 save(dest/"frames.json",rows)
 summary=json.loads((RUN/"artifacts/model_summary.json").read_text())
 html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CF001 视频预测查看</title>
<style>body{margin:0;background:#101722;color:#eef4fd;font:16px system-ui,sans-serif}main{max-width:1160px;margin:28px auto;padding:0 22px}h1{font-size:26px;margin-bottom:8px}.muted{color:#aebfd5;line-height:1.7}.layout{display:grid;grid-template-columns:minmax(0,896px) 220px;gap:20px}video{width:100%;border-radius:10px;background:black}.card{background:#192435;padding:18px;border-radius:10px;margin:16px 0}.controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap}button,select,input{font:inherit}button,select{padding:8px 12px;border:1px solid #485e7c;border-radius:6px;color:#eef4fd;background:#23334a;cursor:pointer}input[type=range]{width:100%;accent-color:#edb73f}b{font-variant-numeric:tabular-nums}.value{font-size:32px;margin:10px 0;color:#edb73f}.bad{color:#ff858f}.blue{color:#6db3e6}.gray{color:#abb5c2}a{color:#97c8ff}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:9px;border-bottom:1px solid #34465c}canvas{width:100%;height:75px;cursor:pointer}code{background:#25364b;padding:2px 5px}@media(max-width:900px){.layout{grid-template-columns:1fr}}
</style><main><h1>CF001：逐帧接触预测</h1><p class="muted">tomato_sauce_demo_7 · 133 帧 · 训练用过的视频，含同视频留出时间块。黄色为模型候选，蓝色为已见表面分离候选，灰色为未知。没有接触真值，当前模型未通过可靠性检查。</p>
<div class="layout"><section><video id="video" controls playsinline preload="auto" poster="图示/CF001_接触时段预测_示例.jpg" src="图示/CF001_接触时段预测.mp4"></video>
<div class="card"><div class="controls"><button id="prev">上一帧</button><button id="play">播放 / 暂停</button><button id="next">下一帧</button><label>速度 <select id="speed"><option value=".25">0.25×</option><option value=".5" selected>0.5×</option><option value="1">1×</option></select></label><label>帧 <input id="number" type="number" min="0" max="132" value="0" style="width:70px"></label><button id="go">跳转</button></div><input id="seek" type="range" min="0" max="132" value="0"><canvas id="timeline" width="1064" height="75"></canvas><p class="muted">上条：区间分类；下方曲线：模型平均分。点击时间线跳帧。红线标出与低几何兼容目标冲突的帧。</p></div></section>
<aside class="card"><b id="frame">帧 0</b><div class="value" id="score"></div><p id="status"></p><p id="range"></p><p id="split"></p><p id="weak"></p><p id="distance"></p><p class="bad" id="conflict"></p></aside></div>
<div class="card"><b>建议先看这三处</b><div class="controls" style="margin-top:12px"><button data-frame="30">30帧：较低预测</button><button data-frame="39">39帧：冲突开始</button><button data-frame="95">95帧：长候选区间</button><button data-frame="60">60帧：未知</button></div><p class="muted">39、40、46–49、51帧的高分与低几何兼容弱目标冲突。这提示模型问题，但弱目标也不是物理真值，不能据此宣称这些帧均为真实误报。95–120帧是较长候选段，仍需结合画面审查。</p></div>
<div class="card"><b>全部接触候选区间（含首尾帧）</b><table><thead><tr><th>帧区间</th><th>帧数</th><th>原记录时间</th><th>查看</th></tr></thead><tbody id="ranges"></tbody></table></div>
<p class="muted">本页只展示冻结模型重新推理，不进行训练、修改标签或宣布阶段通过。网络设计与迭代记录仍维护在<a href="小型验证方案_夹持运动证据与区域表示_v1.md">原 Markdown</a>。腕相机仅供查看；模型视觉输入为主相机。</p></main>
<script>const rows=__ROWS__, intervals=__INTERVALS__, fps=__FPS__;const video=document.getElementById('video'),seek=document.getElementById('seek'),num=document.getElementById('number'),canvas=document.getElementById('timeline'),ctx=canvas.getContext('2d');let current=0;
function update(i){i=Math.max(0,Math.min(rows.length-1,i));current=i;const r=rows[i];seek.value=i;num.value=i;document.getElementById('frame').textContent='帧 '+i+' · 原记录 '+r.seconds.toFixed(3)+' s';document.getElementById('score').textContent=r.score.toFixed(3);document.getElementById('status').textContent=r.state_label;document.getElementById('range').textContent='初始化范围 '+r.low.toFixed(3)+'–'+r.high.toFixed(3);document.getElementById('split').textContent=r.split;document.getElementById('weak').textContent='几何弱目标：'+(r.weak_target===null?'未知':r.weak_target.toFixed(3));document.getElementById('distance').textContent='表面距离：'+(r.distance_mm===null?'未知':r.distance_mm.toFixed(2)+' mm');document.getElementById('conflict').textContent=r.conflict?'高分与低几何兼容弱目标冲突':'';draw(i)}
function jump(i){video.pause();i=Math.max(0,Math.min(132,Number(i)));video.currentTime=(i+.5)/fps;update(i)}
function draw(i){ctx.clearRect(0,0,1064,75);rows.forEach((r,j)=>{ctx.fillStyle=r.state==='contact_candidate'?'#edb73f':r.state==='unknown'?'#99a4b5':'#65a9da';ctx.fillRect(j*8,0,8,18);if(r.conflict){ctx.fillStyle='#f35060';ctx.fillRect(j*8,19,8,3)}});ctx.strokeStyle='#c9def5';ctx.beginPath();rows.forEach((r,j)=>{const y=71-r.score*43;j?ctx.lineTo(j*8+4,y):ctx.moveTo(4,y)});ctx.stroke();ctx.strokeStyle='white';ctx.beginPath();ctx.moveTo(i*8+4,0);ctx.lineTo(i*8+4,75);ctx.stroke()}
video.addEventListener('timeupdate',()=>update(Math.min(132,Math.floor(video.currentTime*fps+1e-4))));video.addEventListener('loadedmetadata',()=>{video.playbackRate=.5;update(0)});document.getElementById('play').onclick=()=>video.paused?video.play():video.pause();document.getElementById('prev').onclick=()=>jump(current-1);document.getElementById('next').onclick=()=>jump(current+1);document.getElementById('go').onclick=()=>jump(num.value);seek.oninput=()=>jump(seek.value);document.getElementById('speed').onchange=e=>video.playbackRate=Number(e.target.value);canvas.onclick=e=>jump(Math.floor((e.clientX-canvas.getBoundingClientRect().left)/canvas.getBoundingClientRect().width*133));document.querySelectorAll('[data-frame]').forEach(b=>b.onclick=()=>jump(b.dataset.frame));
intervals.forEach(r=>{let tr=document.createElement('tr');[r.start_frame+'–'+r.end_frame,r.frame_count,r.start_seconds.toFixed(3)+'–'+r.end_seconds.toFixed(3)+' s'].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.append(td)});let td=document.createElement('td'),b=document.createElement('button');b.textContent='跳到起点';b.onclick=()=>jump(r.start_frame);td.append(b);tr.append(td);document.getElementById('ranges').append(tr)});update(0);</script></html>'''
 html=html.replace("__ROWS__",json.dumps(rows,ensure_ascii=False)).replace("__INTERVALS__",json.dumps(summary["candidate_contact"])).replace("__FPS__",str(fps))
 page=ROOT/"docs/CF001_视频接触预测查看.html";assert not page.exists();page.write_text(html)
 # Decode output video and confirm actual frame count/dimensions.
 import cv2
 cap=cv2.VideoCapture(str(video_path));count=0
 while True:
  ok,fr=cap.read()
  if not ok:break
  assert fr.shape[:2]==(656,896);count+=1
 cap.release();assert count==n
 report={"time":now(),"episode":cfg["episode"],"frames":count,"fps":fps,"weights_sha256":hashes,"reinference_max_abs_difference":diff,"original_classification_identical":True,"video":str(video_path),"page":str(page),"video_bytes":video_path.stat().st_size,"counts":summary["counts"],"training_performed":False,"wrist_view_display_only":True}
 save(dest/"review_summary.json",report)
 lock={str(p):sha(p) for p in [video_path,poster_path,page,dest/"reinference.npz",dest/"frames.json",dest/"review_summary.json",RUN/"src/build_video_review.py"]}
 save(dest/"locked.json",lock)
 text=DOC.read_text();stamp=now();text=re.sub(r"- 最近修订时间：.*","- 最近修订时间："+stamp+"。",text)
 text+="\n### 14.10 冻结模型视频预测展示（"+stamp+"）\n\n应用户要求，加载三个冻结last权重，重新推理tomato_sauce_demo_7全部133帧。与已保存分数最大差"+str(diff)+"，区间分类逐项相同；没有训练或改标签。\n\n[打开视频预测查看页](CF001_视频接触预测查看.html)。含双相机完整画面、逐帧分数/三个初始化范围、候选/分离/未知时间线、帧跳转与0.25/0.5/1倍速。主相机是模型视觉输入，腕相机仅辅助查看。展示的是训练用过的视频，含同视频留出时间块，不是独立视频测试。\n\n原输出62接触候选帧、36已见表面分离候选帧、35未知帧；重点可查看39–51帧的低弱目标冲突与95–120帧长候选段。候选不是真实成功标签；保留模型问题，不以剪辑遮掩。模型本身对所有133帧都输出分数，未知来自记录中的证据/分歧拒识；视频显示完整帧序列。\n\n视频与展示产物均保留远程。review_video/locked.json保存展示哈希；网络设计和结果继续只维护本MD。\n"
 DOC.write_text(text);print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":
 try:main()
 except Exception as e:fail(e);raise SystemExit(1)
