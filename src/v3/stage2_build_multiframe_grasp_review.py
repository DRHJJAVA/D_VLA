from pathlib import Path
import json,numpy as np,h5py,cv2,hashlib,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage2_probe_motion_fingerpads import project
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'outputs/v3_stage2/multiframe_grasp_fusion_v1'
IMG=R/'docs/图示/多帧融合夹持区域';PAGE=R/'docs/第二步_多帧融合与夹持区域对照.html'
def vox(p,mm=.002):return p[np.unique(np.rint(p/mm).astype(int),axis=0,return_index=True)[1]] if len(p) else p
def xyzply(path,p):
 with path.open('x') as f:
  f.write('ply\nformat ascii 1.0\ncomment RGBD multi-frame aligned candidate surface; NOT contact truth\nelement vertex '+str(len(p))+'\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
  for x in p:f.write(' '.join(f'{v:.8f}' for v in x)+'\n')
def main():
 IMG.mkdir(parents=True,exist_ok=False);assert not PAGE.exists()
 refs={x['id']:x for x in json.loads((R/'audits/v3/finger_contact_repair_pilot_v1/protocol.json').read_text())['episodes']}
 summaries=json.loads((O/'summary.json').read_text());cases=[]
 for s in summaries:
  name=s['id']+'_attempt'+str(s['attempt']);z=np.load(O/(name+'.npz'))
  ref=z['anchor_surface_hand_m'];fused=z['fused_surface_hand_m'];reg=z['fused_region_hand_m'];areg=z['anchor_region_hand_m'];pre=z['uncertain_pre_hand_m'];center=ref.mean(0);H=z['hand_to_world'];lo=z['volume_min_hand_m'];hi=z['volume_max_hand_m']
  for label,p in [('融合物体',fused),('夹持区域',reg),('夹前待确认',pre)]:xyzply(O/(name+'_'+label+'.ply'),p@H[:3,:3].T+H[:3,3])
  boxes=np.array([[x,y,zz] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for zz in [lo[2],hi[2]]])
  def local(p):return ((p-center)*1000).round(3).tolist()
  # Remote static figure: equal projection/scale across before and after.
  fig=plt.figure(figsize=(12,5),facecolor='#101c28')
  surfaces=[ref,vox(fused)];regions=[areg,reg]
  allp=(np.r_[ref,fused]-center)*1000;span=max(np.ptp(allp,axis=0).max()/2,50)
  for k in range(2):
   ax=fig.add_subplot(1,2,k+1,projection='3d',facecolor='#101c28')
   p=(surfaces[k]-center)*1000;r=(regions[k]-center)*1000;b=(boxes-center)*1000
   ax.scatter(p[:,0],p[:,1],p[:,2],s=.5,c='#8096a8',alpha=.4)
   if len(r):ax.scatter(r[:,0],r[:,1],r[:,2],s=8,c='#35f16b',depthshade=False)
   for i in range(8):
    for j in range(i+1,8):
     if (i^j) in [1,2,4]:ax.plot(b[[i,j],0],b[[i,j],1],b[[i,j],2],c='white',lw=.6,ls='--')
   ax.set(xlim=(-span,span),ylim=(-span,span),zlim=(-span,span));ax.set_box_aspect([1,1,1]);ax.view_init(elev=15,azim=-35);ax.set_axis_off()
   ax.set_title(('Single frame' if k==0 else 'Multi-frame candidate')+f' | region {len(regions[k])}',color='white')
  fig.suptitle(name+' | green = inter-finger surface, not contact truth',color='white');fig.tight_layout();fig.savefig(IMG/(name+'_compare.png'),dpi=150,facecolor=fig.get_facecolor());plt.close(fig)
  gallery=[]
  if s['held']:
   rec=json.loads((O/(name+'.json')).read_text())['registration_records'];valid=[r for r in rec if r.get('valid')]
   preframes=[r for r in valid if r['phase']=='pre'];held=[r for r in valid if r['phase']=='held']
   selected=([preframes[0]] if preframes else [])+([held[0],held[-1]] if held else [])
   with h5py.File(refs[s['id']]['rgbd'],'r') as f:
    panels=[]
    for row in selected:
     u=row['frame'];cam_panels=[]
     for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
      g=f[cam];im=g['rgb'][u].copy();overlay=im.copy();use=z['accepted_sample_indices'];p=z['source_aligned_hand_m'][use]
      mask=(z['source_frame'][use]==u)&(z['source_camera'][use]==ci)&np.all(p>=lo,1)&np.all(p<=hi,1)
      pixels=z['source_pixel'][use][mask]
      if len(pixels):overlay[pixels[:,1],pixels[:,0]]=[30,245,70]
      # Show same object-coordinate center transformed back to the source frame.
      T=np.array(row['source_world_to_anchor_hand']);cw=np.r_[(lo+hi)/2,1]@np.linalg.inv(T).T
      pos,dep=project(cw[None,:3],g['K'][u],g['camera_to_world'][u]);cx,cy=np.rint(pos[0]).astype(int);w=180
      x0=max(0,min(im.shape[1]-w,cx-w//2));y0=max(0,min(im.shape[0]-w,cy-w//2))
      out=np.concatenate([cv2.resize(a[y0:y0+w,x0:x0+w],(320,320),interpolation=cv2.INTER_NEAREST) for a in [im,overlay]],1)
      cv2.rectangle(out,(0,0),(640,26),(10,20,30),-1);cv2.putText(out,f'{row["phase"]} t{u} {cam} | source / kept region {len(pixels)}',(6,18),0,.42,(255,255,255),1);cam_panels.append(out)
     panels.append(np.concatenate(cam_panels))
    if panels:
     src=IMG/(name+'_sources.jpg');cv2.imwrite(str(src),cv2.cvtColor(np.concatenate(panels,1),cv2.COLOR_RGB2BGR));gallery.append('图示/多帧融合夹持区域/'+src.name)
  cases.append(dict(name=name,held=s['held'],base=local(vox(ref)),fused=local(vox(fused)),baseRegion=local(areg),region=local(reg),pre=local(pre),box=local(boxes),summary=s,static='图示/多帧融合夹持区域/'+name+'_compare.png',gallery=gallery))
 html=r"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多帧融合与夹持区域对照</title><style>
body{max-width:1320px;margin:26px auto;padding:0 20px;font:17px/1.7 system-ui;background:#f4f6f8;color:#21313f}h1{font-size:29px}aside{background:#fff0c7;padding:16px;border-radius:8px}.controls{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}select,button{font:inherit;padding:7px}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:12px}canvas{background:#101c28;width:100%;height:510px;touch-action:none;border-radius:10px}h2{font-size:21px}img{max-width:100%;height:auto}#numbers{background:white;padding:14px}small{color:#4b6070}@media(max-width:800px){.twocol{grid-template-columns:1fr}}</style>
<h1>多帧融合后，再取两根夹指之间的区域</h1><aside><b>小样本尝试，尚未训练；补充了可见表面，不代表完整物体已恢复。</b><br>灰色：物体表面候选；绿色：同一个夹持范围内的表面。左为单帧，右为多帧融合。橙色可选显示“只有夹住前支持、尚未核实”的候选，不计入右侧绿色结果。</aside>
<div class="controls"><select id="case"></select><button id="reset">重置视角</button><label><input id="only" type="checkbox">只看夹持区域</label><label><input id="pre" type="checkbox">显示夹住前待确认点（橙色）</label></div>
<p id="numbers"></p><div class="twocol"><div><h2>单帧</h2><canvas id="before"></canvas></div><div><h2>多帧融合候选</h2><canvas id="after"></canvas></div></div>
<p>拖动任一图同步旋转，滚轮缩放。虚线盒的范围保持不变。绿色点为便于观察放大显示，不表示接触面积。</p>
<small>运动帧以夹爪运动作初值，再用颜色和深度几何配准；夹住前单独配准。新增点需要至少两个非参考帧支持，其中至少一个为夹持帧。仍可能存在物体滑动、对称表面配准误差或前景混入，不能把表面对齐当作身份真值。没有读取仿真物体位姿或接触真值，没有生成看不见的表面。</small>
<h2>来源帧核对</h2><p>每列是一个来源时刻；上方主相机、下方腕相机；各视角左侧原图、右侧绿色为保留进融合夹持区域的来源像素。可检查夹住前、刚夹稳和运动后期。</p><div id="sources"></div>
<h2>静态前后对照</h2><img id="static"><p>NPZ保留源XYZ、相机/像素/帧、配准变换、逐点帧支持数；PLY输出融合表面、夹持区域和单独待确认候选。全部仅留远程。</p>
<script>
const cases=__DATA__,sel=document.querySelector('#case'),cvs=[document.querySelector('#before'),document.querySelector('#after')];let idx=3,yaw=.65,pitch=-.3,zoom=2.6,drag=null;
cases.forEach((d,i)=>{let o=document.createElement('option');o.value=i;o.textContent=d.name;sel.append(o)});sel.value=idx;
function proj(p,cv){let x=Math.cos(yaw)*p[0]+Math.sin(yaw)*p[1],y=-Math.sin(yaw)*p[0]+Math.cos(yaw)*p[1];return [cv.width/2+x*zoom,cv.height/2+(Math.cos(pitch)*p[2]-Math.sin(pitch)*y)*zoom]}
function draw(){let d=cases[idx];cvs.forEach((cv,k)=>{cv.width=cv.clientWidth;cv.height=510;let ctx=cv.getContext('2d');ctx.fillStyle='#101c28';ctx.fillRect(0,0,cv.width,cv.height);
function dots(a,col,r){ctx.fillStyle=col;a.forEach(p=>{let v=proj(p,cv);ctx.beginPath();ctx.arc(...v,r,0,Math.PI*2);ctx.fill()})}
if(!document.querySelector('#only').checked)dots(k?d.fused:d.base,'#8296a6',1);
dots(k?d.region:d.baseRegion,'#2cf16b',2.3);if(k&&document.querySelector('#pre').checked)dots(d.pre,'#ffb34b',2);
ctx.strokeStyle='#dae9f5';ctx.setLineDash([4,5]);for(let i=0;i<8;i++)for(let j=i+1;j<8;j++)if([1,2,4].includes(i^j)){let a=proj(d.box[i],cv),b=proj(d.box[j],cv);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}ctx.setLineDash([]);
ctx.fillStyle='#dae9f5';ctx.font='14px sans-serif';ctx.fillText('20 mm',16,465);ctx.beginPath();ctx.moveTo(16,480);ctx.lineTo(16+zoom*20,480);ctx.stroke()})}
function update(){let d=cases[idx],s=d.summary;document.querySelector('#numbers').textContent=d.held?'参考帧 '+s.anchor_frame+'：原区域 '+s.anchor_region_points+' 个观测点；同一手坐标 1 mm 网格对照 '+s.anchor_region_voxels+' → '+s.fused_region_voxels+'。其中距原点云超过 2 mm 的新增网格 '+s.new_region_voxels_farther_than_2mm+'；夹住前待确认 '+s.uncertain_pre_region_voxels+'。接受 '+s.accepted_frames.length+' 个来源帧（其中夹住前 '+s.accepted_pre_frames.length+'）；这些不是抓取次数或补全准确率。':'本次空夹，融合后也不输出成功夹持区域。';
document.querySelector('#static').src=d.static;let e=document.querySelector('#sources');e.replaceChildren();d.gallery.forEach(url=>{let im=document.createElement('img');im.src=url;e.append(im)});draw()}
sel.onchange=()=>{idx=+sel.value;update()};document.querySelectorAll('input').forEach(e=>e.onchange=draw);document.querySelector('#reset').onclick=()=>{yaw=.65;pitch=-.3;zoom=2.6;draw()};
cvs.forEach(cv=>{cv.onpointerdown=e=>{drag=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId)};cv.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];draw()};cv.onpointerup=()=>drag=null;cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.5,Math.min(15,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false})});window.addEventListener('resize',draw);update();
</script></html>"""
 PAGE.write_text(html.replace('__DATA__',json.dumps(cases,ensure_ascii=False,separators=(',',':'))),encoding='utf-8')
 print('viewer complete',flush=True)
if __name__=='__main__':main()
