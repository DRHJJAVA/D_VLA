"""Remote interactive 3-D viewer of measured RGB-D contact estimates; no new training."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,numpy as np,h5py,hashlib
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from stage2_probe_motion_fingerpads import cloud,corners
from stage2_robot_only_depth import RobotDepth
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'audits/v3/motion_finger_contact_pilot_v1'
OUT=R/'outputs/v3_stage2/motion_contact_pointcloud_review_v1'
PAGE=R/'docs/第二步_夹指接触三维点云审核.html'
def ply(path,p,color,flag,side):
 with path.open('x') as f:
  f.write('ply\nformat ascii 1.0\ncomment measured world coordinates in meters; flags are contact ESTIMATES, not contact truth\nelement vertex '+str(len(p))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar contact_estimate\nproperty char finger_side\nend_header\n')
  for xyz,rgb,fl,si in zip(p,color,flag,side):f.write(' '.join([*(f'{v:.7f}' for v in xyz),*(str(int(v)) for v in rgb),str(int(fl)),str(int(si))])+'\n')
def main():
 OUT.mkdir(parents=True,exist_ok=False);assert not PAGE.exists()
 refs=json.loads((R/'audits/v3/finger_contact_repair_pilot_v1/protocol.json').read_text())['episodes'];by={x['id']:x for x in refs}
 frames=json.loads((D/'review_frames.json').read_text());datasets=[];summary=[]
 for item in frames:
  ident=item['id'];t=item['frame'];row=by[ident];name=ident+'_attempt'+str(item['attempt'])
  with np.load(D/(ident+'.npz')) as z:
   hand=z['hand_to_world'][t];q=z['measured_finger_q'][t];a,b=z['frame_offsets'][t:t+2]
   cp=z['points_world'][a:b];sides=z['point_side'][a:b];cams=z['point_camera'][a:b];pixels=z['point_pixel'][a:b]
  if not item['accepted_by_motion_visual_gate']:cp=cp[:0];sides=sides[:0];cams=cams[:0];pixels=pixels[:0]
  context=[];colors=[]
  with h5py.File(row['rgbd'],'r') as f:
   robot=RobotDepth(f['agentview/rgb'].shape[1])
   contactrgb=np.array([f[['agentview','robot0_eye_in_hand'][int(cam)]]['rgb'][t,int(v),int(u)] for cam,(u,v) in zip(cams,pixels)],np.uint8).reshape(-1,3)
   for cam in ['agentview','robot0_eye_in_hand']:
    g=f[cam];dep=g['depth_m'][t];K=g['K'][t];T=g['camera_to_world'][t]
    rd=robot.render(q,np.linalg.inv(hand)@T,K)
    p=cloud(dep,K,T);ph=(p-hand[:3,3])@hand[:3,:3]
    roi=(dep<rd-.002).reshape(-1)&(abs(ph[:,0])<.10)&(ph[:,1]>q[1]-.09)&(ph[:,1]<q[0]+.09)&(ph[:,2]>.065)&(ph[:,2]<.25)
    context.append(p[roi]);colors.append(g['rgb'][t].reshape(-1,3)[roi])
   robot.close()
  p=np.concatenate(context);rgb=np.concatenate(colors)
  if len(p):
   _,ix=np.unique(np.rint(p/.002).astype(int),axis=0,return_index=True);p=p[ix];rgb=rgb[ix]
  before=len(p);foreground='local RGB-D context; no accepted-contact seed'
  if len(cp) and len(p):
   tree=cKDTree(p);pairs=tree.query_pairs(.006,output_type='ndarray')
   if len(pairs):
    rr=np.r_[pairs[:,0],pairs[:,1]];cc=np.r_[pairs[:,1],pairs[:,0]]
    graph=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(p),len(p)))
    _,labels=connected_components(graph,directed=False)
   else:labels=np.arange(len(p))
   dist,near=tree.query(cp);wanted=np.unique(labels[near[dist<.005]])
   keep=np.isin(labels,wanted)
   if keep.any():p=p[keep];rgb=rgb[keep];foreground='automatic connected visible surface around accepted-contact seeds; not simulator object mask'
  # Keep every contact point at its original measured XYZ, even if context was voxel reduced.
  points=np.r_[p,cp];rawrgb=np.r_[rgb,contactrgb];flag=np.r_[np.zeros(len(p),np.uint8),np.ones(len(cp),np.uint8)]
  side=np.r_[np.full(len(p),-1,np.int8),sides.astype(np.int8)]
  np.savez_compressed(OUT/(name+'.npz'),xyz_world_m=points,rgb=rawrgb,contact_estimate=flag,finger_side=side,contact_source_camera=cams,contact_source_pixel=pixels,frame=t,hand_to_world=hand)
  displayrgb=rawrgb.copy();displayrgb[flag>0]=[30,245,70]
  ply(OUT/(name+'.ply'),points,displayrgb,flag,side)
  # Hand-relative coordinate display is a rigid transform, not a new reconstructed surface.
  local=(points-hand[:3,3])@hand[:3,:3];center=local.mean(0) if len(local) else np.array([0,0,.1])
  vis=(local-center)*1000
  contacts=[]
  for j in range(len(cp)):
   contacts.append({'world_m':cp[j].round(7).tolist(),'side':int(sides[j])+1,'camera':['agentview','robot0_eye_in_hand'][int(cams[j])],'pixel':pixels[j].tolist(),'frame':t})
  pads=[]
  for sign,k in [(1,0),(-1,1)]:
   plane=q[k]-sign*.0005
   pts=np.array([[-.008,plane,.0854],[.008,plane,.0854],[.008,plane,.1014],[-.008,plane,.1014]])
   pads.append(((pts-center)*1000).round(3).tolist())
  datasets.append({'name':name,'frame':t,'point':vis.round(3).tolist(),'rgb':rawrgb.tolist(),'flags':flag.tolist(),'side':side.tolist(),'contact_start':len(p),'contacts':contacts,'pads':pads,'held':item['accepted_by_motion_visual_gate'],'both':item['both_sides_visible'],'counts':item['visible_points_per_side']})
  summary.append({'id':ident,'attempt':item['attempt'],'frame':t,'context_points_before_component':before,'context_points':len(p),'contact_points':len(cp),'sides':item['visible_points_per_side'],'foreground':foreground,'npz':str(OUT/(name+'.npz')),'ply':str(OUT/(name+'.ply'))})
  print(name,len(p),len(cp),flush=True)
 data=json.dumps(datasets,ensure_ascii=False,separators=(',',':'))
 template="""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>夹指接触三维点云审核</title><style>
body{font:17px/1.6 system-ui,sans-serif;max-width:1160px;margin:28px auto;padding:0 20px;background:#f4f6f8;color:#18252e}h1{font-size:29px}aside{padding:16px;background:#fff0c5;border-radius:8px}canvas{width:100%;height:620px;touch-action:none;background:#101c28;border-radius:10px;cursor:grab}select,button{font:inherit;padding:7px 12px}label{margin-right:16px}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:16px 0}#info{white-space:pre-wrap;background:white;padding:15px;border:1px solid #d8e0e8}small{color:#526473}
</style><h1>夹指接触位置：真实深度点云中的估计</h1><aside><b>实测三维点，不只是图片上的绿色标记。</b>绿色为夹指接触位置估计，尚非已确认物理接触真值；没有为遮挡侧补造表面。尚未训练，仍等你核对。</aside>
<p>拖动旋转，滚轮缩放；点击绿色点查看其世界坐标、左右夹指编号、来源相机与像素。灰色或原色是邻近的可见表面；不透明完整曲面尚未重建。</p>
<div class="controls"><select id="case"></select><button id="reset">重置视角</button><label><input id="rgb" type="checkbox">原始颜色</label><label><input id="pads" type="checkbox">夹指内侧垫面参考框（不是接触面）</label><label><input id="only" type="checkbox">只看绿色点</label></div>
<p id="status"></p><canvas id="view"></canvas><p id="info">点击绿色点查看来源。</p><small>显示坐标为刚体变换后的局部毫米坐标；点击得到原始世界坐标（米）。上下文点云按2毫米体素显示，绿色点保留原始坐标。失败尝试不输出成功接触点。</small>
<p>远程逐例保存NPZ与PLY（含XYZ、RGB、contact_estimate、finger_side）；数据始终留在服务器。contact_estimate不是接触真值标签。机器人基座与自身深度已排除，局部物体前景由三维连通关系提取，仍可能不完整。</p>
<script>const cases=__DATA__;
const cv=document.querySelector('#view'),ctx=cv.getContext('2d'),sel=document.querySelector('#case'),inf=document.querySelector('#info'),st=document.querySelector('#status');
let yaw=.65,pitch=-.3,zoom=3,drag=null,moved=false,visible=[],idx=0;
cases.forEach((d,i)=>{let o=document.createElement('option');o.value=i;o.textContent=d.name+' · 帧'+d.frame;sel.append(o)});
function proj(p){let x=Math.cos(yaw)*p[0]+Math.sin(yaw)*p[1],y=-Math.sin(yaw)*p[0]+Math.cos(yaw)*p[1],z=p[2];let v=Math.cos(pitch)*z-Math.sin(pitch)*y,dep=Math.sin(pitch)*z+Math.cos(pitch)*y;return [cv.width/2+x*zoom,cv.height/2+v*zoom,dep]}
function draw(){let d=cases[idx];cv.width=Math.round(cv.clientWidth);cv.height=620;ctx.fillStyle='#101c28';ctx.fillRect(0,0,cv.width,cv.height);let order=d.point.map((p,i)=>({v:proj(p),i})).sort((a,b)=>a.v[2]-b.v[2]);visible=[];let rgb=document.querySelector('#rgb').checked,only=document.querySelector('#only').checked;
for(let x of order){let fl=d.flags[x.i];if(!fl){if(only)continue;ctx.fillStyle=rgb?'rgb('+d.rgb[x.i].join(',')+')':'#93a3b1';ctx.beginPath();ctx.arc(x.v[0],x.v[1],1.35,0,Math.PI*2);ctx.fill()}}
for(let x of order){if(!d.flags[x.i])continue;ctx.fillStyle='#31f34d';ctx.strokeStyle='#08220c';ctx.lineWidth=1;ctx.beginPath();ctx.arc(x.v[0],x.v[1],4.4,0,Math.PI*2);ctx.fill();ctx.stroke();visible.push({x:x.v[0],y:x.v[1],index:x.i-d.contact_start})}
if(document.querySelector('#pads').checked){ctx.strokeStyle='#ffbf52';ctx.lineWidth=1.3;ctx.setLineDash([5,4]);d.pads.forEach(p=>{ctx.beginPath();p.forEach((v,j)=>{let q=proj(v);if(j)ctx.lineTo(q[0],q[1]);else ctx.moveTo(q[0],q[1])});ctx.closePath();ctx.stroke()});ctx.setLineDash([])}
ctx.fillStyle='#e7edf4';ctx.font='15px sans-serif';ctx.fillText('绿色：接触估计点（画大便于查看，非真实接触面积）',18,26);let mm=20;ctx.strokeStyle='#dce7f0';ctx.beginPath();ctx.moveTo(25,cv.height-30);ctx.lineTo(25+mm*zoom,cv.height-30);ctx.stroke();ctx.fillText('20 mm',25,cv.height-43);
st.textContent=d.name+'：两侧估计点 '+d.counts.join(' / ')+'。'+(!d.held?'本次空夹，无成功接触点。':d.both?'两侧均有可见深度支持，待审核。':'仅一侧有可见支持，另一侧未知，不能认为两侧已恢复。')}
sel.onchange=()=>{idx=Number(sel.value);inf.textContent='点击绿色点查看来源。';draw()};document.querySelectorAll('input').forEach(x=>x.onchange=draw);document.querySelector('#reset').onclick=()=>{yaw=.65;pitch=-.3;zoom=3;draw()};
cv.onpointerdown=e=>{drag=[e.clientX,e.clientY];moved=false;cv.setPointerCapture(e.pointerId)};
cv.onpointermove=e=>{if(!drag)return;let dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(Math.abs(dx)+Math.abs(dy)>1)moved=true;yaw+=dx*.008;pitch+=dy*.008;drag=[e.clientX,e.clientY];draw()};
cv.onpointerup=e=>{if(!moved){let r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let near=visible.map(p=>({...p,dist:Math.hypot(p.x-x,p.y-y)})).sort((a,b)=>a.dist-b.dist)[0];if(near&&near.dist<15){let c=cases[idx].contacts[near.index];inf.textContent='夹指 '+c.side+' · 帧 '+c.frame+'\n世界坐标（米） '+c.world_m.join(', ')+'\n来源 '+c.camera+' · 像素 '+c.pixel.join(', ')+'\n接触位置估计，非物理接触真值。'}}drag=null};
cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.3,Math.min(15,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});window.addEventListener('resize',draw);draw();</script></html>"""
 PAGE.write_text(template.replace('__DATA__',data),encoding='utf-8')
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'locked.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(OUT.iterdir())+[PAGE,R/'src/v3/stage2_build_contact_pointcloud_review.py'] if p.is_file()}},indent=2))
 print('complete',flush=True)
if __name__=='__main__':main()
