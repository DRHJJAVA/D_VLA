"""Visible object-side RGB-D samples inside the finite inter-finger grasp volume.
No contact-truth labels, no hidden surface completion, no new training."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,hashlib,numpy as np,h5py,cv2
from stage2_probe_motion_fingerpads import cloud
from stage2_robot_only_depth import RobotDepth
R=Path('/mnt/nvme1/drh/D_VLA')
D=R/'audits/v3/motion_finger_contact_pilot_v1'
OLD=R/'outputs/v3_stage2/motion_contact_pointcloud_review_v1'
OUT=R/'outputs/v3_stage2/interfinger_region_review_v1'
PAGE=R/'docs/第二步_夹爪之间物体点云.html'
IMG=R/'docs/图示/夹爪之间物体点云'
def writeply(path,p,rgb):
 with path.open('x') as f:
  f.write('ply\nformat ascii 1.0\ncomment Visible RGBD samples inside inter-finger volume, NOT contact truth\nelement vertex '+str(len(p))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
  for xyz,c in zip(p,rgb):f.write(' '.join([*(f'{x:.8f}' for x in xyz),*(str(int(x)) for x in c)])+'\n')
def main():
 OUT.mkdir(exist_ok=False,parents=True);IMG.mkdir(exist_ok=False,parents=True);assert not PAGE.exists()
 rows={x['id']:x for x in json.loads((R/'audits/v3/finger_contact_repair_pilot_v1/protocol.json').read_text())['episodes']}
 items=json.loads((D/'review_frames.json').read_text());cases=[];summary=[]
 for it in items:
  ident,t=it['id'],it['frame'];name=ident+'_attempt'+str(it['attempt']);held=it['accepted_by_motion_visual_gate']
  with np.load(D/(ident+'.npz')) as z: H=z['hand_to_world'][t];q=z['measured_finger_q'][t]
  lo=np.array([-.008,q[1]+.0005,.0854]);hi=np.array([.008,q[0]-.0005,.1014])
  parts=[];rgbparts=[];camparts=[];uvparts=[];panels=[]
  with h5py.File(rows[ident]['rgbd'],'r') as f:
   rob=RobotDepth(f['agentview/rgb'].shape[1])
   for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
    g=f[cam];depth=g['depth_m'][t];K=g['K'][t];T=g['camera_to_world'][t];rgb=g['rgb'][t]
    p=cloud(depth,K,T);local=(p-H[:3,3])@H[:3,:3];rd=rob.render(q,np.linalg.inv(H)@T,K)
    mask=np.isfinite(p).all(1)&(depth.reshape(-1)>0)&(depth<rd-.002).reshape(-1)&np.all(local>=lo,axis=1)&np.all(local<=hi,axis=1)
    if not held:mask[:]=False
    v,u=np.indices(depth.shape);uv=np.stack([u,v],-1).reshape(-1,2)
    parts.append(p[mask]);rgbparts.append(rgb.reshape(-1,3)[mask]);camparts.append(np.full(mask.sum(),ci,np.uint8));uvparts.append(uv[mask])
    overlay=rgb.copy();overlay.reshape(-1,3)[mask]=[30,245,70]
    # Focus both images on known finger volume, with surrounding object context.
    center=(lo+hi)/2;cw=center@H[:3,:3].T+H[:3,3];cc=(cw-T[:3,3])@T[:3,:3]
    if cc[2]>0:
     cx,cy=int(K[0,0]*cc[0]/cc[2]+K[0,2]),int(K[1,1]*cc[1]/cc[2]+K[1,2])
    else:cx,cy=depth.shape[1]//2,depth.shape[0]//2
    half=90;x0=max(0,min(depth.shape[1]-2*half,cx-half));y0=max(0,min(depth.shape[0]-2*half,cy-half))
    images=[cv2.resize(a[y0:y0+2*half,x0:x0+2*half],(420,420),interpolation=cv2.INTER_NEAREST) for a in [rgb,overlay]]
    panel=np.concatenate(images,axis=1);cv2.rectangle(panel,(0,0),(840,28),(15,25,35),-1)
    cv2.putText(panel,f'{cam} frame {t} | original / inter-finger visible points {int(mask.sum())}',(8,20),0,.48,(255,255,255),1)
    panels.append(panel)
   rob.close()
  p=np.concatenate(parts);rgb=np.concatenate(rgbparts);cams=np.concatenate(camparts);uv=np.concatenate(uvparts)
  local=(p-H[:3,3])@H[:3,:3];assert np.all(local>=lo-1e-8) and np.all(local<=hi+1e-8)
  np.savez_compressed(OUT/(name+'.npz'),xyz_world_m=p,xyz_hand_m=local,rgb=rgb,source_camera=cams,source_pixel=uv,frame=t,hand_to_world=H,volume_min_hand_m=lo,volume_max_hand_m=hi,held_candidate=held,region_not_contact_truth=True)
  writeply(OUT/(name+'.ply'),p,rgb)
  cv2.imwrite(str(IMG/(name+'.jpg')),cv2.cvtColor(np.concatenate(panels),cv2.COLOR_RGB2BGR))
  with np.load(OLD/(name+'.npz')) as z: context=z['xyz_world_m'][z['contact_estimate']==0];old=z['xyz_world_m'][z['contact_estimate']>0]
  loccontext=(context-H[:3,3])@H[:3,:3];center=loccontext.mean(0)
  corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
  unique=len(np.unique(np.rint(p/.001).astype(np.int64),axis=0)) if len(p) else 0
  s={'id':ident,'attempt':it['attempt'],'frame':t,'held_candidate':held,'old_contact_estimate_points':len(old),'visible_region_samples':len(p),'camera_counts':[int((cams==j).sum()) for j in range(2)],'occupied_1mm_voxels':unique,'volume_min_hand_m':lo.tolist(),'volume_max_hand_m':hi.tolist()}
  summary.append(s)
  cases.append(dict(name=name,frame=t,held=held,context=((loccontext-center)*1000).round(3).tolist(),region=((local-center)*1000).round(3).tolist(),old=(((old-H[:3,3])@H[:3,:3]-center)*1000).round(3).tolist(),box=((corners-center)*1000).round(3).tolist(),world=p.round(8).tolist(),cams=cams.tolist(),uv=uv.tolist(),counts=s['camera_counts'],voxels=unique,image='图示/夹爪之间物体点云/'+name+'.jpg'))
  print(name,'old',len(old),'region',len(p),'1mm_voxels',unique,flush=True)
 template=r"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>夹爪之间的物体点云</title>
<style>body{max-width:1160px;margin:26px auto;padding:0 20px;font:17px/1.7 system-ui;color:#192d3c;background:#f4f6f8}h1{font-size:28px}aside{background:#fff0c9;padding:16px;border-radius:8px}select,button{font:inherit;padding:8px}label{margin:8px}canvas{width:100%;height:570px;background:#101c28;border-radius:10px;touch-action:none}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:20px}#info{white-space:pre-wrap;background:white;padding:12px}img{max-width:840px;width:100%}small{color:#4b6070}</style>
<h1>夹爪之间的物体点云</h1><aside><b>绿色改为“夹持范围内的可见物体表面”，不再只选贴近夹指的窄条。</b><br>范围：两根夹指内侧之间，宽度和高度限于夹指的 16 × 16 毫米夹持段。去除机器人自身深度后，保留范围内全部观测点。不把绿色称为真实接触面，不补造遮挡面。</aside>
<div class="controls"><select id="case"></select><button id="reset">重置视角</button><label><input id="box" type="checkbox" checked>显示夹持范围</label><label><input id="only" type="checkbox">只看绿色区域</label><label><input id="old" type="checkbox">对比旧接触估计点（橙色）</label></div>
<p id="status"></p><canvas id="view"></canvas><p id="info">拖动旋转，滚轮缩放。点击绿色点查看原始坐标和来源。</p>
<small>灰色为局部可见表面，绿色为夹指之间的观测点，虚线盒为几何范围。未插值、未增加合成点；双视角可能重复观测同一表面，因此同时列出 1 毫米网格数。上下文前景仍可能不完整；区域属于物体的判断依赖持物状态与机器人深度排除，尚待你核对。</small>
<h2>来源图像核对</h2><p>左：原始 RGB；右：对应像素上的绿色区域。上：主相机；下：腕相机。这是示范分析，尚未用于新模型训练。</p><img id="source">
<p>全分辨率区域 XYZ、RGB、相机、像素、帧及夹持范围已保存为远程 NPZ/PLY。数据不下载到本地。失败尝试不输出成功夹持区域。</p>
<script>
const cases=__DATA__,cv=document.querySelector('#view'),ctx=cv.getContext('2d'),sel=document.querySelector('#case'),info=document.querySelector('#info');
let idx=3,yaw=.65,pitch=-.3,zoom=5,drag=null,moved=false,shown=[];
cases.forEach((d,i)=>{let o=document.createElement('option');o.value=i;o.textContent=d.name+' · 帧'+d.frame;sel.append(o)});sel.value=idx;
function proj(p){let x=Math.cos(yaw)*p[0]+Math.sin(yaw)*p[1],y=-Math.sin(yaw)*p[0]+Math.cos(yaw)*p[1],z=p[2];return [cv.width/2+x*zoom,cv.height/2+(Math.cos(pitch)*z-Math.sin(pitch)*y)*zoom]}
function dots(p,color,r,record){ctx.fillStyle=color;p.forEach((a,i)=>{let v=proj(a);ctx.beginPath();ctx.arc(v[0],v[1],r,0,Math.PI*2);ctx.fill();if(record)shown.push({x:v[0],y:v[1],i})})}
function draw(){let d=cases[idx];cv.width=cv.clientWidth;cv.height=570;ctx.fillStyle='#101c28';ctx.fillRect(0,0,cv.width,cv.height);shown=[];
if(!document.querySelector('#only').checked)dots(d.context,'#728899',1.1,false);dots(d.region,'#30ee65',2.2,true);
if(document.querySelector('#old').checked)dots(d.old,'#ffab42',3.1,false);
if(document.querySelector('#box').checked){ctx.strokeStyle='#e4edf5';ctx.setLineDash([5,5]);for(let i=0;i<8;i++)for(let j=i+1;j<8;j++)if([1,2,4].includes(i^j)){let a=proj(d.box[i]),b=proj(d.box[j]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}ctx.setLineDash([])}
ctx.fillStyle='#e2edf5';ctx.font='15px sans-serif';ctx.fillText('绿色：夹持范围内可见表面；并非每个点都与夹指接触',16,25);
ctx.strokeStyle='#e2edf5';ctx.beginPath();ctx.moveTo(20,540);ctx.lineTo(20+20*zoom,540);ctx.stroke();ctx.fillText('20 mm',20,525);
document.querySelector('#status').textContent=d.name+'：旧接触估计 '+d.old.length+' 点 → 范围内 '+d.region.length+' 个观测点（主/腕 '+d.counts.join(' / ')+'；1 mm 网格 '+d.voxels+'）。'+(d.held?'已判为持物候选，待人工核对。':'本次空夹，区域不输出。');
document.querySelector('#source').src=d.image}
sel.onchange=()=>{idx=+sel.value;info.textContent='点击绿色点查看原始坐标和来源。';draw()};document.querySelectorAll('input').forEach(x=>x.onchange=draw);document.querySelector('#reset').onclick=()=>{yaw=.65;pitch=-.3;zoom=5;draw()};
cv.onpointerdown=e=>{drag=[e.clientX,e.clientY];moved=false;cv.setPointerCapture(e.pointerId)};
cv.onpointermove=e=>{if(!drag)return;let dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(Math.abs(dx)+Math.abs(dy)>1)moved=true;yaw+=dx*.008;pitch+=dy*.008;drag=[e.clientX,e.clientY];draw()};
cv.onpointerup=e=>{if(!moved){let b=cv.getBoundingClientRect(),x=e.clientX-b.left,y=e.clientY-b.top,hit=shown.map(p=>({...p,dist:Math.hypot(p.x-x,p.y-y)})).sort((a,b)=>a.dist-b.dist)[0];if(hit&&hit.dist<12){let d=cases[idx],i=hit.i;info.textContent='世界坐标（米）：'+d.world[i].join(', ')+'\n来源：'+['主相机','腕相机'][d.cams[i]]+'，像素 '+d.uv[i].join(', ')+'，帧 '+d.frame+'\n夹持范围内表面点，不是接触真值。'}}drag=null};
cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.5,Math.min(20,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});window.addEventListener('resize',draw);draw();
</script></html>"""
 PAGE.write_text(template.replace('__DATA__',json.dumps(cases,ensure_ascii=False,separators=(',',':'))),encoding='utf-8')
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 (OUT/'protocol.json').write_text(json.dumps({'input':'same six reviewed frames; complete dual-camera depth at native resolution; aligned measured q; robot-only render','selection':'held gate AND nonrobot depth AND finite prism between inner finger pad planes with x +/-8mm and z85.4..101.4mm in hand coordinates','no_synthetic_points':True,'occluded_surface':'unknown','not_contact_truth':True,'no_training':True},indent=2))
 files=list(OUT.iterdir())+list(IMG.iterdir())+[PAGE,Path(__file__)]
 (OUT/'locked.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
 print('complete',flush=True)
if __name__=='__main__':main()
