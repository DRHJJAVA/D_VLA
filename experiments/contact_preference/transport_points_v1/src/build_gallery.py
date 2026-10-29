from pathlib import Path
import json, hashlib, datetime, html
import numpy as np
import h5py
import cv2

ROOT=Path('/mnt/nvme1/drh/D_VLA')
EXP=ROOT/'experiments/contact_preference/transport_points_v1'
OUT=ROOT/'docs/接触偏好_夹持后运输多帧点云.html'
cfg=json.loads((EXP/'config.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as q:
        for b in iter(lambda:q.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
inputs={}
for case in cfg['cases']:
    key=case['key']
    paths=[ROOT/'experiments/region_pose/iter_024/train'/(key+'.npz'),ROOT/'data/v3/trainval_rgbd_448'/case['asset']/(case['demo']+'.h5')]
    for p in paths:
        assert p.exists()
        inputs[str(p.relative_to(ROOT))]=sha(p)
(EXP/'input_locked.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2))
all_cases=[]
global_max_xyz_error=0.0
global_rgb_mismatch=0
global_bad_index=0
for case in cfg['cases']:
    key=case['key']; npz=ROOT/'experiments/region_pose/iter_024/train'/(key+'.npz')
    h5=ROOT/'data/v3/trainval_rgbd_448'/case['asset']/(case['demo']+'.h5')
    case_out={'key':key,'title':case['title'],'frames':[]}
    with np.load(npz,allow_pickle=False) as z,h5py.File(h5) as h:
        frame_indices=h['frame_indices'][:].tolist()
        assert case['frames']==z['selected_frames'].tolist()
        p0=np.array(z['hand_to_world_'+str(case['frames'][0])][:3,3],dtype=float)
        for t in case['frames']:
            assert frame_indices.count(t)==1
            j=frame_indices.index(t)
            H=np.array(z['hand_to_world_'+str(t)],dtype=float)
            grip=H@np.array([0.,0.,.093,1.])
            sel=np.where(z['frame']==t)[0]
            cloud=[]
            panels=[]
            cam_names=['agentview','robot0_eye_in_hand']
            count_by_cam={}
            for cam_id,cam in enumerate(cam_names):
                rgb=np.array(h[cam+'/rgb'][j])
                dep=np.array(h[cam+'/depth_m'][j])
                K=np.array(h[cam+'/K'][j],dtype=float)
                T=np.array(h[cam+'/camera_to_world'][j],dtype=float)
                assert rgb.shape==(448,448,3) and dep.shape==(448,448)
                ix=sel[z['camera'][sel]==cam_id]
                count_by_cam[cam]=int(len(ix))
                im=rgb[:,:,::-1].copy()
                for i in ix:
                    u,v=map(int,z['pixel'][i])
                    if not (0<=u<448 and 0<=v<448):
                        global_bad_index+=1;continue
                    dep_i=float(dep[v,u])
                    xyz_cam=np.array([(u-K[0,2])*dep_i/K[0,0],(v-K[1,2])*dep_i/K[1,1],dep_i,1.])
                    xyz_world=(T@xyz_cam)[:3]
                    global_max_xyz_error=max(global_max_xyz_error,float(np.linalg.norm(xyz_world-z['source_xyz'][i])))
                    if not np.array_equal(rgb[v,u],z['rgb'][i]): global_rgb_mismatch+=1
                    col=(40,215,90) if bool(z['supported_candidate'][i]) else (185,60,240)
                    cv2.circle(im,(u,v),5,(0,0,0),-1,cv2.LINE_AA)
                    cv2.circle(im,(u,v),4,col,-1,cv2.LINE_AA)
                yy,xx=np.mgrid[0:448:5,0:448:5]
                dd=dep[::5,::5]
                valid=np.isfinite(dd)&(dd>.05)&(dd<3)
                x=(xx-K[0,2])*dd/K[0,0]
                y=(yy-K[1,2])*dd/K[1,1]
                pc=np.stack([x,y,dd,np.ones_like(dd)],axis=-1)[valid]
                world=(pc@T.T)[:,:3]
                hand=(np.c_[world,np.ones(len(world))]@np.linalg.inv(H).T)[:,:3]
                radius=np.linalg.norm(world-grip[:3],axis=1)
                keep=radius<.11
                hand=hand[keep]
                colors=rgb[::5,::5][valid][keep]
                if len(hand)>1800:
                    ids=np.linspace(0,len(hand)-1,1800,dtype=int)
                    hand=hand[ids];colors=colors[ids]
                cloud.extend([[round(float(v[0]),4),round(float(v[1]),4),round(float(v[2]),4),int(c[0]),int(c[1]),int(c[2]),0] for v,c in zip(hand,colors)])
                finite=dep[np.isfinite(dep)&(dep>.05)&(dep<3)]
                low=float(np.percentile(finite,3));high=float(np.percentile(finite,97))
                dn=np.clip((dep-low)/(max(high-low,1e-5)),0,1)
                depthim=cv2.applyColorMap((255*(1-dn)).astype(np.uint8),cv2.COLORMAP_TURBO)
                depthim[~np.isfinite(dep)|(dep<=0)]=(20,20,20)
                for i in ix:
                    u,v=map(int,z['pixel'][i])
                    if not (0<=u<448 and 0<=v<448):continue
                    col=(40,215,90) if bool(z['supported_candidate'][i]) else (185,60,240)
                    cv2.circle(depthim,(u,v),5,(0,0,0),-1,cv2.LINE_AA)
                    cv2.circle(depthim,(u,v),4,col,-1,cv2.LINE_AA)
                panels.append((im,depthim))
            for i in sel:
                v=z['source_hand_xyz'][i]
                col=[36,215,90] if bool(z['supported_candidate'][i]) else [238,52,188]
                cloud.append([round(float(v[0]),4),round(float(v[1]),4),round(float(v[2]),4),*col,1])
            canvas=np.full((944,944,3),(18,24,32),dtype=np.uint8)
            for cam_id,(im,depthim) in enumerate(panels):
                x=16+cam_id*464
                canvas[34:482,x:x+448]=im
                canvas[490:938,x:x+448]=depthim
                cv2.putText(canvas,cam_names[cam_id]+' RGB', (x,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,235,235),1,cv2.LINE_AA)
            cv2.putText(canvas,'RGB-D frame '+str(t)+' | green=multiframe candidate, magenta=other source',(16,486),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1,cv2.LINE_AA)
            fn=key+'_frame'+str(t)+'.jpg'
            assert cv2.imwrite(str(EXP/'images'/fn),canvas,[cv2.IMWRITE_JPEG_QUALITY,91])
            case_out['frames'].append({'frame':t,'image':'图示/接触偏好_夹持后运输/'+fn,'observed':int(len(sel)),'supported':int(np.count_nonzero(z['supported_candidate'][sel])),'by_camera':count_by_cam,'hand_displacement_mm':round(float(np.linalg.norm(H[:3,3]-p0)*1000),2),'cloud':cloud})
    all_cases.append(case_out)
assert global_rgb_mismatch==0 and global_bad_index==0 and global_max_xyz_error<1e-5
assert {k:sha(ROOT/k) for k in inputs}==inputs
(EXP/'audit.json').write_text(json.dumps({'cases':[{k:v for k,v in c.items() if k!='frames'}|{'frames':[{k:v for k,v in fr.items() if k!='cloud'} for fr in c['frames']]} for c in all_cases],'rgb_mismatches':global_rgb_mismatch,'bad_pixel_indices':global_bad_index,'max_reproject_xyz_error_m':global_max_xyz_error,'input_hashes_unchanged':True},ensure_ascii=False,indent=2))
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
data=json.dumps(all_cases,ensure_ascii=False,separators=(',',':'))
page=r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>夹持后运输：真实多帧 RGB-D 点云</title><style>
:root{color-scheme:dark;--bg:#0c1421;--panel:#142338;--line:#30445c;--muted:#a9bacd}*{box-sizing:border-box}body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:var(--bg);color:#f4f8ff;margin:0;padding:26px;line-height:1.55}main{max-width:1300px;margin:auto}h1{font-size:28px;margin:0 0 8px}.lead{color:var(--muted);max-width:1050px}.warn{background:#302319;border-left:4px solid #f5b34d;padding:12px 16px;border-radius:8px;margin:18px 0}.tabs,.frames{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}button{font:inherit;border:1px solid var(--line);background:#172a42;color:#e9f1ff;border-radius:8px;padding:8px 13px;cursor:pointer}button[aria-pressed="true"]{background:#146788;border-color:#51b8dc}section{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin:18px 0}.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(320px,0.78fr);gap:18px;align-items:start}.panelimg{width:100%;border-radius:8px;border:1px solid var(--line)}canvas{width:100%;height:520px;background:#08101a;border:1px solid var(--line);border-radius:8px;cursor:grab}.legend{color:var(--muted);font-size:14px}.stat{padding:9px 12px;background:#0d1b2d;border-radius:8px;margin:9px 0}.green{color:#52df86}.pink{color:#f361cf}@media(max-width:850px){.grid{grid-template-columns:1fr}canvas{height:390px}}</style></head><body><main>
<h1>夹住后运输：多帧观测点与点云</h1><p class="lead">三段已有示范，每段切换多个夹住后的帧。左侧为同帧主视角和腕相机 RGB-D，右侧为该帧夹爪坐标系下可旋转的实测点云。帧号对应原示范回放帧，手位移相对本案例首个展示帧。</p>
<div class="warn"><strong>读图边界：</strong><span class="green">绿色</span>是此前跨帧支持的有限夹指体积内<strong>真实观测来源候选</strong>；<span class="pink">玫红色</span>是该帧其余有限体积真实观测。两者都未证实逐点属于目标物体、未证实物理接触；灰色邻域点云更可能含机械臂、背景或邻物。被夹指遮住而没有实测深度的表面保留未知。这里不显示模型评分，也没有新增训练标签。</div>
<div id="caseTabs" class="tabs"></div><section><h2 id="caseTitle"></h2><div id="frameTabs" class="frames"></div><div id="stats" class="stat"></div><div class="grid"><div><img id="panel" class="panelimg" alt="两路相机 RGB-D 与观测点"><p class="legend">图像上排 RGB、下排深度，左列主视角、右列腕相机。彩色点均按保存的真实像素和深度位置绘制。</p></div><div><canvas id="cloud"></canvas><p class="legend">拖动旋转；滚轮缩放。3D 点以夹爪坐标显示，便于比较不同运输帧。灰色背景是距夹指中心 11 cm 内的抽样 RGB-D 点，并非物体分割。</p></div></div></section>
<p class="lead">生成时间 __TIME__。输入来自已保存双相机 RGB-D 与第024轮锁定的来源观测；没有重训模型或读取预留测试。</p></main><script>const cases=__DATA__;
let ci=0,fi=0,yaw=.35,pitch=-.25,zoom=1,drag=false,lastX=0,lastY=0;const C=document.getElementById('cloud'),ctx=C.getContext('2d');
function btn(label,on,active){let b=document.createElement('button');b.textContent=label;b.setAttribute('aria-pressed',active?'true':'false');b.onclick=on;return b}
function render(){let c=cases[ci],f=c.frames[fi];document.getElementById('caseTitle').textContent=c.title;let tabs=document.getElementById('caseTabs');tabs.replaceChildren();cases.forEach((x,i)=>tabs.appendChild(btn(x.title,()=>{ci=i;fi=0;render()},i===ci)));let ft=document.getElementById('frameTabs');ft.replaceChildren();c.frames.forEach((x,i)=>ft.appendChild(btn('帧 '+x.frame,()=>{fi=i;render()},i===fi)));document.getElementById('panel').src=f.image;document.getElementById('stats').textContent='帧 '+f.frame+' · 相对首帧夹爪位移 '+f.hand_displacement_mm+' mm · 真实有限体积来源 '+f.observed+' 点（跨帧支持候选 '+f.supported+'；主相机 '+f.by_camera.agentview+'，腕相机 '+f.by_camera.robot0_eye_in_hand+'）';draw()}
function draw(){let w=C.clientWidth,h=C.clientHeight,dpr=window.devicePixelRatio||1;C.width=Math.round(w*dpr);C.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#08101a';ctx.fillRect(0,0,w,h);let pts=cases[ci].frames[fi].cloud;let out=[];for(let p of pts){let x=p[0],y=p[1],z=p[2]-.093;let xx=x*Math.cos(yaw)-y*Math.sin(yaw), yy=x*Math.sin(yaw)+y*Math.cos(yaw);let zz=z*Math.cos(pitch)-yy*Math.sin(pitch), ry=z*Math.sin(pitch)+yy*Math.cos(pitch);let sc=1800*zoom;let sx=w/2+xx*sc,sy=h/2-zz*sc;if(sx<-10||sx>w+10||sy<-10||sy>h+10)continue;out.push([ry,sx,sy,p])}out.sort((a,b)=>a[0]-b[0]);for(let a of out){let p=a[3];ctx.beginPath();ctx.arc(a[1],a[2],p[6]?4.2:1.35,0,Math.PI*2);ctx.fillStyle=p[6]?'rgb('+p[3]+','+p[4]+','+p[5]+')':'rgba('+p[3]+','+p[4]+','+p[5]+',.37)';ctx.fill()}ctx.strokeStyle='#f5bc4e';ctx.lineWidth=1.6;ctx.beginPath();ctx.moveTo(w/2-8,h/2);ctx.lineTo(w/2+8,h/2);ctx.moveTo(w/2,h/2-8);ctx.lineTo(w/2,h/2+8);ctx.stroke()}
C.addEventListener('pointerdown',e=>{drag=true;lastX=e.clientX;lastY=e.clientY;C.setPointerCapture(e.pointerId)});C.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-lastX)*.01;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-lastY)*.01));lastX=e.clientX;lastY=e.clientY;draw()});C.addEventListener('pointerup',()=>drag=false);C.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});window.addEventListener('resize',draw);render();
</script></body></html>"""
page=page.replace('__DATA__',data).replace('__TIME__',now)
OUT.write_text(page,encoding='utf-8')
doc=ROOT/'docs/夹爪物体接触偏好学习_完整方案.md'
with doc.open('a',encoding='utf-8') as q:q.write('\n'+now+'：搬运多帧展示已生成，3 个训练/开发事件共 12 帧；主/腕 RGB 源像素 0 不一致，真实深度反投影最大误差 '+format(global_max_xyz_error,'.3g')+' m，输入前后哈希一致。绿色与玫红均非接触真值；页面 '+str(OUT.relative_to(ROOT))+'。未训练，阶段2接触面定位仍未验收。\n')
products=[OUT,EXP/'audit.json',EXP/'config.json',EXP/'input_locked.json',EXP/'src/build_gallery.py']+sorted((EXP/'images').glob('*.jpg'))
(EXP/'completed_locked.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in products},ensure_ascii=False,indent=2))
(EXP/'stage_state.json').write_text(json.dumps({'stage':'complete','cases':3,'frames':12,'contact_surface_accepted':False,'model_trained':False,'generated_beijing':now},ensure_ascii=False,indent=2))
print(json.dumps({'page':str(OUT),'cases':len(all_cases),'frames':sum(len(x['frames']) for x in all_cases),'max_xyz_error_m':global_max_xyz_error,'rgb_mismatches':global_rgb_mismatch,'files_locked':len(products),'time':now},ensure_ascii=False))
