from pathlib import Path
import json,h5py,numpy as np,datetime,hashlib,html
from PIL import Image,ImageDraw
from matplotlib import colormaps
root=Path('/mnt/nvme1/drh/D_VLA')
out=root/'experiments/contact_preference/fixed_camera_pilot_v1'
docs=root/'docs';fig=docs/'图示/接触偏好_固定相机试验';fig.mkdir(parents=True,exist_ok=True)
cfg=json.loads((out/'config.json').read_text())
audit=json.loads((out/'audit.json').read_text())
newaudit=json.loads((out/'new_source_candidate_audit.json').read_text())
old=root/'experiments/contact_frames/cf017'
cams=['agentview','robot0_eye_in_hand','frontview','sideview','birdview']
names={'agentview':'原固定主视角','robot0_eye_in_hand':'原腕相机','frontview':'新增正面','sideview':'新增侧面','birdview':'新增俯视'}
titles={'stove_demo_0':'炉灶开关','drawer_demo_0':'拉开抽屉','two_objects_demo_0':'双物体搬运'}
action={'stove_demo_0':86,'drawer_demo_0':70,'two_objects_demo_0':300}
clouds={}
images={}
def local_world(depth,K,C,step=4):
    h,w=depth.shape
    vv,uu=np.mgrid[0:h:step,0:w:step];u=uu.ravel();v=vv.ravel();z=depth[v,u].astype(np.float64)
    ok=np.isfinite(z)&(z>0)&(z<3)
    u=u[ok];v=v[ok];z=z[ok]
    xc=np.stack([(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z],axis=1)
    return xc@C[:3,:3].T+C[:3,3]
def project(X,K,C,depth):
    inv=np.linalg.inv(C)
    xc=X@inv[:3,:3].T+inv[:3,3]
    z=xc[:,2]
    u=np.rint(K[0,0]*xc[:,0]/z+K[0,2]).astype(int)
    v=np.rint(K[1,1]*xc[:,1]/z+K[1,2]).astype(int)
    good=(z>0)&(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
    j=np.flatnonzero(good)
    good[j]=np.abs(depth[v[j],u[j]]-z[j])<=0.005
    return u,v,good
for case,c in cfg['cases'].items():
    motion=np.load(old/'cases'/case/'motion.npz')
    ref=audit['cases'][case]
    patch_rec=json.loads((root/'experiments/contact_preference/stage012_v1/stage2_audit.json').read_text())['cross_task_localization'][case]
    H=motion['hand_to_world'][patch_rec['reference_frame']]
    q=motion['measured_finger_q'][patch_rec['reference_frame']]
    centers=np.stack([H[:3,:3]@np.array([0,q[0],0.093])+H[:3,3],H[:3,:3]@np.array([0,q[1],0.093])+H[:3,3],H[:3,:3]@np.array([0,0,0.097])+H[:3,3]])
    with np.load(old/'cases'/case/'artifacts/reference.npz') as z: X=z['surface_xyz']
    d=np.min(np.linalg.norm(X[:,None,:]-centers[None,:,:],axis=2),axis=1)
    patch=X[d<patch_rec['hand_radius_m_diagnostic']]
    with np.load(out/(case+'_new_source_candidates.npz')) as nz:newpoints=nz['new_surface_points']
    assert len(patch)==ref['source_patch_points']
    clouds[case]={};images[case]={}
    with h5py.File(old/'rgbd'/(case+'.h5'),'r') as orig,h5py.File(out/(case+'.h5'),'r') as new:
        fi=new['frame_indices'][:].tolist()
        for mode,t in [('source',0),('action',action[case])]:
            slot=fi.index(t)
            Ht=motion['hand_to_world'][t]
            hand=Ht[:3,:3]@np.array([0,0,0.097])+Ht[:3,3]
            focus=np.mean(patch,axis=0) if mode=='source' else hand
            tiles=[];pts=[]
            for ci,cam in enumerate(cams):
                h=orig if ci<2 else new
                ix=t if ci<2 else slot
                rgb=np.asarray(h[cam]['rgb'][ix]);depth=np.asarray(h[cam]['depth_m'][ix])
                K=np.asarray(h[cam]['K'][ix],float);C=np.asarray(h[cam]['camera_to_world'][ix],float)
                im=Image.fromarray(rgb).resize((260,260))
                overlay=Image.new('RGBA',im.size,(0,0,0,0));draw=ImageDraw.Draw(overlay)
                if mode=='source':
                    u,v,vis=project(patch,K,C,depth)
                    for j in np.flatnonzero(vis)[::max(1,len(patch)//130)]:
                        x=u[j]*260/rgb.shape[1];y=v[j]*260/rgb.shape[0]
                        draw.ellipse((x-2,y-2,x+2,y+2),fill=(30,220,90,220))
                    nu,nv,nvis=project(newpoints,K,C,depth)
                    for j in np.flatnonzero(nvis)[::max(1,len(newpoints)//160)]:
                        x=nu[j]*260/rgb.shape[1];y=nv[j]*260/rgb.shape[0]
                        draw.ellipse((x-2,y-2,x+2,y+2),fill=(255,55,125,220))
                else:
                    inv=np.linalg.inv(C);xc=hand@inv[:3,:3].T+inv[:3,3]
                    if xc[2]>0:
                        u=K[0,0]*xc[0]/xc[2]+K[0,2];v=K[1,1]*xc[1]/xc[2]+K[1,2]
                        x=u*260/rgb.shape[1];y=v*260/rgb.shape[0]
                        if 0<=x<260 and 0<=y<260:
                            draw.line((x-7,y,x+7,y),fill=(255,215,0,255),width=3)
                            draw.line((x,y-7,x,y+7),fill=(255,215,0,255),width=3)
                im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
                z=depth[np.isfinite(depth)&(depth>0)]
                lo,hi=np.percentile(z,[2,98])
                val=np.clip((depth-lo)/max(hi-lo,1e-6),0,1)
                color=(colormaps['turbo'](1-val)[...,:3]*255).astype(np.uint8)
                dep=Image.fromarray(color).resize((260,260))
                tile=Image.new('RGB',(268,560),'#0f172a');tile.paste(im,(4,24));tile.paste(dep,(4,292))
                td=ImageDraw.Draw(tile);td.text((8,4),cam,fill='white');td.text((8,552),f'{lo:.2f}-{hi:.2f}m',fill='white')
                tiles.append(tile)
                world=local_world(depth,K,C,step=4)
                local=world-focus
                take=np.linalg.norm(local,axis=1)<0.115
                world=world[take]
                if len(world)>1200:world=world[np.linspace(0,len(world)-1,1200,dtype=int)]
                pts.extend([[round(float(v),1) for v in (x-focus)*1000]+[ci,0] for x in world])
            if mode=='source':
                pts.extend([[round(float(v),1) for v in (x-focus)*1000]+[5,1] for x in patch[::max(1,len(patch)//300)]])
                pts.extend([[round(float(v),1) for v in (x-focus)*1000]+[7,3] for x in newpoints[::max(1,len(newpoints)//400)]])
            else:
                pts.append([0,0,0,6,2])
            composite=Image.new('RGB',(268*5,560),'#0f172a')
            for ci,tile in enumerate(tiles):composite.paste(tile,(268*ci,0))
            file=fig/(case+'_'+mode+'.jpg');composite.save(file,quality=87,optimize=True)
            images[case][mode]='图示/接触偏好_固定相机试验/'+file.name
            clouds[case][mode]=pts
# HTML stores only measured point clouds and stage2 historical source candidates.
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
payload=json.dumps(clouds,separators=(',',':'))
blocks=[]
for case in cfg['cases']:
    r=audit['cases'][case];u=r['near_hand_8cm']['union']
    vis=r['fixed_views']
    novel=newaudit['cases'][case]['new_gt3mm_unique_1mm_representatives']
    rows=''.join(f'<tr><td>{html.escape(names[c])}</td><td>{vis[c]["source_patch_depth_consistent_5mm"]}/{r["source_patch_points"]}</td><td>{r["near_hand_8cm"][c]["voxel3mm"]}</td></tr>' for c in cams)
    blocks.append(f'''<section id="{case}"><h2>{titles[case]}</h2><p>原示范 demo0：接触前来源帧 0；操作查看帧 {action[case]}。来源点为既有阶段2手部邻域候选 {r["source_patch_points"]} 个真实深度点；新增视角按同一手部邻域提取、距原点集超过3 mm且以1 mm格去重的未筛物体身份的新增邻域点 {novel} 个。两者均不是物理接触标注。</p><table><thead><tr><th>视角</th><th>来源帧0候选可见 / 总点数</th><th>操作帧夹爪中心 8 cm 内 3 mm 格</th></tr></thead><tbody>{rows}</tbody></table><p>两路原视角并集 {u["old_voxels"]} 格；三路固定新增视角中，与原视角不重合的 {u["new_unique_voxels"]} 格；合并 {u["all_voxels"]} 格。这是每 3 像素抽样后 3 mm 网格的差异，受采样影响，也可能含夹爪和背景；不能当新增接触面积。</p><div class="toolbar"><button onclick="show('{case}','source')">看夹前观测来源</button><button onclick="show('{case}','action')">看操作帧</button><span id="{case}_mode"></span></div><canvas id="{case}_cloud" width="1050" height="550"></canvas><p class="small">拖动旋转点云；滚轮缩放。灰蓝＝原主视角、青＝原腕相机、橙＝新增正面、粉＝新增侧面、紫＝新增俯视。绿色为帧0既有来源候选，玫红为新增视角夹前未筛物体身份的邻域点；黄色十字是操作帧实测夹爪中心。点来自 RGB-D 实测深度，不是完整物体模型。</p><div class="panel"><h3>同帧 RGB 与深度</h3><p>绿色既有候选、玫红新增邻域点只叠在帧0深度一致的像素；操作帧黄色十字是夹爪中心几何投影，可能被遮挡。每个图片上排 RGB、下排深度；五列依次为原主视角、原腕相机、固定正面、固定侧面、固定俯视。</p><img id="{case}_img" src="{images[case]['source']}"></div></section>''')
style='''body{font:16px/1.55 system-ui,sans-serif;background:#0a1221;color:#e4ebf6;max-width:1480px;margin:auto;padding:28px}h1{font-size:30px}h2{border-top:1px solid #40506a;padding-top:32px;margin-top:44px}a{color:#8fcfff}section{margin-bottom:38px}.lead{background:#142239;padding:18px;border-radius:12px}.toolbar{display:flex;gap:10px;align-items:center;margin:16px 0}button{background:#1d385b;border:1px solid #6788ae;color:white;padding:9px 15px;border-radius:8px;cursor:pointer}button:hover{background:#325a88}canvas{width:100%;max-width:1050px;background:#080d17;border:1px solid #40506a;border-radius:12px;touch-action:none}.panel{background:#122037;border-radius:10px;padding:12px;margin-top:16px}.panel img{width:100%;max-width:1340px;height:auto}.small{font-size:13px;color:#b5c4d9}table{border-collapse:collapse;margin:10px 0}th,td{border:1px solid #40506a;padding:5px 12px}th{background:#1b314f}'''
js='''const DATA=__DATA__;const IMG=__IMG__;const colors=['#94a3b8','#48d8e9','#ffad4a','#f377bb','#a792ef','#41e684','#ffde58','#ff4d87'];const state={};function show(k,m){if(!state[k])state[k]={a:0.6,e:0.4,z:3.2};state[k].mode=m;document.getElementById(k+'_mode').textContent=m==='source'?'帧0：接触前观测来源':'操作帧：夹爪附近观测，尚无点级接触判定';document.getElementById(k+'_img').src=IMG[k][m];draw(k)}function draw(k){const c=document.getElementById(k+'_cloud'),ctx=c.getContext('2d'),s=state[k];ctx.clearRect(0,0,c.width,c.height);const ca=Math.cos(s.a),sa=Math.sin(s.a),ce=Math.cos(s.e),se=Math.sin(s.e);let p=[];for(const q of DATA[k][s.mode]){const x=q[0],y=q[1],z=q[2],u=x*ca-y*sa,v=x*sa+y*ca,sy=z*ce-v*se,d=z*se+v*ce;p.push([c.width/2+u*s.z,c.height/2-sy*s.z,d,q[3],q[4]])}p.sort((a,b)=>a[2]-b[2]);for(const q of p){ctx.beginPath();const r=q[4]===1||q[4]===3?3:q[4]===2?7:1.5;ctx.arc(q[0],q[1],r,0,2*Math.PI);ctx.fillStyle=colors[q[3]];ctx.globalAlpha=q[4]===0?0.54:1;ctx.fill()}ctx.globalAlpha=1}for(const k of Object.keys(DATA)){const c=document.getElementById(k+'_cloud');let down=false,px=0,py=0;c.addEventListener('pointerdown',e=>{down=true;px=e.clientX;py=e.clientY;c.setPointerCapture(e.pointerId)});c.addEventListener('pointermove',e=>{if(!down)return;state[k].a+=(e.clientX-px)*0.01;state[k].e=Math.max(-1.3,Math.min(1.3,state[k].e+(e.clientY-py)*0.01));px=e.clientX;py=e.clientY;draw(k)});c.addEventListener('pointerup',()=>down=false);c.addEventListener('wheel',e=>{e.preventDefault();state[k].z=Math.max(1,Math.min(12,state[k].z*(e.deltaY>0?0.9:1.1)));draw(k)},{passive:false});show(k,'source')}'''
js=js.replace('__DATA__',payload).replace('__IMG__',json.dumps(images,ensure_ascii=False))
page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>接触偏好 · 固定 RGB-D 多视角与点云</title><style>{style}</style><h1>固定 RGB-D 多视角：接触候选的可见性</h1><div class="lead"><p>从相同官方状态重放添加三路固定 RGB-D（正面、侧面、俯视），原示范只读。对照已开发的炉灶开关、抽屉和双物体搬运各一段；不同视角显示完全相同的帧号。下方可旋转的是实测深度点云。</p><p><strong>读图边界：</strong>绿色为前一版保存的手部附近表面来源候选，玫红为新增视角按同一几何定义补到、尚未核实物体身份的夹前邻域点，均非接触真值或新网络评分。炉灶玫红主要落在旁边另一控制件，不能并入目标区域；操作帧只有实测夹爪位置及附近可见点云，遮挡内表面仍未知。新增相机不能自动判断接触，尤其不能把背景、机械臂或另一物体计入接触面。</p><p>试验时间 {now}。所有结果均为派生文件，未读预留测试。</p></div>{''.join(blocks)}<script>{js}</script></html>'''
pagepath=docs/'接触偏好_固定RGBD多视角与点云核查.html'
pagepath.write_text(page,encoding='utf-8')
manifest={'created_beijing':now,'page':str(pagepath.relative_to(root)),'page_sha256':hashlib.sha256(pagepath.read_bytes()).hexdigest(),'images':{k:{m:hashlib.sha256((docs/v).read_bytes()).hexdigest() for m,v in a.items()} for k,a in images.items()},'cloud_points':{k:{m:len(v) for m,v in d.items()} for k,d in clouds.items()}}
(out/'gallery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps({'page':manifest['page'],'images':sum(len(x) for x in images.values()),'cloud_points':manifest['cloud_points'],'page_bytes':pagepath.stat().st_size},ensure_ascii=False))
