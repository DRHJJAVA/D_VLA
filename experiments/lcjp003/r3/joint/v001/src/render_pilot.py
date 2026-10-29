import pathlib,sys,json,h5py,numpy as np
from PIL import Image,ImageDraw,ImageFont
N=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r3/joint/v001')
sys.path.insert(0,str(N/'src'))
from geometry import ROOT,policy_input,CAMS,frame_index
from supervision import supervision
from visibility import visible_reference
ep=json.loads((N/'episodes.json').read_text())[0]
report=json.loads((N/'checks/supervision_pilot.json').read_text())
D=ROOT/'docs/LCJP003_R3_supervision_assets';D.mkdir(exist_ok=True)
rows=[]
for rec in report['windows']:
 t=rec['frame'];d,y,objects,meta=policy_input(ep,t);s=supervision(ep,t,objects,d);k=int(s['role'].argmax())
 if k==len(objects):continue
 obj=objects[k]
 with h5py.File(ROOT/'experiments/contact_joint_object/cache/rgbd'/(ep['id']+'.h5'),'r') as h:
  ix=frame_index(h,t);cameras=[dict(K=h[c+'/K'][ix],camera_to_world=h[c+'/camera_to_world'][ix],depth=h[c+'/depth_m'][ix],robot_mask=h[c+'/robot_mask'][ix]) for c in CAMS]
  xyz=d['xyz'][k].numpy();vis=visible_reference(xyz,cameras);valid=d['valid'][k].numpy()
  panels=[]
  for v,c in enumerate(CAMS):
   raw=h[c+'/rgb'][ix];im=Image.fromarray(raw);dr=ImageDraw.Draw(im)
   mask=obj['masks'][v];pix=np.array(im).astype(float);pix[mask]=pix[mask]*.55+np.array([0,220,255])*.45;im=Image.fromarray(pix.astype('uint8'));dr=ImageDraw.Draw(im)
   values=s['parts'][k];ids=np.flatnonzero(valid&vis['visible'][:,v])
   if len(ids):
    for j in sorted(ids,key=lambda x:values[x]):
     x,y=vis['uv'][j,v]
     if values[j]>values.max()*.1:dr.ellipse((x-1,y-1,x+1,y+1),fill=(255,int(220*(1-values[j]/max(values.max(),1e-9))),0))
   panels.append(im.resize((512,512)))
  canvas=Image.new('RGB',(1024,554),'white');canvas.paste(panels[0],(0,42));canvas.paste(panels[1],(512,42))
  ImageDraw.Draw(canvas).text((12,12),f"TRAIN OBSERVATION t={t} | teacher target ids={obj['ids']} | q={s['role'][k]:.3f} | weight={s['role_weight']:.3f}",fill='black')
  file=f'frame_{t}.png';canvas.save(D/file);rows.append(f'<h2>训练示范 第{t}帧：自动监督核对（不是模型结果）</h2><img src="LCJP003_R3_supervision_assets/{file}" style="width:1024px;max-width:100%">')
html='<meta charset="utf-8"><title>R3 示范监督核对</title><body style="font-family:sans-serif;background:#eee;padding:20px"><h1>字母汤罐：训练监督检查</h1><p>左：外部相机；右：腕部相机。青色：监督分布最高的候选；暖色：部位支持。仅检查独立监督来源，尚未训练。</p>'+''.join(rows)+'</body>'
(ROOT/'docs/LCJP003_R3_监督核对.html').write_text(html)
print(json.dumps({'page':'LCJP003_R3_监督核对.html','frames':len(rows)}))
