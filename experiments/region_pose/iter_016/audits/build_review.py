from pathlib import Path
import sys,json,hashlib
import numpy as np,torch,h5py
from PIL import Image,ImageDraw
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose/iter_016';V=B.parent/'reviews/iter_016'
sys.path.insert(0,str(B/'src'))
from train import loadrows,forward
from model import RegionPoseNet
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
assert (B/'summary.json').exists()
V.mkdir(exist_ok=False);imgs=P/'docs/图示/第016轮模型对照';imgs.mkdir(exist_ok=False)
m=json.loads((B/'manifest.json').read_text());rows=loadrows(m,'cuda');rows=[(r,d) for r,d in rows if r['split']=='val']
names=['late_object_only','late_pose_conditioned','multi_object_only','multi_pose_conditioned'];scores={};audit=[]
for run in names:
 ck=torch.load(B/'runs'/run/'best.pt',map_location='cuda',weights_only=False);net=RegionPoseNet().cuda();net.load_state_dict(ck['state']);net.eval()
 with torch.inference_mode():
  for r,d in rows:
   a,b=forward(net,d);s=(a if run.endswith('object_only') else b).sigmoid().cpu().numpy()
   saved=np.load(B/'runs'/run/'val_predictions'/(r['key']+'.npz'))['score'];delta=float(np.max(np.abs(s-saved)));assert delta<1e-7
   scores[(run,r['key'])]=s;np.savez_compressed(V/(run+'_'+r['key']+'.npz'),score=s)
   audit.append({'run':run,'key':r['key'],'recomputed_score_max_difference':delta,'checkpoint_sha256':hashlib.sha256((B/'runs'/run/'best.pt').read_bytes()).hexdigest()})
 del net;torch.cuda.empty_cache()
html=['<!doctype html><meta charset="utf-8"><title>第016轮真实模型对照</title><style>body{font:16px sans-serif;background:#f4f6fa;color:#182431;max-width:1500px;margin:30px auto}.row{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}section{background:white;padding:18px;margin:20px 0}img{width:100%}figure{margin:0}</style><h1>第016轮：相同训练预算下的真实模型预测</h1><p>6训练事件24查询；这里完整显示罐头、瓷杯2个可评估旧验证事件的8个查询。白黄杯正来源不足、番茄酱无持物证据，没有纳入定量结果。没有盲测、没有阶段验收。</p><p>黄色为对应冻结best模型对全部输入点的前20%评分；不是自动目标或物理接触真值。第一列是实际完整物体RGB输入；各模型点云输入和当前夹爪信息相同。晚查询/多时刻组各192次更新，基础头不使用夹爪条件，条件头使用当前夹爪位姿。下面只显示模型输出，未混入目标。</p>']
for r,d in rows:
 z=np.load(B/'samples'/(r['key']+'.npz'));uv=z['input_rgb_pixel'];pix=z['pixel'];box=z['input_crop_box'];x1,y1,x2,y2=box
 with h5py.File(str(z['query_rgbd']),'r') as f:rgb=f['agentview/rgb'][int(z['query_frame'])]
 full=np.full_like(rgb,127);full[uv[:,1],uv[:,0]]=rgb[uv[:,1],uv[:,0]]
 html.append('<section><h2>'+r['episode']+'：闭合前'+str(r['offset'])+'帧（query '+str(r['query_frame'])+'）</h2><div class="row">')
 for label,run in [('完整RGB输入',None)]+[(x,x) for x in names]:
  pic=Image.fromarray(full[y1:y2,x1:x2]).resize((384,384))
  if run:
   s=scores[(run,r['key'])];ii=np.argsort(-s,kind='stable')[:int(np.ceil(.2*len(s)))];draw=ImageDraw.Draw(pic)
   for u,v in pix[ii]:
    xx=(u-x1+.5)/(x2-x1)*384;yy=(v-y1+.5)/(y2-y1)*384;draw.ellipse((xx-2,yy-2,xx+2,yy+2),fill=(255,215,0))
  fn=r['key']+'_'+(run or 'input')+'.png';pic.save(imgs/fn)
  html.append('<figure><img src="图示/第016轮模型对照/'+fn+'"><figcaption>'+label+'</figcaption></figure>')
 html.append('</div></section>')
page=P/'docs/第016轮_真实模型匹配预算对照.html';page.write_text('\n'.join(html))
(V/'prediction_audit.json').write_text(json.dumps(audit,indent=2))
files=list(imgs.glob('*.png'))+list(V.glob('*.npz'))+[V/'prediction_audit.json',page,Path(__file__)]
(V/'locked.json').write_text(json.dumps({str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print('REVIEW_COMPLETE',len(rows),len(audit),len(list(imgs.glob('*.png'))))
