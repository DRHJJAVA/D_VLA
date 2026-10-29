from pathlib import Path
import os,sys,json,hashlib,html
import numpy as np,torch,h5py
from PIL import Image,ImageDraw
R=Path('/mnt/nvme1/drh/D_VLA');I=R/'experiments/region_pose/iter_003';O=Path(__file__).resolve().parent
sys.path[:0]=[str(I/'src'),str(R/'src/v3'),str(R/'src')]
from model import RegionPoseNet
from train import loadrows,forward
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
manifest=json.loads((I/'manifest.json').read_text())
rows=loadrows(manifest,torch.device('cuda'))
# Actual frozen checkpoint inference for every available query; no fitting or parameter selection.
pred={};hashes={}
for variant in ['object_only','pose_conditioned']:
 ckfile=I/'runs'/variant/'best.pt';hashes[variant]=hashlib.sha256(ckfile.read_bytes()).hexdigest()
 ck=torch.load(ckfile,map_location='cuda',weights_only=False);net=RegionPoseNet().cuda();net.load_state_dict(ck['state']);net.eval()
 with torch.inference_mode():
  for meta,d in rows:
   a,b=forward(net,d);pred.setdefault(meta['key'],{})[variant]=(a if variant=='object_only' else b).sigmoid().cpu().numpy()
 del net;torch.cuda.empty_cache()
pd=O/'predictions';pd.mkdir()
images=R/'docs/图示/第003轮真实预测';images.mkdir()
sections=[];files=[]
for meta,d in rows:
 key=meta['key'];scores=pred[key]
 np.savez_compressed(pd/(key+'.npz'),**scores)
 # Show every validation query and first available query for each training episode.
 if meta['split']=='train' and key!=next(r['key'] for r in manifest['rows'] if r['episode']==meta['episode']):continue
 with np.load(I/'samples'/(key+'.npz')) as z:
  pix=z['pixel'].astype(np.int64);rgb=z['rgb'];target=z['target']>.5;known=z['known'];pose=z['pose']
  with h5py.File(str(z['query_rgbd']),'r') as hf: shape=hf['agentview/rgb'].shape[1:3]
  canvas=np.full((*shape,3),127,np.uint8);canvas[pix[:,1],pix[:,0]]=rgb
  lo=pix.min(0);hi=pix.max(0)+1;x1,y1=lo;x2,y2=hi
  def render(tag,selected=None):
   im=Image.fromarray(canvas[y1:y2,x1:x2]).convert('RGB')
   # Nearest-neighbor display preserves the actual masked input pixels rather than inventing texture.
   im=im.resize((448,448),Image.Resampling.NEAREST)
   if selected is not None:
    draw=ImageDraw.Draw(im)
    for xy in pix[selected]:
     u=(xy[0]-x1+.5)/(x2-x1)*448;v=(xy[1]-y1+.5)/(y2-y1)*448
     col=(60,225,120) if tag=='target' else (255,213,38)
     draw.ellipse((u-1.3,v-1.3,u+1.3,v+1.3),fill=col)
   file=images/(key+'_'+tag+'.png');im.save(file);files.append(file)
   return '图示/第003轮真实预测/'+file.name
  imgs=[('实际单物体输入',render('input'))]
  for variant,title in [('object_only','只看物体：前20%高分点'),('pose_conditioned','加入当前夹爪：前20%高分点')]:
   order=np.argsort(-scores[variant],kind='stable');sel=np.zeros(len(pix),bool);sel[order[:int(np.ceil(.2*len(pix)))]]=True
   imgs.append((title,render(variant,sel)))
  tg=render('target',target)
  label='验证集（未参与训练）' if meta['split']=='val' else '训练集示例（不能证明泛化）'
  cells=''.join('<figure><figcaption>'+title+'</figcaption><img loading="lazy" src="'+url+'"></figure>' for title,url in imgs)
  body='<section><h2>'+html.escape(meta['episode'])+' · 帧'+str(meta['query_frame'])+'</h2><p>'+label+'；'+str(len(pix))+'个输入点，当前夹爪开度'+format(float(pose[-1]*100),'.2f')+'毫米。</p><div class="grid">'+cells+'</div><details><summary>显示自动融合目标，仅作监督对照</summary><p>绿色是自动估计的有限夹持范围，不是真实接触真值，也不是模型输出。未知区域不当负例。</p><img class="target" src="'+tg+'"></details></section>'
  sections.append((meta['split'],body))
head='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第003轮：真实模型预测与输入</title><style>body{font:16px system-ui,sans-serif;background:#f5f7fa;color:#17212d;margin:24px auto;max-width:1450px;padding:0 20px;line-height:1.7}h1{font-size:28px}.notice{padding:16px;border-left:5px solid #c28010;background:#fff4d6}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img{width:100%;image-rendering:auto}figcaption{font-weight:600}section{background:white;border:1px solid #d9e0e6;padding:16px;margin:20px 0}details{margin-top:15px}.target{max-width:448px}small{color:#555}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style><h1>第003轮：真实模型预测与输入</h1><div class="notice">第二步尚未验收。黄色只表示模型在全部输入点中评分最高的20%，不是物理接触真值或校准成功概率。每排使用同一单物体当前输入；条件版另输入当前夹爪位姿与开度。查询没有未来帧。</div><p>自动数据筛选后，实际训练11段/32查询，验证2段/6查询。验证仅罐头与瓷杯，白黄杯和番茄酱验证仍缺失，不将展示当作四物体泛化。所有图和模型留在远程。</p><p>输入图忠实呈现本轮DINO看到的遮罩裁剪，灰色是非输入像素；若出现稀疏灰孔，这是当前采样后的输入实现，不做美化补图。</p><h2>全部可评估验证查询</h2>'''
page=head+''.join(b for s,b in sections if s=='val')+'<h2>训练集示例：每段首个有效查询</h2>'+''.join(b for s,b in sections if s=='train')+'</html>'
dest=R/'docs/第003轮_真实模型预测与输入.html';dest.write_text(page)
lock={'checkpoint_sha256':hashes,'files':{str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files+[dest]+list(pd.glob('*.npz'))},'no_training_performed':True,'model_query_count':len(rows),'display_query_count':len(sections)}
(O/'locked.json').write_text(json.dumps(lock,indent=2))
print(json.dumps({'page':str(dest),'queries':len(rows),'display':len(sections),'images':len(files)}),flush=True)
