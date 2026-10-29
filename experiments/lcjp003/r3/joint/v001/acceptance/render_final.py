import os,sys,pathlib,json,time,traceback,html,hashlib
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,h5py
from PIL import Image
N=pathlib.Path(__file__).resolve().parents[1];ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA')
sys.path[:0]=[str(N/'src'),str(N/'engineering')]
import geometry as g
from fps_fast import fps
from visibility import visible_reference
g.fps=fps
D=ROOT/'docs';A=D/'LCJP003_R3_final_assets';A.mkdir(exist_ok=True)
names={'alphabet_soup':'字母汤罐','bbq_sauce':'烧烤酱','butter':'黄油','chocolate_pudding':'巧克力布丁','cream_cheese':'奶油奶酪','ketchup':'番茄酱','milk':'牛奶','orange_juice':'橙汁','salad_dressing':'沙拉酱','tomato_sauce':'番茄沙司'}
try:
 data=json.loads((N/'eval3000.json').read_text());es={e['id']:e for e in json.loads((N/'episodes.json').read_text())};cards=[];checks=[]
 for i,r in enumerate(data['records']):
  eid=r['episode'];t=r['frame'];e=es[eid]
  d,y,objects,meta=g.policy_input(e,t)
  asset=N/'eval_assets/step3000'/f'{eid}_{t}.npz'
  with np.load(asset) as z:
   assert np.array_equal(z['xyz'],d['xyz'].numpy()),'coordinate_order'
   assert np.array_equal(z['valid'],d['valid'].numpy()),'valid_order'
   selected=int(z['selected']);alpha=z['alpha'].copy();valid=z['valid'].copy()
  cams=[];rgbs=[]
  with h5py.File(ROOT/'experiments/contact_joint_object/cache/rgbd'/f'{eid}.h5','r') as h:
   assert int(h['frame_indices'][meta['index']])==t
   for name in g.CAMS:
    z=h[name];j=meta['index'];K=z['K'][()];K=K[j] if K.ndim==3 else K;T=z['camera_to_world'][()];T=T[j] if T.ndim==3 else T
    cams.append(dict(K=K,camera_to_world=T,depth=z['depth_m'][j],robot_mask=z['robot_mask'][j].astype(bool)));rgbs.append(z['rgb'][j])
  panels=[];counts=[]
  if selected<len(objects):
   o=objects[selected];assert o['ids']==r['selected_candidate_ids'],'candidate_ids'
   score=alpha[selected][valid[selected]];assert len(score)==len(o['xyz'])
   vis=visible_reference(o['xyz'],cams)
   order=np.argsort(-score,kind='stable');keep=order[:int(np.ceil(.8*len(order)))];rank=np.zeros(len(order));rank[keep]=1-(np.arange(len(keep))+.5)/len(keep)
   colors=np.zeros((len(order),3),np.uint8)
   for lo,col in [(0,(45,110,220)),(.6,(35,185,110)),(.8,(245,210,45)),(.9,(255,135,30)),(.95,(235,45,45))]:colors[rank>=lo]=col
   chosen=np.zeros(len(order),bool);chosen[keep]=True
   for v in range(2):
    rgb=rgbs[v];mask=o['masks'][v].astype(bool)&~cams[v]['robot_mask'];maskim=rgb.copy();maskim[mask]=(rgb[mask]*.45+np.array([0,220,210])*.55).astype(np.uint8)
    scored=rgb.copy();vv=vis['visible'][:,v]&chosen;uv=vis['uv'][vv,v].astype(int)
    scored[uv[:,1],uv[:,0]]=colors[vv]
    panels.extend([rgb,maskim,scored]);counts.append(int(vv.sum()))
  else:
   for rgb in rgbs:panels.extend([rgb,rgb,rgb])
   counts=[0,0]
  image=Image.new('RGB',(768,512))
  for j,p in enumerate(panels):image.paste(Image.fromarray(p).convert('RGB'),((j%3)*256,(j//3)*256))
  fn=f'window_{i:03d}.png';image.save(A/fn)
  key=next((k for k in names if f'pick_up_the_{k}_and' in eid),'unknown');cn=names.get(key,key);split='开发集' if r['split']=='dev' else '训练集'
  ss=alpha[selected][valid[selected]] if selected<len(objects) else np.array([])
  text=f'原始分数范围 {ss.min():.6f}–{ss.max():.6f}' if len(ss) else '模型选择未知目标'
  cards.append(f'<article data-split="{r["split"]}" data-task="{key}"><h2>{i+1:02d} · 抓取{cn}并放入篮子</h2><p>{split} · {e["episode"]} · 第{t}帧 · 模型候选{r["selected_candidate_ids"]} · 选择权重{r["target_confidence"]:.1%}</p><div class="labels"><span>原始画面</span><span>模型所选候选的分割</span><span>所选目标的部位评分</span></div><img loading="lazy" src="LCJP003_R3_final_assets/{fn}" width="768" height="512"><p>上排：外部相机；下排：腕部相机。可见着色点：{counts[0]} / {counts[1]}。</p><p>{text} · 动作MAE {r["normalized_motion_mae"]:.4f} · 夹爪错误率 {r["gripper_error"]:.2%}</p></article>')
  checks.append(dict(index=i,episode=eid,frame=t,selected=selected,visible_colored_points=counts,coordinates_exact=True,frame_exact=True))
  (N/'acceptance/progress.json').write_text(json.dumps({'completed':i+1,'total':80}))
 summary=data['summary']
 doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>R3 第3000次训练验收</title><style>body{background:#101722;color:#e3eaf4;font:16px system-ui;margin:0 auto;max-width:1100px;padding:28px}h1{font-size:30px}p{line-height:1.7;color:#b9c6d8}header{border-bottom:1px solid #40506a;padding-bottom:20px}article{background:#192434;border:1px solid #35455b;border-radius:12px;padding:20px;margin:24px 0}img{display:block;width:100%;height:auto;image-rendering:auto}.labels{display:grid;grid-template-columns:repeat(3,1fr);text-align:center;padding:10px;background:#25354b}select{padding:10px;background:#25354b;color:white;border:1px solid #627593;margin-right:12px}.legend{display:flex;gap:15px;flex-wrap:wrap}.legend b{padding:6px} .note{color:#ffcb72}</style><header><h1>R3 · 第3000次训练验收</h1><p>固定80窗口：训练40、开发40。使用保存的第3000次模型输出；目标由模型自己选择，未按任务名称人工替换。</p><p>两路均做分割，模型从三维候选中选择目标；只给所选目标着色。上排外部相机，下排腕部相机。遮挡点不投影到遮挡物上。</p><p class="note">颜色表示目标内相对排名，不代表绝对高分或成功概率。先保留最高80%真实点，再在保留点内排名。低分与完整点云仍保存在原评估数据中。</p><div class="legend"><b style="color:#eb2d2d">95–100% 红</b><b style="color:#ff871e">90–95% 橙</b><b style="color:#f5d22d">80–90% 黄</b><b style="color:#23b96e">60–80% 绿</b><b style="color:#2d6edc">0–60% 蓝</b><b style="color:#00dcd2">青色：候选掩膜</b></div><p>这是固定示范观测上的离线验收，尚未进行本轮闭环抓取。选错目标、评分分散和分割错误均原样保留。</p>'''
 doc+=f'<p>训练动作MAE {summary["train"]["normalized_motion_mae"]:.4f}；开发 {summary["dev"]["normalized_motion_mae"]:.4f}。开发原/均匀/打乱：{summary["dev"]["normalized_motion_mae"]:.4f} / {summary["dev"]["uniform_mae"]:.4f} / {summary["dev"]["shuffle_mae"]:.4f}。</p>'
 doc+='<select id="split"><option value="dev">开发集40窗口</option><option value="train">训练集40窗口</option><option value="all">全部80窗口</option></select><select id="task"><option value="all">全部任务</option>'+''.join(f'<option value="{k}">{v}</option>' for k,v in names.items())+'</select></header>'+''.join(cards)
 doc+='''<script>function filter(){document.querySelectorAll('article').forEach(a=>a.hidden=!((split.value==='all'||a.dataset.split===split.value)&&(task.value==='all'||a.dataset.task===task.value)))}const split=document.getElementById('split'),task=document.getElementById('task');split.onchange=task.onchange=filter;filter();</script></html>'''
 page=D/'LCJP003_R3_训练后验收.html';page.write_text(doc)
 (N/'acceptance/render_complete.json').write_text(json.dumps(dict(page=str(page),windows=80,records=checks,checkpoint_sha256=hashlib.sha256((N/'checkpoints/step3000.pt').read_bytes()).hexdigest()),indent=2))
except Exception as ex:
 (N/'acceptance/failure.json').write_text(json.dumps(dict(error_type=type(ex).__name__,message=str(ex)[:160],frames=[dict(file=f.filename,line=f.lineno) for f in traceback.extract_tb(ex.__traceback__)])))
