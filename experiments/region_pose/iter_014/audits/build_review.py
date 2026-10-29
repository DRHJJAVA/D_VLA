from pathlib import Path
import json,hashlib
import numpy as np,h5py
from PIL import Image,ImageDraw
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/region_pose/iter_014';V=B.parent/'reviews/iter_014';V.mkdir(exist_ok=False)
images=R/'docs/图示/第014轮早期查询';images.mkdir(exist_ok=False)
refs={r['id']:r for r in json.loads((B.parent/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
rows=json.loads((B/'train/summary.json').read_text())['rows']
parts=['<!doctype html><meta charset="utf-8"><title>第014轮 更早查询与几何提示</title><style>body{font:16px sans-serif;max-width:1300px;margin:32px auto;background:#f5f7fa;color:#152132}section{background:white;padding:24px;margin:24px 0}.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:100%}figure{margin:4px}td,th{padding:9px;border:1px solid #ccd}table{border-collapse:collapse}</style><h1>第014轮：更早查询与夹爪几何提示</h1><p>仅6个训练事件。固定闭合前15/10/5/1帧，共24个查询，其中6个是已锁定参考本身，只有18个实际跨帧追踪。没有新模型训练，没有读取验证/测试。</p><p>彩色编号只是物体表面的跨帧对应，不是夹持区域或抓取预测。通过数值检查不等于逐点身份真值。参考时刻位姿仅作事后基线比较，不能作为早期查询模型输入。</p>']
for ident in sorted(set(r['episode'] for r in rows)):
 rr=[r for r in rows if r['episode']==ident]
 parts.append('<section><h2>'+ident+'</h2><table><tr><th>提前帧数</th><th>查询帧</th><th>正/负/未知</th><th>当前夹爪AP</th><th>参考夹爪AP：同一批点</th><th>末端平移差mm</th></tr>')
 for r in rr:
  parts.append('<tr><td>'+str(r['offset'])+'</td><td>'+str(r['query'])+'</td><td>'+str(r.get('positive'))+'/'+str(r.get('negative'))+'/'+str(r.get('unknown'))+'</td><td>'+format(r.get('early_geometry',{}).get('ap',float('nan')),'.4f')+'</td><td>'+format(r.get('reference_geometry_on_same_points',{}).get('ap',float('nan')),'.4f')+'</td><td>'+format(r.get('hand_translation_to_reference_mm',0),'.1f')+'</td></tr>')
 parts.append('</table>')
 with h5py.File(refs[ident]['rgbd'],'r') as f:
  for r in rr:
   if r['query']==r['reference_query']:continue
   name=r['event']+'_query'+str(r['query']);z=np.load(B/'train'/(name+'.npz'))
   parts.append('<details'+(' open' if r['offset']==15 else '')+'><summary>闭合前'+str(r['offset'])+'帧：'+str(r['query'])+' → '+str(r['reference_query'])+'，数值对应'+('通过' if r.get('correspondence_valid') else '不足')+'</summary><div class="row">')
   for side,t,k in [('early',r['query'],'query_match_uv'),('reference',r['reference_query'],'reference_match_uv')]:
    pic=Image.fromarray(f['agentview/rgb'][t]).resize((672,672));d=ImageDraw.Draw(pic);s=672/f['agentview/rgb'].shape[1]
    if k in z:
     uv=z[k];ids=z['track_ids'];ins=z['inlier']
     for (u,v),idx,good in zip(uv,ids,ins):
      if not good:continue
      color=tuple(int(x) for x in np.random.default_rng(int(idx)+2026).integers(70,255,3));x,y=float(u*s),float(v*s)
      d.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2);d.text((x+3,y+2),str(idx),fill=color)
    fn=name+'_'+side+'.png';pic.save(images/fn);parts.append('<figure><img src="图示/第014轮早期查询/'+fn+'"><figcaption>'+('早期查询RGB，仅用于检查表面对应' if side=='early' else '闭合前参考RGB，不是早期输入')+'</figcaption></figure>')
   parts.append('</div></details>')
 parts.append('</section>')
page=R/'docs/第014轮_更早查询与几何提示诊断.html';page.write_text('\n'.join(parts))
files=list(images.glob('*.png'))+[page,Path(__file__)]
(V/'locked.json').write_text(json.dumps({str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
print('images',len(files)-2)
