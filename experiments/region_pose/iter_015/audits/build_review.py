from pathlib import Path
import json,hashlib
import numpy as np,h5py
from PIL import Image,ImageDraw
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/region_pose/iter_015';V=B.parent/'reviews/iter_015';V.mkdir(exist_ok=False)
images=R/'docs/图示/第015轮早期查询';images.mkdir(exist_ok=False)
refs={r['id']:r for r in json.loads((B.parent/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
rows=json.loads((B/'val/summary.json').read_text())['rows']
parts=['<!doctype html><meta charset="utf-8"><title>第015轮 固定旧验证与来源去重</title><style>body{font:16px sans-serif;max-width:1300px;margin:32px auto;background:#f5f7fa;color:#152132}section{background:white;padding:24px;margin:24px 0}.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:100%}figure{margin:4px}td,th{padding:9px;border:1px solid #ccd}table{border-collapse:collapse}</style><h1>第015轮：固定旧验证与来源去重</h1><p>训练规则冻结后固定处理3个旧验证事件，共12查询：9个跨帧对应、3个参考自身。来源去重后只有罐头和瓷杯2事件可评估，白黄杯每个查询只有1个不同参考正点，均不纳入定量指标；番茄酱缺持物证据。没有新模型训练，没有读取测试。</p><p>彩色编号只是物体表面的跨帧对应，不是夹持区域或抓取预测。通过数值检查不等于逐点身份真值。参考时刻位姿仅作事后基线比较，不能作为早期查询模型输入。</p>']
for ident in sorted(set(r['episode'] for r in rows)):
 rr=[r for r in rows if r['episode']==ident]
 parts.append('<section><h2>'+ident+'</h2><table><tr><th>提前帧数</th><th>查询帧</th><th>正/负/未知</th><th>当前夹爪AP</th><th>参考夹爪AP：同一批点</th><th>末端平移差mm</th></tr>')
 for r in rr:
  parts.append('<tr><td>'+str(r['offset'])+'</td><td>'+str(r['query'])+'</td><td>'+str(r.get('positive'))+'/'+str(r.get('negative'))+'/'+str(r.get('unknown'))+'</td><td>'+format(r.get('early_geometry',{}).get('ap',float('nan')),'.4f')+'</td><td>'+format(r.get('reference_geometry_on_same_points',{}).get('ap',float('nan')),'.4f')+'</td><td>'+format(r.get('hand_translation_to_reference_mm',0),'.1f')+'</td></tr>')
 parts.append('</table>')
 support=json.loads((B/'audits/val_source_support.json').read_text())['rows']
 parts.append('<p>不同参考正点数（按15/10/5/1顺序）：'+str([r.get('unique_positive_sources') for r in support if r['episode']==ident])+'；像素数增加不等于独立支持增加。</p>')
 with h5py.File(refs[ident]['rgbd'],'r') as f:
  for r in rr:
   if r['query']==r['reference_query']:continue
   name=r['event']+'_query'+str(r['query']);z=np.load(B/'val'/(name+'.npz'))
   parts.append('<details'+(' open' if r['offset']==15 else '')+'><summary>闭合前'+str(r['offset'])+'帧：'+str(r['query'])+' → '+str(r['reference_query'])+'，数值对应'+('通过' if r.get('correspondence_valid') else '不足')+'</summary><div class="row">')
   for side,t,k in [('early',r['query'],'query_match_uv'),('reference',r['reference_query'],'reference_match_uv')]:
    pic=Image.fromarray(f['agentview/rgb'][t]).resize((672,672));d=ImageDraw.Draw(pic);s=672/f['agentview/rgb'].shape[1]
    if k in z:
     uv=z[k];ids=z['track_ids'];ins=z['inlier']
     for (u,v),idx,good in zip(uv,ids,ins):
      if not good:continue
      color=tuple(int(x) for x in np.random.default_rng(int(idx)+2026).integers(70,255,3));x,y=float(u*s),float(v*s)
      d.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2);d.text((x+3,y+2),str(idx),fill=color)
    fn=name+'_'+side+'.png';pic.save(images/fn);parts.append('<figure><img src="图示/第015轮早期查询/'+fn+'"><figcaption>'+('早期查询RGB，仅用于检查表面对应' if side=='early' else '闭合前参考RGB，不是早期输入')+'</figcaption></figure>')
   parts.append('</div></details>')
 parts.append('</section>')
page=R/'docs/第015轮_固定旧验证与来源去重诊断.html';page.write_text('\n'.join(parts))
files=list(images.glob('*.png'))+[page,Path(__file__)]
(V/'locked.json').write_text(json.dumps({str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
print('images',len(files)-2)
