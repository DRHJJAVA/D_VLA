from pathlib import Path
import json,hashlib,html
import numpy as np,h5py
from PIL import Image,ImageDraw
R=Path('/mnt/nvme1/drh/D_VLA');I=R/'experiments/region_pose/iter_006';O=Path(__file__).resolve().parent
refs={r['id']:r for r in json.loads((I.parent/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
summary=json.loads((I/'summary.json').read_text());D=R/'docs/图示/第006轮对应诊断';D.mkdir()
sections=[];files=[]
for row in summary['cases']:
 key=row['event']+'_query'+str(row['query']);nf=I/'correspondences'/(key+'.npz')
 body='<section><h2>'+html.escape(row['id'])+' · 查询帧'+str(row['query'])+'</h2><p>'+html.escape(row['split'])+'；未通过。原因：'+html.escape(row['reason'])+'</p>'
 if nf.exists():
  with np.load(nf) as z,h5py.File(refs[row['id']]['rgbd'],'r') as f:
   for bridge in row.get('bridge_results',[]):
    t=bridge['frame'];a=z['query_uv_'+str(t)];b=z['bridge_uv_'+str(t)];inside=z['inlier_'+str(t)]
    if not len(a):continue
    alluv=np.r_[a,b];lo=np.maximum(0,np.floor(alluv.min(0)-24).astype(int));hi=np.minimum(448,np.ceil(alluv.max(0)+25).astype(int));x1,y1=lo;x2,y2=hi
    urls=[]
    for phase,frame,uv in [('query',row['query'],a),('bridge',t,b)]:
     rgb=f['agentview/rgb'][frame];im=Image.fromarray(rgb[y1:y2,x1:x2]).resize((448,448))
     draw=ImageDraw.Draw(im)
     for k,xy in enumerate(uv):
      u=(xy[0]-x1)/(x2-x1)*448;v=(xy[1]-y1)/(y2-y1)*448;color=(0,230,240) if inside[k] else (230,230,230)
      draw.ellipse((u-3,v-3,u+3,v+3),outline=color,width=2);draw.text((u+4,v),str(k),fill=color,stroke_width=1,stroke_fill=(20,20,20))
     path=D/(key+'_to'+str(t)+'_'+phase+'.png');im.save(path);files.append(path);urls.append('图示/第006轮对应诊断/'+path.name)
    desc='桥接帧'+str(t)+'：端点匹配'+str(bridge.get('matches',0))+'，刚体内点'+str(bridge.get('inliers',0))+'；单桥接通过='+str(bridge['valid'])
    body+='<h3>'+desc+'</h3><div class="grid"><figure><figcaption>查询来源画面（不是评分网络输入）</figcaption><img src="'+urls[0]+'"></figure><figure><figcaption>桥接来源画面；同编号为追踪对应</figcaption><img src="'+urls[1]+'"></figure></div>'
 if row.get('dual_bridge_difference'):
  body+='<p>两个桥接映射到参考坐标的差异：中位'+format(row['dual_bridge_difference']['median_mm'],'.2f')+'毫米，95分位'+format(row['dual_bridge_difference']['p95_mm'],'.2f')+'毫米；超出3/6毫米门槛，拒绝生成目标。</p>'
 sections.append(body+'</section>')
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第006轮：夹前表面对应诊断</title><style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:20px;line-height:1.7;background:#f4f6f8}section{background:white;padding:18px;border:1px solid #ccd6df;margin:20px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img{width:100%}.warn{background:#fff2ce;padding:16px}</style><h1>第006轮：夹前表面对应诊断</h1><p class="warn">10个待修复查询全部未通过双桥接一致性；没有生成新训练目标，没有新模型训练。本页是来源对应诊断，不是抓取评分图。青色编号为刚体内点，浅灰为未通过刚体一致性的点，不表示接触区域。两个桥接共享来源，不能当成两次独立抓取。</p>'''+''.join(sections)+'</html>'
dest=R/'docs/第006轮_夹前表面对应诊断.html';dest.write_text(page);files.append(dest)
(O/'locked.json').write_text(json.dumps({str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files+[Path(__file__)]},indent=2))
print('review complete',len(files)-1,'images')
