from pathlib import Path
import json,hashlib
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('/mnt/nvme1/drh/D_VLA');R=B/'experiments/region_pose/iter_035';V=B/'experiments/region_pose/reviews/iter_035';V.mkdir(exist_ok=False);(V/'images').mkdir()
rec=json.loads((R/'summary.json').read_text());ident=rec['episode'];ref=json.loads((R/'refs.json').read_text())[ident]
z=np.load(next((R/'train/correspondences').glob('*.npz')));s=np.load(R/'seeds'/(ident+'.npz'))
rows=[]
with h5py.File(ref['rgbd'],'r') as f:
 g=f['agentview'];rgb=g['rgb'][36]
 fig,axes=plt.subplots(1,3,figsize=(15,5));axes[0].imshow(rgb);axes[0].set_title('Current RGB: query 36')
 axes[1].imshow(s['mask'],cmap='gray');axes[1].set_title('Causal eroded foreground')
 axes[2].imshow(rgb);uv=s['initial_uv'];axes[2].scatter(uv[:21,0],uv[:21,1],s=12,c='cyan');axes[2].scatter(uv[21:,0],uv[21:,1],s=9,c='magenta');axes[2].set_title('21 natural + 70 grid points')
 for ax in axes:ax.axis('off')
 fig.tight_layout();fig.savefig(V/'images/query36.png',dpi=130);plt.close(fig)
 for row in rec['endpoints']:
  t=row['frame'];ids=z[f'track_id_{t}'];ok=z[f'inlier_{t}'];a=z[f'query_uv_{t}'];b=z[f'bridge_uv_{t}'];im=g['rgb'][t]
  fig,axes=plt.subplots(1,2,figsize=(12,6))
  for ax,img,uv in zip(axes,[rgb,im],[a,b]):
   ax.imshow(img);ax.scatter(uv[ok,0],uv[ok,1],s=30,facecolors='none',edgecolors='lime',linewidths=.8)
   ax.scatter(uv[~ok,0],uv[~ok,1],s=20,c='orange',marker='x',linewidths=.7)
   for ix in np.linspace(0,len(ids)-1,min(18,len(ids)),dtype=int):ax.text(uv[ix,0]+2,uv[ix,1]-2,str(ids[ix]),color='cyan' if ids[ix]<21 else 'magenta',fontsize=7)
   ax.axis('off')
  axes[0].set_title('Query 36 (same track IDs)');axes[1].set_title(f'Held candidate {t}; inliers {row["inliers"]}/{row["matches"]}')
  fig.suptitle('Correspondence evidence ONLY - NOT grasp labels; green=inliers, orange=outliers\n'+('Later diagnostic only: do not cross failure at 65' if t==82 else ('Quality failed: ratio below 0.6' if t==65 else 'Predefined endpoint correspondence')))
  fig.tight_layout();name=f'query36_to_{t}.png';fig.savefig(V/'images'/name,dpi=130);plt.close(fig)
  rows.append({'frame':t,'image':name,**row})
alias=B/'docs/图示/iter_035';assert not alias.exists();alias.symlink_to(V/'images',target_is_directory=True)
html='<meta charset="utf-8"><title>第035轮 白黄杯6冻结跨时对应</title><style>body{max-width:1200px;margin:auto;font-family:sans-serif}img{max-width:100%}section{margin:2em 0}</style><h1>第035轮：白黄杯6冻结跨时对应</h1><p>仅1训练episode。对应点不是夹持标签或物理接触真值。绿色圈为刚体内点，橙色为外点，编号颜色区分原角点/网格。65帧失败后截断，82仅诊断。未新增监督，第二步未验收。</p><img src="图示/iter_035/query36.png">'
for row in rows:html+=f'<section><h2>36 → {row["frame"]}</h2><p>内点 {row["inliers"]}/{row["matches"]}；质量通过 {row["valid"]}。</p><img src="图示/iter_035/{row["image"]}"></section>'
page=B/'docs/第035轮_白黄杯6冻结跨时对应.html';page.write_text(html,encoding='utf-8')
(V/'manifest.json').write_text(json.dumps(rows,indent=2))
files=list((V/'images').glob('*.png'))+[V/'manifest.json',page]
(V/'immutable_locked.json').write_text(json.dumps({str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},ensure_ascii=False,indent=2))
print(json.dumps({'images':6,'page':str(page)}))
