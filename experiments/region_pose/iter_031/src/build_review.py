from pathlib import Path
import json,hashlib,html,numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];R=B.parents[2]
D=R/'docs';G=D/'图示/iter_031';G.mkdir(parents=True)
V=B.parent/'reviews/iter_031';V.mkdir(parents=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=json.loads((B/'config.json').read_text());m=json.loads((B/'manifest.json').read_text())
rows=[r for r in m['rows'] if r['split']=='val' or (r['split']=='train' and r['offset']==15)]
runs=['control029_object_only','control029_pose_conditioned','expanded10_object_only','expanded10_pose_conditioned']
summ=json.loads((B/'evaluation/summary.json').read_text())
best={s['run']:s for s in summ if s['checkpoint']=='best'}
page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>031真实模型排序负子集对照</title><style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:24px;background:#f5f6fa;color:#162333}img{max-width:100%;height:auto}section{background:white;margin:24px 0;padding:20px;border-radius:10px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd}.models{display:grid;grid-template-columns:1fr 1fr;gap:16px}.warn{background:#fff0ca;padding:16px}a{color:#145a85}</style>',
'<h1>第031轮：排序负子集单因素真实预测对照</h1>',
'<p>control029为029扩展10事件冻结控制；expanded10为031新训练。四个best模型均用相同10训练事件40查询和3旧验证事件12查询，同初始化、同计划、240更新预算。031只改变训练rank负子集为最高logit的最多64个已知负点；完整BCE与0.1 rank系数、条件0.5辅助、网络和输入均不变。best仍按原029验证头loss选择，实际best步数不同，last240步另见evaluation。</p>',
'<p class="warn">黄色为全部输入点前20%模型评分，未先剔除未知；连续色图为非校准sigmoid分数。绿色自动监督仅折叠显示，不是模型预测或物理接触真值。12验证查询仅3独立事件，白黄杯只有3正来源，番茄验证仍缺失。第二步尚未验收。</p>',
'<p>另展示10训练事件各最早查询，四模型均训练过这些事件。soup5弱8/9持物内点、瓷杯5的58帧高敏感仍保留，白黄杯5零正排除、薄空前景、4无持物和侧向不足仍未解决。</p>',
'<p class="warn">031.1核查发现旧029/030当前几何基线对短云用了错误的点序索引。031已按实际样本causal_fps_indices修正：40训练平均AP 0.119275→0.219665；12旧验证0.206372→0.265695。中心、随机及所有模型预测/损失/最优选择不变，旧版保留，逐查询更正见audits/baseline_index_correction.json。</p>',
'<table><tr><th>run</th><th>best轮</th><th>原头loss</th><th>AP</th><th>10/20/30%已知覆盖</th></tr>']
for run in runs:
 s=best[run];v=s['validation']['mean']
 page.append(f"<tr><td>{run}</td><td>{s['epoch']}</td><td>{v['loss']:.5f}</td><td>{v['ap']:.4f}</td><td>{v['coverage10']:.3f}/{v['coverage20']:.3f}/{v['coverage30']:.3f}</td></tr>")
page.append('</table><p>逐事件、提前量、1/2/5/10/20/30%全点预算、随机64次/中心/当前几何基线完整保存在本轮evaluation。覆盖率高不能替代精确排序，未知不算负例。</p>')
manifest=[];slicehash={}
def save(fig,name):
 p=G/name;fig.savefig(p,dpi=130,bbox_inches='tight');plt.close(fig)
 return str(p.relative_to(D))
def img(path):return '<a href="'+html.escape(path)+'"><img loading="lazy" src="'+html.escape(path)+'"></a>'
for row in rows:
 key=row['key'];z=np.load(B/'samples'/(key+'.npz'));a=np.load(row['target_source'])
 with h5py.File(str(z['query_rgbd']),'r') as f:
  rgb=f['agentview/rgb'][int(z['query_frame'])];dep=f['agentview/depth_m'][int(z['query_frame'])]
 for name,array in [('rgb',rgb),('depth_m',dep)]:slicehash[key+'/'+name]=hashlib.sha256(array.tobytes()).hexdigest()
 pix=z['pixel'].astype(int);full=a['query_pixel'].astype(int)
 assert np.array_equal(rgb[pix[:,1],pix[:,0]],z['rgb'])
 mask=np.full_like(rgb,127);mask[full[:,1],full[:,0]]=rgb[full[:,1],full[:,0]]
 fig,ax=plt.subplots(1,3,figsize=(12,4))
 for aax,data,title in zip(ax,[rgb,np.where(dep>0,dep,np.nan),mask],['Current full RGB','Current depth (m)','Complete causal foreground']):
  aax.imshow(data);aax.set_title(title);aax.axis('off')
 inputpath=save(fig,key+'_input.png')
 fig,ax=plt.subplots(figsize=(6,6));ax.imshow(rgb)
 known=a['known'];pos=a['target'];colors=np.full((len(full),3),.55);colors[known&~pos]=[.9,.2,.2];colors[pos]=[0,.95,.2]
 ax.scatter(full[:,0],full[:,1],s=7,c=colors,alpha=.85);ax.set_title('AUTOMATIC supervision: green + / red - / grey unknown');ax.axis('off')
 targetpath=save(fig,key+'_automatic.png')
 new=False
 page.append('<section id="'+key+'"><h2>'+html.escape(row['split']+' / '+key)+f"</h2><p>闭合前{row['offset']}帧；正/负不同参考来源 {row['unique_positive']}/{row['unique_negative']}。"+('控制组未训练此事件。' if row['split']=='train' and new else '')+'</p>'+img(inputpath)+'<div class="models">')
 predictions={}
 for run in runs:
  score=np.load(B/'evaluation'/run/'best'/(key+'.npz'))['score'];assert len(score)==len(pix)
  ids=np.argsort(-score,kind='stable')[:max(1,int(np.ceil(.2*len(score))))]
  fig,ax=plt.subplots(1,2,figsize=(10,5))
  ax[0].imshow(rgb);p=ax[0].scatter(pix[:,0],pix[:,1],c=score,s=8,cmap='viridis',vmin=0,vmax=1);fig.colorbar(p,ax=ax[0],fraction=.04)
  ax[0].set_title('Frozen best: all input point scores')
  ax[1].imshow(rgb);ax[1].scatter(pix[:,0],pix[:,1],s=3,c='#777777',alpha=.3);ax[1].scatter(pix[ids,0],pix[ids,1],s=12,c='#ffff00',edgecolors='#9f7e00',linewidths=.2);ax[1].set_title('Top 20% of ALL input points')
  for aax in ax:aax.axis('off')
  path=save(fig,key+'_'+run+'.png');predictions[run]=path
  rr=next(r for r in best[run]['validation' if row['split']=='val' else 'all_training_diagnostic']['rows'] if r['key']==key)
  page.append('<div><h3>'+run+'</h3>'+img(path)+f"<p>AP {rr['ap']:.4f}；前20%观测正覆盖 {rr['all_coverage20']:.3f}；入选未知 {rr['all_unknown_fraction20']:.1%}，已知负 {rr['all_known_negative_fraction20']:.1%}。</p></div>")
 page.append('</div><details><summary>冻结自动监督对照（不是模型预测）</summary>'+img(targetpath)+'</details></section>')
 manifest.append(dict(key=key,split=row['split'],episode=row['episode'],offset=row['offset'],input=inputpath,automatic_supervision=targetpath,predictions=predictions))
page.append('</html>')
dest=D/'第031轮_真实模型排序负子集对照.html';dest.write_text(''.join(page),encoding='utf-8')
(V/'manifest.json').write_text(json.dumps(manifest,indent=2));(B/'audits/review_image_slices.json').write_text(json.dumps(slicehash,indent=2))
files=[dest,V/'manifest.json']+list(G.glob('*.png'))
(V/'immutable_locked.json').write_text(json.dumps({str(p.relative_to(R)):sha(p) for p in files},indent=2))
assert len(rows)==22 and len(list(G.glob('*.png')))==132
(B/'audits/prediction_review_audit.json').write_text(json.dumps(dict(displayed_queries=22,validation_queries=12,training_examples=10,models=4,prediction_images=88,input_images=22,automatic_target_images=22,total_images=132,actual_best_inference_from_evaluation=True,validation_predictions_exact=True,not_user_acceptance=True),indent=2))
old_slices=json.loads((B.parent/'iter_029/audits/review_image_slices.json').read_text())
assert all(old_slices[k]==v for k,v in slicehash.items())
(B/'audits/review_slice_comparison.json').write_text(json.dumps({'current_rgb_depth_slices':len(slicehash),'equal029':True,'no_future_query_input':True},indent=2))
print('REVIEW_COMPLETE',132,flush=True)
