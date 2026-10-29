from pathlib import Path
import sys,json,hashlib,html,ast,numpy as np,h5py
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/region_pose';B=P/'iter_028'
sys.path.insert(0,str(P/'iter_015/src'));import diagnose_validation as m
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
for lock in ['source_locked.json','input_locked.json','propagation_source_locked.json','review_source_locked.json']:
 for f,h in json.loads((B/lock).read_text()).items():assert sha(R/f)==h
assert (P/'iter_027/pre_validation_freeze.json').stat().st_mtime<(B/'设计方案.md').stat().st_mtime
summary=json.loads((B/'summary.json').read_text());cases={x['event']:x for x in json.loads((B/'cases.json').read_text())['rows']}
checks=[];baselines=[]
for rec in summary['rows']:
 key=f"{rec['event']}_query{rec['query']}";a=np.load(B/'val'/(key+'.npz'));ref=np.load(cases[rec['event']]['target_source'])
 known=a['known'];y=a['target'];idx=a['reference_source_index'];sel=a['selected_index']
 if rec['query']==rec['reference_query']:
  exk=ref['known'].astype(bool);exy=ref['target'].astype(bool);conf=ref['confidence']
 else:
  tree=cKDTree(ref['query_xyz']);transforms=[a['query_to_reference_world']]+list(a['loo_query_to_reference_world'])
  mapped=[m.apply(a['query_xyz'],t) for t in transforms];pairs=[tree.query(v) for v in mapped]
  ids=np.stack([x[1] for x in pairs]);dd=np.stack([x[0] for x in pairs])
  spread=np.max(np.stack([np.linalg.norm(x-mapped[0],axis=1) for x in mapped]),axis=0)
  assert np.array_equal(ids,a['all_transform_reference_index']);assert np.max(abs(spread-a['position_sensitivity_m']))<1e-10
  exk=np.all(ref['known'][ids],axis=0)&np.all(ref['target'][ids]==ref['target'][ids[0]],axis=0)&np.all(dd<=.002,axis=0)&(spread<=.002)
  if len(transforms)-1<7:exk[:]=False
  exy=ref['target'][ids[0]].astype(bool)&exk;conf=np.min(ref['confidence'][ids],axis=0)*np.exp(-spread/.002);conf[~exk]=0
 assert np.array_equal(known,exk) and np.array_equal(y,exy)
 assert np.max(abs(a['confidence']-conf))<1e-7
 assert np.all(a['confidence'][~known]==0)
 assert len(np.unique(idx[sel][y[sel]]))==rec['positive_sources']
 assert len(np.unique(idx[sel][known[sel]&~y[sel]]))==rec['negative_sources']
 raw=np.load(R/rec['source_file']);center=-np.linalg.norm(a['query_xyz']-a['query_xyz'].mean(0),axis=1)
 baselines.append({'event':rec['event'],'query':rec['query'],'offset':rec['offset'],'current_geometry':m.metrics(raw['early_geometry_score'],y,known),'center':m.metrics(center,y,known),'posthoc_reference_not_query_input':m.metrics(raw['reference_geometry_score'],y,known)})
 checks.append({'key':key,'transform_neighbor_query_reconstructed':True,'targets_confidence_sources_equal':True})
dump(B/'audits/propagation_independent.json',{'rows':checks,'passed':True})
dump(B/'audits/baselines.json',{'rows':baselines,'no_model_scores':True,'random_expected_coverage_in_metrics':True})
train=json.loads((P/'iter_027/summary.json').read_text())['rows'];chosen={'alphabet_soup_demo_5','alphabet_soup_demo_7','porcelain_mug_demo_5','white_yellow_mug_demo_4'}
selected=[('train',x,P/'iter_027/train') for x in train if x['episode'] in chosen]+[('val',x,B/'val') for x in summary['rows']]
protocol={x['id']:x for x in json.loads((P/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
D=R/'docs/图示/iter_028';D.mkdir(parents=True,exist_ok=False)
V=P/'reviews/iter_028';V.mkdir(parents=True,exist_ok=False)
page=R/'docs/第028轮_早期查询自动监督与来源.html';manifest=[];slices={};parts=[]
def save(fig,name):
 fig.savefig(D/name,dpi=140,bbox_inches='tight');plt.close(fig)
 return str((D/name).relative_to(R/'docs'))
def axes_off(ax):
 ax.set_xlim(-.5,447.5);ax.set_ylim(447.5,-.5);ax.axis('off')
def overlay(ax,rgb,uv,y,known,title,selection=None):
 ax.imshow(rgb)
 mask=np.ones(len(uv),bool) if selection is None else np.isin(np.arange(len(uv)),selection)
 for good,col,size in [(~known,'#aaaaaa',2),(known&~y,'#388cff',3),(y,'#28ff4b',14)]:
  ids=np.flatnonzero(good&mask);ax.scatter(uv[ids,0],uv[ids,1],s=size,c=col,linewidths=0,alpha=.9)
 ax.set_title(title,fontsize=9);axes_off(ax)
for split,rec,root in selected:
 key=f"{rec['event']}_query{rec['query']}";a=np.load(root/(key+'.npz'));raw=np.load(R/rec['source_file']);uv=a['query_pixel'].astype(int)
 with h5py.File(protocol[rec['episode']]['rgbd'],'r') as f:
  g=f['agentview'];rgb=g['rgb'][rec['query']];dep=g['depth_m'][rec['query']];ref_rgb=g['rgb'][rec['reference_query']]
 for name,arr in [('query_rgb',rgb),('query_depth',dep),('reference_rgb',ref_rgb)]:
  slices[f"{key}:{name}"]=hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()
 assert np.array_equal(rgb[uv[:,1],uv[:,0]],a['query_rgb'])
 fg=np.zeros_like(rgb);fg[uv[:,1],uv[:,0]]=rgb[uv[:,1],uv[:,0]]
 fig,ax=plt.subplots(1,3,figsize=(12,4));ax[0].imshow(rgb);ax[0].set_title('Complete current RGB')
 vals=dep[np.isfinite(dep)&(dep>0)];lo,hi=np.quantile(vals,[.02,.98])
 ax[1].imshow(dep,cmap='viridis',vmin=lo,vmax=hi);ax[1].set_title('Current depth (m)')
 ax[2].imshow(fg);ax[2].set_title('Complete causal foreground')
 for x in ax:axes_off(x)
 input_path=save(fig,key+'_input.png')
 fig,ax=plt.subplots(1,2,figsize=(8,4))
 overlay(ax[0],rgb,uv,a['target'],a['known'],'Automatic supervision: green + / blue - / gray unknown')
 overlay(ax[1],rgb,uv,a['target'],a['known'],f"FPS: distinct reference + {rec['positive_sources']} / - {rec['negative_sources']}",a['selected_index'])
 target_path=save(fig,key+'_target.png');pair_path=None
 if rec['query']!=rec['reference_query']:
  fig,ax=plt.subplots(1,2,figsize=(8,4));p0=raw['query_match_uv'];p1=raw['reference_match_uv'];inside=raw['inlier'].astype(bool);trackid=raw['track_ids']
  for j,(im,px) in enumerate([(rgb,p0),(ref_rgb,p1)]):
   ax[j].imshow(im)
   for z,pt in enumerate(px):
    col='#35ff57' if inside[z] else '#ff7137'
    ax[j].scatter(*pt,s=24,facecolors='none',edgecolors=col,linewidths=.8);ax[j].text(pt[0]+2,pt[1]-2,str(int(trackid[z])),color=col,fontsize=5)
   ax[j].set_title(f"{'Query' if j==0 else 'Pre-close reference'} {rec['query'] if j==0 else rec['reference_query']} | rigid inliers, NOT grasp labels",fontsize=8);axes_off(ax[j])
  pair_path=save(fig,key+'_correspondence.png')
 item={'split':split,'key':key,'event':rec['event'],'query':rec['query'],'reference_query':rec['reference_query'],'offset':rec['offset'],'positive_sources':rec['positive_sources'],'negative_sources':rec['negative_sources'],'input':input_path,'automatic_supervision':target_path,'correspondence':pair_path}
 manifest.append(item)
 links=''.join(f'<figure><a href="{html.escape(path)}"><img loading="lazy" src="{html.escape(path)}"></a><figcaption>{label}</figcaption></figure>' for path,label in [(input_path,'完整当前输入'),(target_path,'自动正负未知监督与FPS'),(pair_path,'自然角点刚体对应：不是夹持标签')] if path)
 parts.append(f"<section><h2>{split} · {html.escape(key)} · 提前{rec['offset']}帧</h2><p>不同正参考来源 {rec['positive_sources']}；负来源 {rec['negative_sources']}。同事件不是独立抓取。</p>{links}</section>")
page.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>第028轮早期查询来源</title><style>body{font:16px sans-serif;max-width:1200px;margin:32px auto;background:#f4f5f7;color:#17202a}section{background:white;padding:20px;margin:20px 0}img{max-width:100%;height:auto}figure{margin:16px 0}p{line-height:1.7}</style><h1>第028轮：早期查询自动监督与来源</h1><p>自动监督与对应证据，不是模型预测或物理接触真值。4新增训练事件+3旧验证事件，每事件固定4查询。绿色在监督图表示自动正候选，在对应图表示刚体内点，两者不可混同；蓝色为本事件可靠可见未选表面，灰色为未知。对应参考仍是夹前帧，持物来源见第026轮。未按目视修改标签。</p><p>soup5持物对应仅8/9内点；瓷杯5高敏感58帧未删除；白黄杯验证仅3参考正来源。番茄酱验证缺持物、侧向不足仍未解决。第二步未验收。</p>'+''.join(parts)+'</html>',encoding='utf-8')
dump(V/'manifest.json',{'cases':manifest,'image_count':sum(2+(x['correspondence'] is not None) for x in manifest),'not_model_prediction':True})
dump(B/'review_slices_locked.json',slices)
dump(V/'immutable_locked.json',{str(f.relative_to(R)):sha(f) for f in [page,V/'manifest.json']+list(D.glob('*.png'))})
for f,h in json.loads((B/'input_locked.json').read_text()).items():assert sha(R/f)==h
dump(B/'audits/report_audit.json',{'cases':len(manifest),'images':len(list(D.glob('*.png'))),'query_pixel_rgb_equal':True,'source_input_hashes_unchanged':True,'training_freeze_before_validation_design':True,'browser_review_pending':True})
dump(B/'stage_state.json',{'stage':'review_ready','validation_queries':12,'review_images':49})
print(json.dumps({'review_ready':True,'cases':len(manifest),'images':len(list(D.glob('*.png')))}))
