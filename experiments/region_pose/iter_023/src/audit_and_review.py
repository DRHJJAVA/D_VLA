from pathlib import Path
import ast,json,hashlib,time,html
import numpy as np,cv2,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';O=B/'iter_023'
V=O/'val';REVIEW=B/'reviews/iter_023';IMG=P/'docs/图示/iter_023'
REVIEW.mkdir(parents=True,exist_ok=False);IMG.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def lockcheck(path,root):
 d=json.loads(path.read_text())
 for k,h in d.items():
  p=Path(k);p=p if p.is_absolute() else root/p
  assert sha(p)==h,str(p)
 return len(d)
checks={n:lockcheck(O/n,P) for n in ['source_locked.json','source_review_addendum.json','input_locked.json','seed_locked.json']}
checks['022_completed']=lockcheck(B/'iter_022/completed_locked.json',P)
checks['011_val']=lockcheck(B/'iter_011/val/locked.json',B/'iter_011/val')
new=json.loads((V/'summary.json').read_text())['cases'];old={c['id']:c for c in json.loads((B/'iter_011/val/summary.json').read_text())['cases']}
tree=ast.parse((B/'iter_011/src/extend_cotracker.py').read_text())
aug=next(n for n in ast.walk(tree) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='endpoint')
idsnode=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='ids' for x in n.targets))
filtercode=compile(ast.fix_missing_locations(ast.Module(body=[aug,idsnode],type_ignores=[])),'original_filter','exec')
rows=[];filterrows=[];quality_names=['rigid','apply','estimate','patchcorr']
a=ast.parse((O/'src/extend_cotracker_grid.py').read_text());b=ast.parse((B/'iter_022/src/extend_cotracker_grid_v3.py').read_text())
for name in quality_names:assert ast.dump(next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name))==ast.dump(next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==name))
for c in new:
 ident=c['id'];oc=old[ident];q=c['query'];stem=c['event']+'_query'+str(q)
 assert c['query']==oc['query'] and c['selected_held_frames']==oc['selected_held_frames']
 seed=np.load(O/'seeds'/(ident+'.npz'));tr=np.load(V/'trajectories'/(stem+'.npz'));co=np.load(V/'correspondences'/(stem+'.npz'));ot=np.load(B/'iter_011/val/trajectories'/(stem+'.npz'))
 assert np.array_equal(seed['initial_uv'],tr['initial_uv']);nold=len(seed['original_uv']);assert np.array_equal(seed['original_uv'],ot['initial_uv'])
 v,u=np.where(seed['mask']>0);pts=list(seed['original_uv'].copy())
 for cy,cx in sorted(set(zip((v//8).tolist(),(u//8).tolist()))):
  if len(pts)>=300:break
  ix=np.where((v//8==cy)&(u//8==cx))[0];ds=(u[ix]-cx*8-3.5)**2+(v[ix]-cy*8-3.5)**2
  k=ix[np.lexsort((u[ix],v[ix],ds))[0]];p=np.array([u[k],v[k]],np.float32)
  if len(pts) and np.min(np.linalg.norm(np.asarray(pts)-p,axis=1))<3:continue
  pts.append(p)
 assert np.array_equal(np.asarray(pts,np.float32),seed['initial_uv'])
 assert np.array_equal(tr['frames'],ot['frames'])
 vis=tr['visible'][:,:nold]&ot['visible'];disp=np.linalg.norm(tr['tracks'][:,:nold]-ot['tracks'],axis=-1)[vis]
 endpoints=[]
 for br in c['bridge_results']:
  f=int(br['frame']);ids=co['track_id_'+str(f)];inl=co['inlier_'+str(f)].astype(bool)
  fm=np.load(V/'filter_masks'/(ident+'_frame'+str(f)+'.npz'));e={'np':np,'alive':fm['corr__alive'],'endpoint':fm['corr__endpoint'].copy(),'corr':fm['corr__corr']}
  before=e['endpoint'].copy();exec(filtercode,e);assert np.array_equal(e['ids'],ids)
  filterrows.append({'episode':ident,'frame':f,'model_depth_robot_joint_ok':int(fm['ok__ok'].sum()),'alive':len(e['alive']),'joint_endpoint_rejected':int((~before).sum()),'appearance_after_endpoint_rejected':int((before&~e['endpoint']).sum()),'appearance_rejected_all_alive':int((fm['corr__corr']<.65).sum()),'matches':len(ids),'independent_depth_rejections':None,'independent_robot_rejections':None,'ids_exact':True})
  ep={k:br.get(k) for k in ['frame','valid','reason','matches','inliers','ratio','median_inlier_mm','second_axis_scale_mm']}
  fitted=br.get('reason')!='fewer_than_8_endpoint_matches'
  ep.update(original_matches=int((ids<nold).sum()),grid_matches=int((ids>=nold).sum()),original_inliers=int(((ids<nold)&inl).sum()) if fitted else None,grid_inliers=int(((ids>=nold)&inl).sum()) if fitted else None)
  endpoints.append(ep)
 rows.append({'episode':ident,'query':q,'old_early_valid':oc['valid'],'new_early_valid':c['valid'],'initial_original':nold,'initial_grid':len(seed['initial_uv'])-nold,'dual_bridge_difference':c.get('dual_bridge_difference'),'time_comparisons':c.get('time_comparisons'),'endpoints':endpoints,'old_prefix_prediction_change_px':{'observations':int(len(disp)),'median':float(np.median(disp)),'p95':float(np.percentile(disp,95)),'max':float(disp.max())}})
save(O/'comparison_summary.json',{'cases':rows,'old_candidates':sum(c['valid'] for c in old.values()),'new_candidates':sum(c['valid'] for c in new),'not_grasp_success_rate':True,'tomato4_missing_held_support':True})
save(O/'filter_attribution.json',{'rows':filterrows,'stage_semantics':'Only current ok__ok and corr__alive/endpoint/corr used. Replay original in-place appearance mask before ids. Serial counts overlap with raw appearance rejection; do not sum as independent causes.','separate_robot_depth_unavailable':True})
# Source review: fixed seven training cases plus all three validation cases.
wanted=['alphabet_soup_demo_5','alphabet_soup_demo_7','porcelain_mug_demo_5','white_yellow_mug_demo_4','white_yellow_mug_demo_5','porcelain_mug_demo_1','white_yellow_mug_demo_3']
train={c['id']:c for c in json.loads((B/'iter_022/train/summary.json').read_text())['cases']}
selected=[(B/'iter_022','train',train[i]) for i in wanted]+[(O,'val',c) for c in new]
refs={x['id']:x for x in json.loads((B/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
inputpaths=[]
for root,split,c in selected:
 stem=c['event']+'_query'+str(c['query'])
 inputpaths.extend([root/'seeds'/(c['id']+'.npz'),root/split/'trajectories'/(stem+'.npz'),root/split/'correspondences'/(stem+'.npz'),Path(refs[c['id']]['rgbd'])])
save(REVIEW/'input_locked.json',{str(p):sha(p) for p in dict.fromkeys(inputpaths)})
manifest=[];sections=[]
def outfig(fig,name):
 path=IMG/name;fig.savefig(path,dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig);return '图示/iter_023/'+name
def overlay(ax,uv,source,ids,ins,visible=None):
 known=np.zeros(len(uv),bool);known[ids]=True;good=np.zeros(len(uv),bool);good[ids[ins]]=True
 for grid,marker in [(False,'o'),(True,'^')]:
  for state,color in [('unmatched','#999999'),('outlier','#e65a2f'),('inlier','#00b86b')]:
   m=(source==grid)&({'unmatched':~known,'outlier':known&~good,'inlier':good}[state])
   if visible is not None:m &= visible
   ax.scatter(uv[m,0],uv[m,1],s=35,marker=marker,facecolors='none',edgecolors=color,linewidths=1.0)
 for j in ids:
  ax.text(uv[j,0]+1,uv[j,1],str(int(j)),fontsize=5.3,color='#005c35' if good[j] else '#9e2706',bbox={'facecolor':'white','alpha':.5,'pad':.1,'edgecolor':'none'})
for root,split,c in selected:
 ident=c['id'];q=c['query'];stem=c['event']+'_query'+str(q)
 seed=np.load(root/'seeds'/(ident+'.npz'));tr=np.load(root/split/'trajectories'/(stem+'.npz'));co=np.load(root/split/'correspondences'/(stem+'.npz'))
 origin=seed['source_is_grid'].astype(bool);uv=seed['initial_uv'];records=[]
 with h5py.File(refs[ident]['rgbd'],'r') as h:
  rgb=h['agentview/rgb'][q]
  fig,ax=plt.subplots(1,2,figsize=(10,5))
  for a in ax:a.imshow(rgb);a.axis('off')
  ax[0].set_title('Full current RGB / query '+str(q))
  for m,marker,col in [(~origin,'o','#008da8'),(origin,'^','#b820bf')]:ax[1].scatter(uv[m,0],uv[m,1],s=30,marker=marker,facecolors='none',edgecolors=col,linewidths=1)
  ax[1].set_title('Original circles / Grid triangles')
  fig.suptitle(ident+' | '+split)
  overview=outfig(fig,ident+'_input.png')
  for f in c['selected_held_frames']:
   f=int(f);rgb1=h['agentview/rgb'][f];ti=int(np.where(tr['frames']==f)[0][0]);ids=co['track_id_'+str(f)].astype(int);ins=co['inlier_'+str(f)].astype(bool)
   br=next(b for b in c['bridge_results'] if int(b['frame'])==f)
   fig,ax=plt.subplots(1,2,figsize=(12,6))
   for a,img in zip(ax,[rgb,rgb1]):a.imshow(img);a.axis('off')
   overlay(ax[0],uv,origin,ids,ins)
   overlay(ax[1],tr['tracks'][ti],origin,ids,ins,tr['visible'][ti])
   ax[0].set_title('Query '+str(q)+' / original full image')
   ax[1].set_title('Held endpoint '+str(f)+' / fixed source')
   fig.suptitle(ident+' | endpoint valid='+str(br.get('valid'))+' | matches='+str(len(ids))+' | inliers='+str(br.get('inliers','not fitted'))+'\nCircle=original; Triangle=grid; Green=inlier; Orange=outlier; Gray=not matched\nIDs denote source tracks only, not grasp labels.')
   link=outfig(fig,ident+'_pair_'+str(f)+'.png')
   records.append({'frame':f,'image':link,'valid':br.get('valid'),'matches':len(ids),'inliers':br.get('inliers'),'reason':br.get('reason')})
  # Save exact image-slice hashes for review provenance.
  slices={str(t):hashlib.sha256(np.ascontiguousarray(h['agentview/rgb'][t]).tobytes()).hexdigest() for t in [q]+c['selected_held_frames']}
 manifest.append({'episode':ident,'split':split,'query':q,'overview':overview,'endpoints':records,'rgb_slice_sha256':slices,'candidate':c['valid'],'source_root':str(root)})
 weak='<p class="warn">弱支持：soup5最早端点仅8/9内点，新增网格内点1/2个。候选不能直接当可靠标签。</p>' if ident=='alphabet_soup_demo_5' else ''
 sec='<section id="'+ident+'"><h2>'+ident+' · '+split+' · query '+str(q)+'</h2>'+weak+'<p>早期对应候选：'+str(c['valid'])+'；展示全部预定端点。圆形为原角点，三角为网格点；绿色为刚体内点，橙色为外点，灰色为未进入匹配。编号只表示追踪来源。未拟合的全false占位不代表实测零内点。</p><img loading="lazy" src="'+overview+'">'
 for r in records:sec+='<h3>固定端点 '+str(r['frame'])+' · valid='+str(r['valid'])+'</h3><p>'+html.escape(str({k:v for k,v in r.items() if k!='image'}))+'</p><a href="'+r['image']+'"><img loading="lazy" src="'+r['image']+'"></a>'
 sections.append(sec+'</section>')
save(REVIEW/'manifest.json',manifest)
table='<table><tr><th>旧验证事件</th><th>原→新候选</th><th>原+网格</th><th>双端点整查询云中位/p95 mm</th><th>共同内点</th></tr>'
for r in rows:
 d=r['dual_bridge_difference'];table+='<tr><td>'+r['episode']+'</td><td>'+str(r['old_early_valid'])+' → '+str(r['new_early_valid'])+'</td><td>'+str(r['initial_original'])+' + '+str(r['initial_grid'])+'</td><td>'+format(d['median_mm'],'.3f')+' / '+format(d['p95_mm'],'.3f')+'</td><td>'+str(d['common_observed_inliers'])+'</td></tr>'
table+='</table>'
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>第023轮 固定旧验证与来源查看</title><style>body{font-family:system-ui,sans-serif;max-width:1280px;margin:auto;padding:28px;background:#f5f6f8;color:#1d2934;line-height:1.7}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bac5cd;padding:8px}.warn{color:#8c3800;font-weight:600}a{color:#006c88}h1,h2{line-height:1.3}</style><h1>第023轮：固定旧验证与来源查看</h1><p>相同8px网格、3px间距、300上限及冻结CoTracker/质量规则；022训练摘要冻结后一次执行原3旧验证事件。这里是实际追踪对应的来源图，不是抓取区域目标或网络评分。没有新增标签或评分训练，不代表抓取成功率、真实表面身份或阶段通过。</p><p>五个新增训练候选、两个失败训练案例、三个旧验证事件及全部固定端点均保留。杯柄对应不表示杯柄被抓。补点改变联合上下文，原角点预测可变化。番茄酱验证仍缺持物支持；白黄杯对应候选不代表已有充分稳定区域监督。</p>'''+table+'<nav>'+''.join('<a href="#'+m['episode']+'">'+m['episode']+'</a> · ' for m in manifest)+'</nav>'+''.join(sections)+'</html>'
pagepath=P/'docs/第023轮_固定旧验证与来源查看.html';pagepath.write_text(page,encoding='utf8')
assert len(manifest)==10 and sum(len(x['endpoints']) for x in manifest)==46
for p,h in json.loads((REVIEW/'input_locked.json').read_text()).items():assert sha(p)==h
for x in manifest:
 for rel in [x['overview']]+[z['image'] for z in x['endpoints']]:assert (P/'docs'/rel).is_file()
save(O/'audits/report_audit.json',{'passed':True,'cases':10,'endpoint_pairs':46,'image_files':len(list(IMG.glob('*.png'))),'all_fixed_endpoints':True,'full_image_no_crops':True,'no_network_prediction_claim':True,'time':time.time()})
save(O/'audits/completion_audit.json',{'passed':True,'source_input_counts':checks,'validation_cases':3,'fixed_endpoint_checks':15,'reconstructed_filter_ids':len(filterrows),'seed_independent_reconstruction':3,'quality_functions_unchanged':quality_names,'mask_and_original_prefix_checked_before_forward':True,'seed_hash_lock_before_forward':True,'no_new_target_or_training':True,'no_test_access':True,'time':time.time()})
files=list(REVIEW.glob('*.json'))+list(IMG.glob('*.png'))+[pagepath]
save(REVIEW/'immutable_locked.json',{str(p.relative_to(P)):sha(p) for p in files})
print(json.dumps({'audit_passed':True,'val_candidates':sum(c['valid'] for c in new),'cases':10,'pairs':46,'images':len(list(IMG.glob('*.png')))}))
