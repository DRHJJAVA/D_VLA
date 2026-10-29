from pathlib import Path
import json,hashlib,time,sys,traceback,html
import numpy as np
import h5py
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/region_pose';B=P/'iter_026'
D=R/'docs';I=D/'图示/iter_026';V=P/'reviews/iter_026'
sys.path.insert(0,str(P/'iter_025/src'))
from build_targets import apply,inside
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def verify(p):
 for k,v in json.loads(p.read_text()).items():assert sha(R/k)==v,k
def title(ax,s):ax.set_title(s,fontproperties=font,fontsize=10)
def photo(ax,im,s):
 ax.imshow(im);title(ax,s);ax.set_axis_off()
def save(fig,name):
 fig.tight_layout();p=I/name;fig.savefig(p,dpi=130);plt.close(fig);return str(p.relative_to(D))
def overlay(ax,p,c,size=6,marker='o'):
 if len(p):ax.scatter(p[:,0],p[:,1],s=size,c=c,marker=marker,linewidths=.3)
def table(head,rows):
 return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
def img(path,caption):return '<figure><a href="'+path+'"><img loading="lazy" src="'+path+'"></a><figcaption>'+html.escape(caption)+'</figcaption></figure>'
def run():
 verify(B/'source_locked.json');verify(B/'input_locked.json')
 I.mkdir(parents=True,exist_ok=False);V.mkdir(parents=True,exist_ok=False);(B/'cases').mkdir()
 protocol={r['id']:r for r in json.loads((P/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
 cases=json.loads((B/'config.json').read_text())['cases'];allrows=[];sections=[];images=[];slicehash={}
 for row in cases:
  split=row['split'];ep=row['episode'];q=row['query'];key=row['event']+'_query'+str(q)
  z=np.load(P/'iter_024'/split/(key+'.npz'));y=np.load(P/'iter_025/targets'/split/(key+'.npz'));sample=np.load(P/'iter_025/sampling'/split/(key+'.npz'))
  previous=P/('iter_022' if split=='train' else 'iter_023')
  co=np.load(previous/split/'correspondences'/(key+'.npz'));seed=np.load(previous/'seeds'/(ep+'.npz'))
  query=z['query_xyz'];pixels=z['query_pixel'].astype(int);pos=y['target'].astype(bool);known=y['known'].astype(bool);neg=known&~pos;unknown=~known
  posids=np.flatnonzero(pos);sp=sample['selected_index'];sp=sp[pos[sp]]
  assert np.array_equal(query,y['query_xyz'])
  perframe=[];links=[];source_records=[];proof={}
  with h5py.File(protocol[ep]['rgbd'],'r') as f:
   def read(camera,t,kind):
    a=f[camera+'/'+kind][t];slicehash[ep+'/'+camera+'/'+kind+'/'+str(t)]=hashlib.sha256(a.tobytes()).hexdigest();return a
   qim=read('agentview',q,'rgb');qd=read('agentview',q,'depth_m')
   assert np.array_equal(qim[pixels[:,1],pixels[:,0]],z['query_rgb'])
   fig,ax=plt.subplots(1,3,figsize=(13,4))
   photo(ax[0],qim,'完整当前RGB 查询'+str(q))
   im=ax[1].imshow(np.where(np.isfinite(qd)&(qd>0),qd,np.nan),cmap='viridis');title(ax[1],'完整当前深度（米）');ax[1].set_axis_off();fig.colorbar(im,ax=ax[1],fraction=.046)
   masked=np.zeros_like(qim);masked[pixels[:,1],pixels[:,0]]=qim[pixels[:,1],pixels[:,0]];photo(ax[2],masked,'因果前景输入，不按目标采样')
   name=save(fig,key+'_input.png');images.append(name);links.append(img(name,'完整当前输入。黑色为前景以外，不代表未知标签。'))
   fig,ax=plt.subplots(1,3,figsize=(13,4))
   for a in ax:photo(a,qim,'')
   for a in ax[:2]:
    overlay(a,pixels[unknown],'#b5b5b5',5);overlay(a,pixels[neg],'#3884ff',5);overlay(a,pixels[pos],'#00ed69',18)
   title(ax[0],'完整视图：绿正/蓝已知负/灰未知')
   if len(pixels):
    lo=pixels.min(0)-12;hi=pixels.max(0)+12;ax[1].set_xlim(lo[0],hi[0]);ax[1].set_ylim(hi[1],lo[1])
   title(ax[1],'前景放大（自动监督，不是预测）')
   overlay(ax[2],pixels[sp],'#00ed69',24)
   for ident in posids:ax[2].annotate(str(ident),pixels[ident],fontsize=6,color='yellow')
   title(ax[2],'黄色编号=完整正来源；绿点=FPS保留正点')
   if len(pixels):
    ax[2].set_xlim(lo[0],hi[0]);ax[2].set_ylim(hi[1],lo[1])
   name=save(fig,key+'_targets.png');images.append(name);links.append(img(name,'目标覆盖与FPS正来源；没有评分模型参与。'))
   for j,t in enumerate(z['selected_frames'].tolist()):
    Q=z['query_to_current_hand_'+str(t)]
    loo=np.load(P/'iter_024'/(split+'_loo')/(key+'_frame'+str(t)+'.npz'))
    positions=np.stack([apply(query,T) for T in [Q,*loo['loo_Q']]])
    spread=np.linalg.norm(positions-positions[0],axis=2).max(0);finger=z['finger_q_'+str(t)]
    robust=inside(positions,finger,.001).all(0);may=inside(positions,finger,.004).any(0);stable=(spread<=.002)&(len(loo['loo_Q'])>=7)
    ids=np.flatnonzero((z['frame']==t)&z['supported_candidate'])
    if len(ids):dist,near=cKDTree(z['mapped_query_xyz'][ids]).query(query);nearest=ids[near]
    else:dist=np.full(len(query),np.inf);nearest=np.full(len(query),-1)
    actual=(dist<=.003)&robust&stable
    assert np.array_equal(actual,y['positive_frames'][j])
    assert np.array_equal(z['query_visible_at_frames'][j]&~may&stable,y['negative_frames'][j])
    active=posids[actual[posids]]
    proof['distance_'+str(t)]=dist;proof['nearest_source_'+str(t)]=nearest;proof['robust_inside_'+str(t)]=robust;proof['may_inside_'+str(t)]=may;proof['stable_'+str(t)]=stable
    for ident in posids:
     n=int(nearest[ident])
     source_records.append(dict(query_source_id=int(ident),frame=t,positive_support=bool(actual[ident]),visible=bool(z['query_visible_at_frames'][j,ident]),robust_inside=bool(robust[ident]),stable=bool(stable[ident]),spread_mm=float(spread[ident]*1000),distance_mm=float(dist[ident]*1000) if np.isfinite(dist[ident]) else None,region_source_id=n if n>=0 else None,source_camera=int(z['camera'][n]) if n>=0 else None,source_pixel=z['pixel'][n].tolist() if n>=0 else None))
    held=[read(c,t,'rgb') for c in ['agentview','robot0_eye_in_hand']]
    for cam in [0,1]:
     m=(z['frame']==t)&(z['camera']==cam);uv=z['pixel'][m].astype(int)
     assert np.array_equal(held[cam][uv[:,1],uv[:,0]],z['rgb'][m])
    fig,ax=plt.subplots(1,3,figsize=(14,4.5))
    photo(ax[0],qim,'查询正点：绿=本帧支持，橙=本帧不支持')
    overlay(ax[0],pixels[posids],'orange',18);overlay(ax[0],pixels[active],'#00ed69',22)
    for ident in posids:ax[0].annotate(str(ident),pixels[ident],fontsize=5,color='yellow')
    for cam in [0,1]:
     photo(ax[cam+1],held[cam],('主相机' if cam==0 else '腕相机')+' 实际有限区域源点 '+str(t))
     allsrc=(z['frame']==t)&(z['camera']==cam);supported=allsrc&z['supported_candidate']
     overlay(ax[cam+1],z['pixel'][allsrc],'#f080ff',6);overlay(ax[cam+1],z['pixel'][supported],'#39e2ff',10)
     for ident in active:
      n=nearest[ident]
      if z['camera'][n]==cam:overlay(ax[cam+1],z['pixel'][n:n+1],'yellow',22);ax[cam+1].annotate(str(ident),z['pixel'][n],fontsize=5,color='yellow')
    name=save(fig,key+'_support'+str(t)+'.png');images.append(name);links.append(img(name,'紫=真实有限体积观测；青=跨时支持候选；黄编号=查询正点最近来源，仅邻近支持，非逐点身份真值。'))
    track=co['track_id_'+str(t)];inlier=co['inlier_'+str(t)].astype(bool);grid=seed['source_is_grid'][track]
    fig,ax=plt.subplots(1,2,figsize=(9,4.5))
    for a,im,uv,label in [(ax[0],qim,co['query_uv_'+str(t)],'查询刚体对应'),(ax[1],held[0],co['bridge_uv_'+str(t)],'持物刚体对应 '+str(t))]:
     photo(a,im,label)
     for g,marker in [(False,'o'),(True,'^')]:
      for good,color in [(False,'orange'),(True,'#00ef66')]:
       m=(grid==g)&(inlier==good);overlay(a,uv[m],color,15,marker)
     for k in np.flatnonzero(inlier):a.annotate(str(track[k]),uv[k],fontsize=5,color='yellow')
    name=save(fig,key+'_rigid'+str(t)+'.png');images.append(name);links.append(img(name,'圆=原角点，三角=网格；绿=刚体内点、橙=外点。刚体对应不是夹持标签，未匹配点不在此图。'))
    perframe.append(dict(frame=t,all_positive_frame_points=int(actual.sum()),final_positive_supported=int(len(active)),inliers=int(inlier.sum()),original_inliers=int((inlier&~grid).sum()),grid_inliers=int((inlier&grid).sum()),loo_fits=len(loo['loo_Q']),raw_region_points=int((z['frame']==t).sum()),supported_region_points=int(((z['frame']==t)&z['supported_candidate']).sum()),final_positive_stable=int(stable[posids].sum())))
  records=[]
  for ident in posids:
   rr=[r for r in source_records if r['query_source_id']==ident]
   supported=[r['frame'] for r in rr if r['positive_support']]
   assert len(supported)>=2
   records.append(dict(query_source_id=int(ident),pixel=pixels[ident].tolist(),fps_selected=bool(ident in sp),support_frames=supported,confidence=float(y['confidence'][ident])))
  np.savez_compressed(B/'cases'/(key+'.npz'),positive_source_ids=posids,**proof)
  report=dict(episode=ep,split=split,query=q,frames=perframe,positive_sources=records,source_edges=source_records,original_targets_unchanged=True,not_identity_or_contact_truth=True)
  dump(B/'cases'/(key+'.json'),report);allrows.append(report)
  warning='弱8/9内点；仅60/62，不跨失败帧恢复。' if 'soup_demo_5' in ep else '58帧高敏感完整保留，不删帧。' if ep=='porcelain_mug_demo_5' else '0正点，保持不足，不当全负训练。' if ep=='white_yellow_mug_demo_5' else '正来源稀少，数量门槛不等于充分证据。' if 'white_yellow' in ep else '重复帧属于同一事件。'
  frame_table=table(['帧','最终正点获支持数','原/grid内点','LOO','区域观测/跨时支持'],[[r['frame'],r['final_positive_supported'],str(r['original_inliers'])+'/'+str(r['grid_inliers']),r['loo_fits'],str(r['raw_region_points'])+'/'+str(r['supported_region_points'])] for r in perframe])
  point_table=table(['查询源ID','像素xy','FPS保留','支持时刻','权重'],[[r['query_source_id'],r['pixel'],r['fps_selected'],r['support_frames'],round(r['confidence'],4)] for r in records]) if records else '<p>没有正候选，完整输入及未知仍展示。</p>'
  sections.append('<section><h2>'+html.escape(split+' '+ep)+' 查询'+str(q)+'</h2><p class="warn">'+warning+'</p>'+frame_table+'<details><summary>逐正来源支持时刻</summary>'+point_table+'</details>'+''.join(links)+'</section>')
 page=D/'第026轮_自动监督来源与逐时刻支持.html'
 page.write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>026 自动监督来源</title><style>body{font:16px system-ui;max-width:1500px;margin:24px auto;padding:20px;background:#f7f8fb;color:#20252d}section{background:white;padding:20px;margin:24px 0}img{width:100%;height:auto}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:7px}figure{margin:24px 0}.warn{background:#fff1cc;padding:15px}figcaption{padding:10px;color:#414955}</style><h1>026 自动监督来源与逐时刻支持</h1><p class="warn">这是025冻结自动监督的来源审计，不是模型预测、人工真值或物理接触精度。绿/蓝/灰仅表示本事件自动正/已知负/未知；未知不当负例。未来持物图像只用于监督侧来源展示。第二步尚未用户验收。</p><p>6个固定案例；所有024已选持物端点保留。圆/三角刚体对应图独立于有限夹持区域图。最近区域源点仅提供3mm邻近支持，不能当作逐点真实身份。点击图片可查看完整尺寸。</p>'+''.join(sections)+'</html>',encoding='utf8')
 verify(B/'source_locked.json');verify(B/'input_locked.json')
 for n in ['024','025']:verify(P/('iter_'+n)/'completed_locked.json')
 dump(B/'input_slices_locked.json',slicehash)
 dump(B/'summary.json',dict(cases=[dict(episode=r['episode'],split=r['split'],query=r['query'],frames=r['frames'],positive_sources=len(r['positive_sources'])) for r in allrows],images=len(images),selected_frames=sum(len(r['frames']) for r in allrows),no_label_or_model_changes=True))
 manifest=dict(page=str(page.relative_to(R)),images=images,cases=[r['episode'] for r in allrows])
 dump(V/'manifest.json',manifest)
 for im in images:assert (D/im).exists()
 dump(B/'audits/completion_audit.json',dict(time=time.time(),all_6_input_pixel_rgb_checks=True,all_selected_frames_reconstructed=True,all_positive_points_two_support_frames=True,source_pixel_rgb_checks=True,original_024_025_hashes_unchanged=True,source_and_input_hashes_unchanged=True,frames=sum(len(r['frames']) for r in allrows),images=len(images),not_model_prediction=True))
 dump(V/'immutable_locked.json',{str(p.relative_to(R)):sha(p) for p in [page,V/'manifest.json',*(D/x for x in images)]})
 dump(B/'stage_state.json',dict(stage='report_ready',time=time.time(),browser_review_pending=True))
 print('REPORT_READY',len(images),len(allrows))
if __name__=='__main__':
 try:run()
 except Exception as e:
  dump(B/'failure.json',dict(error_type=type(e).__name__,message=str(e),time=time.time()))
  with (B/'failure.log').open('w') as f:traceback.print_exc(file=f)
  print('FAILED',type(e).__name__);sys.exit(1)
