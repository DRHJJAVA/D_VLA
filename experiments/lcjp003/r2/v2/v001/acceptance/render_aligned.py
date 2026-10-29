import pathlib,sys,json,hashlib,math,html,datetime,zoneinfo
import numpy as np,torch,h5py
from PIL import Image
r=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r2/v2/v001')
root=pathlib.Path('/mnt/nvme1/drh/D_VLA')
docs=root/'docs'
assets=docs/'LCJP003_R2_V2_model_outputs'
assets.mkdir(exist_ok=True)
sys.path.insert(0,str(r/'src'))
import network
records=json.loads((r/'acceptance/per_window.json').read_text())
episodes={x['id']:x for x in json.loads((r/'episodes.json').read_text())}
metrics=json.loads((r/'acceptance/metrics.json').read_text())
ck=r/'checkpoints/last.pt'
assert hashlib.sha256(ck.read_bytes()).hexdigest()==metrics['checkpoint_sha256']
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
model=network.R2Front().cuda().eval()
model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['front'])
names={'alphabet_soup':'字母汤罐头','bbq_sauce':'烧烤酱','butter':'黄油','chocolate_pudding':'巧克力布丁','cream_cheese':'奶油奶酪','ketchup':'番茄酱','milk':'牛奶','orange_juice':'橙汁','salad_dressing':'沙拉酱','tomato_sauce':'番茄调味酱'}
def chinese(task):
 for k,v in names.items():
  if 'pick_up_the_'+k+'_and_' in task:return '拿起'+v+'并放进篮子'
 return task
palette=[(65,125,239),(40,171,105),(248,202,61),(247,141,58),(225,60,64)]
def color_abs(s):
 return palette[min(4,max(0,int(float(s)*5)))]
def pair(v):
 return Image.fromarray(np.concatenate([v[0],v[1]],axis=1)).resize((768,384),Image.Resampling.NEAREST)
def tint(rgb,m,color,alpha=.72):
 out=rgb.copy()
 out[m]=np.clip(out[m].astype(float)*(1-alpha)+np.asarray(color)*alpha,0,255).astype(np.uint8)
 return out
cards=[];all_match=True;unmatched=0;scored_cells=0;scored_elsewhere=0
for i,a in enumerate(records):
 q=torch.load(r/'cache'/a['episode']/(str(a['frame'])+'.pt'),map_location='cpu',weights_only=False)
 with torch.no_grad():
  out=model.encode(*(q[k].cuda() for k in ('x','valid','h','words','priors')))
 pi=out['pi'].detach().cpu().numpy()
 scores=out['scores'].detach().cpu().numpy()
 k=int(np.argmax(pi[:-1]))
 cid=int(q['candidate_ids'][k])
 if cid!=int(a['top_id']):all_match=False
 assert cid==int(a['top_id']),(i,cid,a['top_id'])
 view=int(q['views'][k])
 with h5py.File(r/'cache'/a['episode']/('observations_'+str(a['frame'])+'.h5'),'r') as f:
  g=f['frames'][str(a['frame'])]
  maps=[{row.tobytes():idx for idx,row in enumerate(g['features_'+str(j)][:].reshape(-1,384))} for j in (0,1)]
  obj=g['object_'+str(cid)]
  maskname='mask_'+str(view)
  assert maskname in obj
  mask=np.asarray(Image.fromarray(np.unpackbits(obj[maskname][:]).reshape(256,256)*255).resize((128,128),Image.Resampling.NEAREST))>0
 e=episodes[a['episode']]
 with h5py.File(root/'experiments/contact_joint_object/cache/rgbd'/(a['episode']+'.h5'),'r') as f:
  inds=np.flatnonzero(f['frame_indices'][:]==a['frame']); assert len(inds)==1
  rgb=[np.asarray(Image.fromarray(f[c+'/rgb'][int(inds[0])]).resize((128,128),Image.Resampling.BILINEAR)) for c in ['agentview','robot0_eye_in_hand']]
 x=q['x'][k,:,:,:384].numpy().astype(np.float16)
 valid=q['valid'][k].numpy().astype(bool)
 sums=np.zeros(1024,np.float64);count=np.zeros(1024,np.int32)
 for patch in range(32):
  for vec in x[patch][valid[patch]]:
   gid=maps[view].get(vec.tobytes())
   if gid is None:unmatched+=1;continue
   sums[gid]+=float(scores[k,patch]);count[gid]+=1
 vals=np.divide(sums,np.maximum(count,1))
 selected=np.where(count>0)[0]
 scored_cells+=len(selected)
 modelmask=[v.copy() for v in rgb]
 modelmask[view]=tint(modelmask[view],mask,(228,48,62))
 absv=[v.copy() for v in rgb]
 relv=[v.copy() for v in rgb]
 eligible=[]
 for gid in selected:
  yy=(gid//32)*4;xx=(gid%32)*4
  if mask[yy:yy+4,xx:xx+4].any():eligible.append(gid)
 for gid in eligible:
  yy=(gid//32)*4;xx=(gid%32)*4
  region=mask[yy:yy+4,xx:xx+4]
  absv[view][yy:yy+4,xx:xx+4]=tint(absv[view][yy:yy+4,xx:xx+4],region,color_abs(vals[gid]),.88)
 order=sorted(eligible,key=lambda gid:(-vals[gid],gid))
 keep=int(math.ceil(.8*len(order)))
 for rank,gid in enumerate(order[:keep]):
  frac=1-rank/max(1,keep)
  band=4 if frac>.95 else 3 if frac>.90 else 2 if frac>.80 else 1 if frac>.60 else 0
  yy=(gid//32)*4;xx=(gid%32)*4
  region=mask[yy:yy+4,xx:xx+4]
  relv[view][yy:yy+4,xx:xx+4]=tint(relv[view][yy:yy+4,xx:xx+4],region,palette[band],.88)
 pair(rgb).save(assets/f'{i:02d}_raw.jpg',quality=93)
 pair(modelmask).save(assets/f'{i:02d}_mask.jpg',quality=93)
 pair(absv).save(assets/f'{i:02d}_absolute.jpg',quality=93)
 pair(relv).save(assets/f'{i:02d}_relative.jpg',quality=93)
 valid_scores=scores[k][q['valid'][k].any(dim=1).numpy()]
 other=sum(int(q['valid'][j].any()) for j in range(len(q['candidate_ids'])) if j!=k)
 scored_elsewhere+=other
 title=chinese(e['task'])
 cards.append('<article><h2>'+str(i+1)+'. '+html.escape(title)+'</h2><p>开发集 '+html.escape(e['episode'])+' · 帧 '+str(a['frame'])+' · 模型选中候选 #'+str(cid)+'（'+('外部' if view==0 else '腕部')+'相机） · 选择概率 '+f'{pi[k]:.3f}'+'；另有 '+str(other)+' 个候选也被评分</p><p>该候选有效 patch 分数：最小 '+f'{valid_scores.min():.4f}'+'，最大 '+f'{valid_scores.max():.4f}'+'，标准差 '+f'{valid_scores.std():.4f}'+'。显示 '+str(len(eligible))+' 个实际映射到 RGB-D 区域的特征网格。</p><h3>原始双视角</h3><img src="LCJP003_R2_V2_model_outputs/'+f'{i:02d}_raw.jpg'+'"><h3>模型选中的候选分割（红色）</h3><img src="LCJP003_R2_V2_model_outputs/'+f'{i:02d}_mask.jpg'+'"><h3>该候选的绝对评分（固定 0–1 色阶）</h3><img src="LCJP003_R2_V2_model_outputs/'+f'{i:02d}_absolute.jpg'+'"><h3>该候选评分最高 80% 的相对排名</h3><img src="LCJP003_R2_V2_model_outputs/'+f'{i:02d}_relative.jpg'+'"></article>')
style='body{font-family:Arial,sans-serif;background:#10151d;color:#f2f2f2;margin:24px;max-width:850px}article{background:#1b2532;padding:18px;margin:22px 0;border-radius:12px}img{width:768px;max-width:100%;display:block}p{color:#c5d3e1;line-height:1.6}h3{color:#a9cbef;margin:15px 0 7px}.warn{background:#533037;padding:16px;border-radius:10px}'
intro='<meta charset="utf-8"><title>LCJP003 R2 V2 训练后验收</title><style>'+style+'</style><h1>LCJP003 R2 V2 训练后验收</h1><p class="warn"><b>更正：</b>旧页把候选物体编号当成列表下标，40 例中 39 例红色掩膜对应错误。本页重新从已训练模型计算首选候选，并逐例断言与验收记录相同。旧图不作为目标识别证据。</p><p>任务名称已译为中文。以下“模型选中”不代表它真的选对；目标正确率仍需独立标注。每例只显示网络首选候选的分割与评分，不人工替它选目标。</p><p>当前评分头实际对每个候选都计算 32 个 patch 分数；图上的色块只画网络真实采样点映射到的特征网格，不补画未评分的区域。绝对色阶始终为蓝 0–0.2、绿 0.2–0.4、黄 0.4–0.6、橙 0.6–0.8、红 0.8–1；相对图先取最高 80% 网格，再在保留部分内排名：红前 5%、橙 5–10%、黄 10–20%、绿 20–40%、蓝其余。相对颜色不表示绝对高分。</p><p>开发集动作 MAE '+f'{metrics["baseline_mae"]:.4f}'+'；评分改均匀/打乱后原始评分优势分别仅 '+f'{metrics["ablation"]["uniform"]["normal_improvement_percent"]:.3f}%'+' / '+f'{metrics["ablation"]["shuffled"]["normal_improvement_percent"]:.3f}%'+'，未达到预登记的 V4 5% 动作引导门槛。</p>'
page=docs/'LCJP003_R2_V2_训练后验收.html'
page.write_text(intro+''.join(cards))
audit={'time_beijing':datetime.datetime.now(zoneinfo.ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),'cases':len(records),'model_id_match':all_match,'score_grid_unmatched_points':unmatched,'selected_scored_grid_cells':scored_cells,'other_candidate_scored_count_total':scored_elsewhere,'images':len(list(assets.glob('*.jpg'))),'page':str(page.relative_to(root))}
(r/'acceptance/model_visualization_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps(audit,ensure_ascii=False))