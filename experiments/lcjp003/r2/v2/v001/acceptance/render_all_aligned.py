import pathlib,sys,json,hashlib,datetime,zoneinfo
import numpy as np,torch,h5py
from PIL import Image
r=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r2/v2/v001');root=pathlib.Path('/mnt/nvme1/drh/D_VLA');docs=root/'docs';assets=docs/'LCJP003_R2_V2_model_outputs'
sys.path.insert(0,str(r/'src'));import network
records=json.loads((r/'acceptance/per_window.json').read_text());episodes={x['id']:x for x in json.loads((r/'episodes.json').read_text())}
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
m=network.R2Front().cuda().eval();m.load_state_dict(torch.load(r/'checkpoints/last.pt',map_location='cpu',weights_only=False)['front'])
palette=np.array([(65,125,239),(40,171,105),(248,202,61),(247,141,58),(225,60,64)],dtype=np.float32)
page=docs/'LCJP003_R2_V2_训练后验收.html';html=page.read_text();all_match=0;all_cells=0
for i,a in enumerate(records):
 q=torch.load(r/'cache'/a['episode']/(str(a['frame'])+'.pt'),map_location='cpu',weights_only=False)
 with torch.no_grad():o=m.encode(*(q[k].cuda() for k in ('x','valid','h','words','priors')))
 pi=o['pi'].detach().cpu().numpy();scores=o['scores'].detach().cpu().numpy();top=int(np.argmax(pi[:-1]));assert int(q['candidate_ids'][top])==int(a['top_id'])
 e=episodes[a['episode']]
 with h5py.File(root/'experiments/contact_joint_object/cache/rgbd'/(a['episode']+'.h5'),'r') as f:
  inds=np.flatnonzero(f['frame_indices'][:]==a['frame']); assert len(inds)==1
  rgb=[np.asarray(Image.fromarray(f[c+'/rgb'][int(inds[0])]).resize((128,128),Image.Resampling.BILINEAR)) for c in ['agentview','robot0_eye_in_hand']]
 accum=np.zeros((2,128,128),np.float64);count=np.zeros((2,128,128),np.int32)
 with h5py.File(r/'cache'/a['episode']/('observations_'+str(a['frame'])+'.h5'),'r') as f:
  g=f['frames'][str(a['frame'])]
  maps=[{row.tobytes():idx for idx,row in enumerate(g['features_'+str(j)][:].reshape(-1,384))} for j in (0,1)]
  for k,cid in enumerate(q['candidate_ids']):
   view=int(q['views'][k]);obj=g['object_'+str(cid)];mk='mask_'+str(view)
   if mk not in obj:continue
   mask=np.asarray(Image.fromarray(np.unpackbits(obj[mk][:]).reshape(256,256)*255).resize((128,128),Image.Resampling.NEAREST))>0
   x=q['x'][k,:,:,:384].numpy().astype(np.float16);valid=q['valid'][k].numpy().astype(bool)
   sums=np.zeros(1024,np.float64);n=np.zeros(1024,np.int32)
   for patch in range(32):
    for vec in x[patch][valid[patch]]:
     gid=maps[view].get(vec.tobytes())
     assert gid is not None
     sums[gid]+=float(scores[k,patch]);n[gid]+=1
   for gid in np.where(n>0)[0]:
    yy=(gid//32)*4;xx=(gid%32)*4;region=mask[yy:yy+4,xx:xx+4]
    if not region.any():continue
    suba=accum[view,yy:yy+4,xx:xx+4];subn=count[view,yy:yy+4,xx:xx+4]
    suba[region]+=sums[gid]/n[gid];subn[region]+=1;all_cells+=1
 views=[]
 for j in (0,1):
  img=rgb[j].copy();valid=count[j]>0;avg=np.zeros((128,128),np.float32);avg[valid]=(accum[j][valid]/count[j][valid]).astype(np.float32)
  band=np.clip((avg*5).astype(int),0,4);col=palette[band]
  img[valid]=np.clip(img[valid].astype(float)*.12+col[valid]*.88,0,255).astype(np.uint8)
  views.append(img)
 combined=Image.fromarray(np.concatenate(views,axis=1)).resize((768,384),Image.Resampling.NEAREST);fn=f'{i:02d}_all_absolute.jpg';combined.save(assets/fn,quality=93)
 needle='<h3>模型选中的候选分割（红色）</h3><img src="LCJP003_R2_V2_model_outputs/'+f'{i:02d}_mask.jpg'+'">'
 ins='<h3>全部候选的原始评分（包括未选中的物体）</h3><img src="LCJP003_R2_V2_model_outputs/'+fn+'">'+needle
 assert needle in html;html=html.replace(needle,ins,1);all_match+=1
html=html.replace('每例只显示网络首选候选的分割与评分，不人工替它选目标。','每例同时显示全部候选的原始评分，以及网络首选候选的分割与评分；不人工替它选目标。全部候选图的重叠色块只为显示而取均值，不是模型的融合公式。')
page.write_text(html)
audit=r/'acceptance/model_visualization_audit.json';d=json.loads(audit.read_text());d.update({'all_candidates_score_images':all_match,'all_candidates_scored_grid_cells':all_cells,'images':len(list(assets.glob('*.jpg')))});audit.write_text(json.dumps(d,ensure_ascii=False,indent=2))
print(json.dumps({'cases':all_match,'scored_grid_cells_all_candidates':all_cells,'images':d['images'],'page_updated':True}))