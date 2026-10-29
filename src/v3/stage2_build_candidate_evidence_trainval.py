"""Read-only candidate evidence viewer. No event or grasp probability claims."""
from pathlib import Path
import json, hashlib, html
import numpy as np, cv2,h5py
R=Path('/mnt/nvme1/drh/D_VLA')
SRC=R/'audits/v3/comoving_occluder_pilot_v1'
OUT=R/'audits/v3/candidate_evidence_review_trainval_v1'
VIS=R/'docs/图示/全部候选来源_v1'
PAGE=R/'docs/第二步_全部训练验证候选来源.html'
def runs(m):
 return list(zip(np.flatnonzero(m&~np.r_[False,m[:-1]]).tolist(),np.flatnonzero(m&~np.r_[m[1:],False]).tolist()))
def jpg(p,im):
 if p.exists():raise RuntimeError(str(p))
 assert cv2.imwrite(str(p),im,[cv2.IMWRITE_JPEG_QUALITY,88])
def crop(im,b):
 x0,y0,x1,y1=b
 sub=im[y0:y1,x0:x1];scale=320/max(sub.shape[:2]);h=max(1,round(sub.shape[0]*scale));w=max(1,round(sub.shape[1]*scale))
 canvas=np.full((320,320,3),225,np.uint8);canvas[(320-h)//2:(320-h)//2+h,(320-w)//2:(320-w)//2+w]=cv2.resize(sub,(w,h),interpolation=cv2.INTER_NEAREST)
 return canvas
def main():
 cv2.setNumThreads(3)
 assert not OUT.exists() and not VIS.exists() and not PAGE.exists()
 OUT.mkdir(parents=True);VIS.mkdir(parents=True)
 protocol=json.loads((R/'audits/v3/surface_klt_rgbd_pilot_v1/protocol.json').read_text())
 summaries=json.loads((SRC/'summary.json').read_text())
 extension=json.loads((R/'audits/v3/surface_klt_rgbd_train_extension_v1/protocol.json').read_text())
 extension_summary=json.loads((R/'audits/v3/comoving_occluder_train_extension_v1/summary.json').read_text())
 source={x['id']:SRC for x in protocol['episodes']}
 pose_source={x['id']:R/'audits/v3/surface_klt_rgbd_pilot_v1' for x in protocol['episodes']}
 source.update({x['id']:R/'audits/v3/comoving_occluder_train_extension_v1' for x in extension['episodes']})
 pose_source.update({x['id']:R/'audits/v3/surface_klt_rgbd_train_extension_v1' for x in extension['episodes']})
 protocol['episodes']+=extension['episodes']
 protocol['episodes'].sort(key=lambda x:(x['split'],x['id']))
 summaries['episodes']+=extension_summary['episodes']
 assert len(protocol['episodes'])==24 and len(source)==24
 byid={x['id']:x for x in summaries['episodes']}
 cards=[];results=[]
 for item in protocol['episodes']:
  ident=item['id'];summary=byid[ident];folder=VIS/ident;folder.mkdir()
  with np.load(source[ident]/(ident+'.npz')) as z:
   pixels=z['first_pixel'].astype(int);p=z['persistent_hypothesis'];q=z['observed_occluder_object_frame'];active=z['active']
   union=p.any((0,1));point_frame=p.any(1);parts=runs(point_frame.any(1))
  n=item['frames'];span_votes=np.zeros(len(pixels),np.uint16)
  partrows=[]
  for i,(a,b) in enumerate(parts):
   mask=point_frame[a:b+1].any(0);span_votes+=mask.astype(np.uint16)
   partrows.append({'fragment':i+1,'start_frame':a,'end_frame':b,'frames':b-a+1,'unique_associated_surface_points':int(mask.sum()),'independent_grasp_event':None})
  sample_frames=set()
  for a,b in parts:sample_frames.update(range(max(0,a-4),min(n,b+5)))
  if not parts:sample_frames.update(np.linspace(0,n-1,9,dtype=int).tolist())
  frames=sorted(sample_frames)
  with np.load(pose_source[ident]/(ident+'.npz')) as z:poses=z['pose']
  with h5py.File(item['rgbd'],'r') as f:
   cam=f['agentview'];rgb=cv2.cvtColor(cam['rgb'][0],cv2.COLOR_RGB2BGR);dep=cam['depth_m'][0]
   mask=np.zeros(dep.shape,bool);mask[pixels[:,1],pixels[:,0]]=True
   im=np.full_like(rgb,225);im[mask]=rgb[mask];mark=im.copy()
   for u,v in pixels[union]:cv2.circle(mark,(int(u),int(v)),1,(0,215,255),-1)
   depths=dep[mask];lo,hi=np.percentile(depths,[2,98])
   gray=np.clip((dep-lo)/max(hi-lo,1e-6)*220+20,0,255).astype(np.uint8)
   dim=np.full_like(rgb,225);dim[mask]=np.repeat(gray[:,:,None],3,2)[mask]
   x0,y0=pixels.min(0);x1,y1=pixels.max(0)
   pad=12;box=(max(0,x0-pad),max(0,y0-pad),min(rgb.shape[1],x1+pad+1),min(rgb.shape[0],y1+pad+1))
   jpg(folder/'object.jpg',np.concatenate([crop(im,box),crop(dim,box),crop(mark,box)],axis=1))
   for t in frames:
    raw=[];marked=[]
    for ci,name in enumerate(['agentview','robot0_eye_in_hand']):
     c=f[name];im=cv2.cvtColor(c['rgb'][t],cv2.COLOR_RGB2BGR);ann=im.copy()
     idx=np.flatnonzero(p[t,ci])
     if len(idx):
      qw=q[t,ci,idx]@poses[t,:3,:3].T+poses[t,:3,3]
      T=c['camera_to_world'][t];K=c['K'][t];cc=(qw-T[:3,3])@T[:3,:3]
      uv=np.rint(np.stack([K[0,0]*cc[:,0]/cc[:,2]+K[0,2],K[1,1]*cc[:,1]/cc[:,2]+K[1,2]],1)).astype(int)
      for u,v in uv:
       if 0<=u<im.shape[1] and 0<=v<im.shape[0]:cv2.circle(ann,(int(u),int(v)),2,(0,215,255),-1)
     raw.append(cv2.resize(im,(448,448)));marked.append(cv2.resize(ann,(448,448)))
    jpg(folder/(str(t)+'_raw.jpg'),np.concatenate(raw,1))
    jpg(folder/(str(t)+'_marked.jpg'),np.concatenate(marked,1))
  np.savez_compressed(OUT/(ident+'.npz'),first_pixel=pixels,episode_candidate_support=union.astype(np.uint8),continuous_fragment_support=span_votes)
  result=dict(id=ident,split=item['split'],total_frames=n,candidate_frames=int(point_frame.any(1).sum()),candidate_fragments=partrows,
    frames_displayed=frames,episode_has_candidate=bool(union.any()),independent_grasp_event_count=None,
    observable_event_count=None,not_observable_event_count=None,unknown_reason='Independent interaction boundaries and point observability not established; fragments are not separate grasps.',
    unique_associated_surface_points=int(union.sum()),initial_surface_points=len(pixels))
  results.append(result)
  url='图示/全部候选来源_v1/'+ident+'/'
  interval='；'.join(str(a)+'–'+str(b) for a,b in parts) or '未检测到候选；不代表没有抓取'
  card=f'<section id="{ident}" data-frames="{html.escape(json.dumps(frames))}" data-base="{url}"><h2>{ident} <small>{item["split"]}</small></h2>'
  card+=f'<p>连续候选片段：{interval}。共{result["candidate_frames"]}个候选帧，{int(union.sum())}个关联表面点。<b>独立抓取次数：待确定。</b></p>'
  card+='<div class="labels"><span>首帧单物体RGB</span><span>首帧物体深度（灰度）</span><span>首帧表面关联候选（黄色）</span></div>'
  card+=f'<img loading="lazy" src="{url}object.jpg" alt="单物体及候选关联点"><p class="note">上排黄色表示该示范中至少出现过一次候选关联；不是频率高低，也不是模型评分。首帧看不到的内壁不能靠这张图完整呈现，请看下方腕相机画面。</p>'
  card+='<details open><summary>查看候选来源与前后画面（示范参考，不是评分模型输入）</summary><div class="controls"><button class="prev">上一帧</button><input class="time" type="range" min="0" max="'+str(len(frames)-1)+'" value="0"><button class="next">下一帧</button><label><input class="overlay" type="checkbox">显示黄色候选</label><output></output></div>'
  card+='<div class="labels two"><span>主相机原始示范</span><span>腕相机原始示范</span></div>'
  card+=f'<img class="sequence" loading="lazy" src="{url}{frames[0]}_raw.jpg" alt="双视角来源"><p class="note">滑块显示候选及前后各4帧，按实际帧号标注。显示候选时，黄点表示当前观测到的近处共动表面，不是接触真值。内壁、杯柄等位置不因合理性判断而剔除。</p></details></section>'
  cards.append(card);print(ident,len(frames),len(parts),flush=True)
 stats={'episodes':results,'episode_total':len(results),'episodes_with_candidate':sum(x['episode_has_candidate'] for x in results),
   'continuous_fragment_total':sum(len(x['candidate_fragments']) for x in results),
   'train_episodes':sum(x['split']=='train' for x in results),'val_episodes':sum(x['split']=='val' for x in results),
   'cross_demonstration_surface_frequency':None,'independent_event_counts':None,'notice':'Candidate evidence audit only, not model prediction, grasp frequency, accuracy or stage acceptance. No new masks/thresholds/tuning.'}
 (OUT/'summary.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')
 style='body{font:17px/1.65 system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 22px;color:#182331;background:#f5f7fa}h1{font-size:30px}h2{font-size:23px}section{background:white;padding:24px;margin:24px 0;border:1px solid #d7dee5;border-radius:12px}img{width:100%;height:auto}.note,small{color:#526170;font-size:15px}.labels{display:grid;grid-template-columns:repeat(3,1fr);text-align:center;font-weight:600}.two{grid-template-columns:repeat(2,1fr)}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:16px 0}input[type=range]{flex:1;min-width:180px}button{padding:8px 16px}aside{padding:18px;background:#fff3cc;border-radius:10px}'
 js="""document.querySelectorAll('section[data-frames]').forEach(s=>{let fs=JSON.parse(s.dataset.frames),r=s.querySelector('.time'),c=s.querySelector('.overlay'),im=s.querySelector('.sequence'),o=s.querySelector('output');function draw(){let t=fs[Number(r.value)];im.src=s.dataset.base+t+(c.checked?'_marked.jpg':'_raw.jpg');o.textContent='实际帧号 '+t+'（'+(Number(r.value)+1)+'/'+fs.length+'）'}r.oninput=draw;c.onchange=draw;s.querySelector('.prev').onclick=()=>{r.value=Math.max(0,Number(r.value)-1);draw()};s.querySelector('.next').onclick=()=>{r.value=Math.min(fs.length-1,Number(r.value)+1);draw()};draw()});"""
 page='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>第二步：抓取候选与来源画面</title><style>'+style+'</style><body><h1>抓取候选与来源画面</h1><p>保留杯内壁、杯口、杯柄、杯身等候选，合理性由你判断。</p><aside><b>这是候选证据统计，不是训练后模型预测。</b>24段（训练20、旧验证4）已全部列出。候选覆盖与片段数量见下表，连续片段可能来自同一次操作，不能当作独立抓取次数。当前尚未得到跨示范同一位置的独立抓取频率，第二步未验收。</aside><p>训练与验证仅分开展示，不混合统计训练频率。相同表面点在一段示范中出现多帧，上排也只标记一次。无法观测与未抓取不混为一类。</p>'+''.join(cards)+'<script>'+js+'</script></body></html>'
 table='<table><tr><th>划分</th><th>总示范</th><th>有候选示范</th><th>连续候选片段</th><th>无候选示范</th></tr>'
 for split in ['train','val']:
  es=[e for e in results if e['split']==split]
  positives=sum(e['episode_has_candidate'] for e in es)
  table+=f'<tr><td>{split}</td><td>{len(es)}</td><td>{positives}</td><td>{sum(len(e["candidate_fragments"]) for e in es)}</td><td>{len(es)-positives}</td></tr>'
 table+='</table><p>以上是候选检出情况，不是模型准确率，也不是区域抓取频率。</p><nav>'+ ' / '.join('<a href="#'+e['id']+'">'+e['id']+'</a>' for e in results)+'</nav>'
 page=page.replace('</aside>','</aside>'+table,1).replace('section{','table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #ccd}nav{font-size:14px}section{',1)
 PAGE.write_text(page,encoding='utf-8')
 files=sorted(x for x in OUT.iterdir() if x.is_file())+[PAGE,R/'src/v3/stage2_build_candidate_evidence_trainval.py']+sorted(VIS.rglob('*.jpg'))
 (OUT/'locked.json').write_text(json.dumps({'files':{str(x.relative_to(R)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}},indent=2),encoding='utf-8')
 print('complete',len(results),'episodes',stats['continuous_fragment_total'],'fragments',flush=True)
if __name__=='__main__':main()
