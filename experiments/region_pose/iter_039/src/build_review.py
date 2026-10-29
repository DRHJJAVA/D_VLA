from pathlib import Path
import json,hashlib,datetime,traceback,html
import numpy as np,h5py
from PIL import Image,ImageDraw
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/region_pose/iter_039';P=R/'experiments/region_pose/iter_038';V=R/'experiments/region_pose/reviews/iter_039';I=V/'images'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def check(p):
 x=json.loads(Path(p).read_text());bad=[k for k,v in x.items() if sha(R/k)!=v];assert not bad,(str(p),len(bad));return len(x)
def caption(im,s):
 out=Image.new('RGB',(im.width,im.height+25),'white');out.paste(im,(0,25));ImageDraw.Draw(out).text((6,6),s,fill='black');return out
def overlay(rgb,uv,old,mode):
 im=Image.fromarray(rgb.copy());dr=ImageDraw.Draw(im)
 for (u,v),n in zip(uv,old):
  if mode=='narrow' and not n: continue
  color=(0,235,255) if n else (255,80,40)
  dr.ellipse((int(u)-2,int(v)-2,int(u)+2,int(v)+2),outline=color,width=1)
 return im
def main():
 check(D/'source_locked.json');check(D/'input_locked.json');oldn=check(P/'completed_locked.json')
 cfg=json.loads((D/'config.json').read_text());z=np.load(P/'cases/tomato_sauce_demo_4.npz');rows=[];slices={};images=[];maxpx=0.;maxdepth=0.
 H=z['hand_to_world'];first=next(t for t in range(50,114) if H[t,2,3]-H[50,2,3]>.025)
 assert first==62
 assert [t for t in range(50,114) if z['full_voxels'][t]>=3]==[64,68,70,71,72]
 cams=['agentview','robot0_eye_in_hand'];rgbd=json.loads((P/'cases.json').read_text())[0]['rgbd']
 with h5py.File(rgbd,'r') as h:
  for item in cfg['selection']:
   t=item['frame'];mini=[]
   for c,cam in enumerate(cams):
    rgb=h[cam]['rgb'][t];dep=h[cam]['depth_m'][t];K=h[cam]['K'][t];T=h[cam]['camera_to_world'][t]
    for name,a in [('rgb',rgb),('depth_m',dep),('K',K),('camera_to_world',T)]: slices[f'{t}/{cam}/{name}']=ah(a)
    lo,hi=z['frame_offsets'][t:t+2];ids=np.arange(lo,hi);ids=ids[z['camera'][ids]==c];uv=z['pixel'][ids].astype(np.int64);old=z['from_narrow'][ids]
    xyz=z['points_world'][ids];source_rgb=rgb[uv[:,1],uv[:,0]];source_dep=dep[uv[:,1],uv[:,0]]
    if len(ids):
     cp=np.linalg.solve(T[:3,:3].astype(float),(xyz-T[:3,3]).T).T
     pix=cp[:,:2]/cp[:,2:]*np.array([K[0,0],K[1,1]])+np.array([K[0,2],K[1,2]])
     ep=float(np.abs(pix-uv).max());ed=float(np.abs(cp[:,2]-source_dep).max());maxpx=max(maxpx,ep);maxdepth=max(maxdepth,ed)
     assert ep<1e-6 and ed<1e-8
    row={'frame':t,'camera':cam,'roles':item['roles'],'count':len(ids),'narrow_count':int(old.sum()),'new_count':int((~old).sum()),'source_038_index':ids.tolist(),'pixel':uv.tolist(),'rgb':source_rgb.tolist(),'depth_m':source_dep.tolist(),'from_narrow':old.tolist(),'world':xyz.tolist()}
    save(D/'sources'/f'{t}_{cam}.json',row);rows.append(row)
    raw=Image.fromarray(rgb);narrow=overlay(rgb,uv,old,'narrow');full=overlay(rgb,uv,old,'full')
    gray=(np.clip(dep/2,0,1)*255).astype(np.uint8);gray[~np.isfinite(dep)|(dep<=0)]=0;depth=Image.fromarray(gray).convert('RGB')
    panels=[caption(raw,f'{t} {cam} RGB'),caption(depth,'depth 0..2m'),caption(narrow,f'narrow n={int(old.sum())} cyan'),caption(full,f'full n={len(ids)} new=orange')]
    quad=Image.new('RGB',(896,946),'white')
    for j,a in enumerate(panels): quad.paste(a,((j%2)*448,(j//2)*473))
    name=f'{t}_{cam}_full.png';quad.save(I/name);images.append(name);row['panel']=name
    mini.extend([caption(raw,f'{t} {cam} RGB'),caption(full,f'full {len(ids)} narrow {int(old.sum())}')])
    if len(ids):
     mins=uv.min(0);maxs=uv.max(0);cx,cy=(mins+maxs)/2;ww=max(96,int(maxs[0]-mins[0]+49));hh=max(96,int(maxs[1]-mins[1]+49))
     left=max(0,min(448-ww,int(cx-ww/2)));top=max(0,min(448-hh,int(cy-hh/2)));box=(left,top,min(448,left+ww),min(448,top+hh))
     a=raw.crop(box).resize(((box[2]-box[0])*3,(box[3]-box[1])*3),Image.Resampling.NEAREST);b=full.crop(box).resize(a.size,Image.Resampling.NEAREST)
     zoom=Image.new('RGB',(a.width*2,a.height+25),'white');zoom.paste(a,(0,25));zoom.paste(b,(a.width,25));ImageDraw.Draw(zoom).text((6,6),f'{t} {cam} raw / sources; crop={box}',fill='black')
     name=f'{t}_{cam}_zoom.png';zoom.save(I/name);images.append(name);row['zoom']=name;row['crop']=box
   overview=Image.new('RGB',(896,946),'white')
   for j,a in enumerate(mini): overview.paste(a,((j%2)*448,(j//2)*473))
   name=f'{t}_overview.png';overview.save(I/name);images.append(name)
 save(D/'manifest.json',rows);save(D/'slices_locked.json',slices)
 page=['<!doctype html><meta charset="utf-8"><title>第039轮 固定来源查看</title><style>body{font:16px sans-serif;max-width:1100px;margin:auto;background:#f4f5f7}img{max-width:100%;height:auto}section{background:white;padding:20px;margin:20px 0}a{margin-right:12px}</style><h1>第039轮：tomato4 固定来源查看</h1><p>仅旧验证一个事件；颜色为实际观测来源，青色原窄垫、橙色完整体积新增，非模型评分或物理接触真值。闭合前43/45/48不是持物证据。尝试50–113，首次抬升超过25mm为62。最长连续支持3帧，未达到5帧；监督分支继续停止。未修改标签，第二步未验收。</p>']
 for item in cfg['selection']:
  t=item['frame'];page.append(f'<section><h2>帧 {t} — {html.escape(" / ".join(item["roles"]))}</h2><a href="图示/iter_039/{t}_overview.png">概览 {t}</a><img src="图示/iter_039/{t}_overview.png">')
  for row in [x for x in rows if x['frame']==t]:
   page.append(f'<h3>{row["camera"]}：完整 {row["count"]} / 窄垫 {row["narrow_count"]} / 新增 {row["new_count"]}</h3><a href="图示/iter_039/{row["panel"]}">完整双行图 {t} {row["camera"]}</a>')
   if 'zoom' in row: page.append(f'<a href="图示/iter_039/{row["zoom"]}">无编号放大 {t} {row["camera"]}</a><img src="图示/iter_039/{row["zoom"]}">')
  page.append('</section>')
 pagepath=R/'docs/第039轮_旧验证tomato4固定来源查看.html';pagepath.write_text(''.join(page))
 link=R/'docs/图示/iter_039';link.symlink_to(I,target_is_directory=True)
 check(D/'source_locked.json');check(D/'input_locked.json');check(P/'completed_locked.json')
 summary={'stage':'rendered_pending_browser','episode':'tomato_sauce_demo_4','split':'old_val','frames':len(cfg['selection']),'views':len(rows),'images':len(images),'nonempty_views':sum(x['count']>0 for x in rows),'observations':sum(x['count'] for x in rows),'reprojection_max_px':maxpx,'depth_max_m':maxdepth,'old_lock_count':oldn,'slice_count':len(slices),'page':str(pagepath.relative_to(R)),'view_counts':[{'frame':x['frame'],'camera':x['camera'],'n':x['count'],'narrow':x['narrow_count']} for x in rows]}
 save(D/'summary.json',summary);save(D/'audits/completion.json',{'selection_verified':True,'source_pixel_depth_verified':True,'source_input_hashes_unchanged':True,'old038_unchanged':True,'no_new_labels':True,'no_scoring_or_training':True,'test_read':False,'summary':summary})
 save(V/'immutable_locked.json',{str(p.relative_to(R)):sha(p) for p in [*sorted(I.glob('*.png')),pagepath]})
 print(json.dumps({'stage':'rendered','frames':len(cfg['selection']),'images':len(images),'views':len(rows)}))
if __name__=='__main__':
 try: main()
 except Exception as e:
  save(D/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback_remote_only':traceback.format_exc()});print(json.dumps({'failed':True,'type':type(e).__name__}))
