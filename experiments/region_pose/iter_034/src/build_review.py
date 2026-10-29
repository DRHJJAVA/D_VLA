from pathlib import Path
import json,hashlib,traceback,html
import numpy as np,h5py
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';R=B/'iter_034';V=B/'reviews/iter_034'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def arrhash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',10)
def title(im,text):
 out=Image.new('RGB',(im.width,im.height+30),'#101820');out.paste(im,(0,30));ImageDraw.Draw(out).text((6,6),text,font=font,fill='white');return out
def marks(rgb,uv,flags,ids,old_only=False):
 im=Image.fromarray(rgb).copy();d=ImageDraw.Draw(im)
 for (u,v),old in zip(uv,flags):
  if old_only and not old:continue
  col='#00ffff' if old else '#ff00ff'
  d.rectangle((int(u),int(v),int(u)+1,int(v)+1),fill=col)
 selected=np.arange(len(ids));selected=selected[flags] if old_only else selected
 if len(selected):
  selected=selected[::max(1,int(np.ceil(len(selected)/24)))]
  for j in selected:
   u,v=map(int,uv[j]);d.text((u+2,v+1),str(int(ids[j])),font=small,fill='white',stroke_width=1,stroke_fill='black')
 return im
def cropbox(uv,w,h):
 x0,y0=uv.min(0);x1,y1=uv.max(0)+1
 cx=(x0+x1)/2;cy=(y0+y1)/2;ww=max(96,int(x1-x0+48));hh=max(96,int(y1-y0+48))
 l=max(0,int(cx-ww/2));t=max(0,int(cy-hh/2));return [l,t,min(w,l+ww),min(h,t+hh)]
try:
 assert not V.exists();V.mkdir(parents=True);(V/'images').mkdir()
 locks=json.loads((R/'input_locked.json').read_text());assert all(sha(Path(k))==v for k,v in locks.items())
 sources=json.loads((R/'source_locked.json').read_text());assert all(sha(Path(k))==v for k,v in sources.items())
 oldlock=json.loads((B/'iter_033/completed_locked.json').read_text());assert all(sha(P/k)==v for k,v in oldlock.items())
 selection=json.loads((R/'selection.json').read_text());slices={};manifest=[];max_pixel=0.;max_depth=0.
 for row in selection:
  ep,t=row['episode'],row['frame'];z=np.load(B/'iter_033/cases'/(ep+'.npz'));lo,hi=z['frame_offsets'][t:t+2];ids=np.arange(lo,hi);H=z['hand_to_world'][t];q=z['measured_finger_q'][t]
  views=[];overview=[]
  with h5py.File(row['rgbd'],'r') as f:
   for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
    c=f[cam];rgb=c['rgb'][t];dep=c['depth_m'][t];K=c['K'][t];T=c['camera_to_world'][t]
    assert rgb.dtype==np.uint8 and rgb.shape[:2]==dep.shape and rgb.shape[2]==3
    ix=ids[z['camera'][ids]==ci];uv=z['pixel'][ix].astype(int);points=z['points_world'][ix];flags=z['from_narrow'][ix]
    assert np.all((uv[:,0]>=0)&(uv[:,0]<dep.shape[1])&(uv[:,1]>=0)&(uv[:,1]<dep.shape[0]))
    if len(ix):
     cp=np.linalg.solve(T[:3,:3].astype(float),(points-T[:3,3]).T).T;proj=cp@K.T;proj=proj[:,:2]/proj[:,2:]
     pe=float(abs(proj-uv).max());de=float(abs(cp[:,2]-dep[uv[:,1],uv[:,0]]).max());max_pixel=max(max_pixel,pe);max_depth=max(max_depth,de);assert pe<1e-6 and de<1e-8
     hand=(points-H[:3,3])@H[:3,:3];assert np.all((abs(hand[:,0])<=.008)&(hand[:,2]>=.0854)&(hand[:,2]<=.1014)&(hand[:,1]>=q[1]+.0005)&(hand[:,1]<=q[0]-.0005))
    for name,a in [('rgb',rgb),('depth_m',dep),('K',K),('camera_to_world',T)]:slices[ep+'/'+str(t)+'/'+cam+'/'+name]=arrhash(a)
    raw=Image.fromarray(rgb);depth=(colormaps['viridis'](np.clip(np.nan_to_num(dep,nan=0)/2,0,1))[:,:,:3]*255).astype(np.uint8);depth[~np.isfinite(dep)|(dep<=0)]=0
    narrow=marks(rgb,uv,flags,ix,True);full=marks(rgb,uv,flags,ix)
    tiles=[title(raw,'RGB - complete frame'),title(Image.fromarray(depth),'Depth 0-2m (black invalid)'),title(narrow,'CYAN: original narrow source'),title(full,'CYAN old / MAGENTA added source')]
    w,h=raw.size;canvas=Image.new('RGB',(w*2,h*2+120),'#101820');dr=ImageDraw.Draw(canvas);dr.text((8,6),ep+' frame '+str(t)+' '+cam,font=font,fill='white');dr.text((8,28),'SOURCE ONLY | not prediction/contact truth | pixels '+str(len(ix))+' old '+str(int(flags.sum())),font=font,fill='white')
    for tile,(x,y) in zip(tiles,[(0,60),(w,60),(0,h+90),(w,h+90)]):canvas.paste(tile,(x,y))
    stem=ep+'_'+str(t)+'_'+cam;main='images/'+stem+'.png';canvas.save(V/main)
    box=cropbox(uv,w,h) if len(ix) else None;zoom=None
    if box:
     # Nearest-neighbor preserves source pixels; this is display enlargement only.
     crop=full.crop(box);crop.thumbnail((640,560));scale=min(640/((box[2]-box[0])),560/((box[3]-box[1])))
     crop=full.crop(box).resize((int((box[2]-box[0])*scale),int((box[3]-box[1])*scale)),Image.Resampling.NEAREST)
     zoom='images/'+stem+'_zoom.png';title(crop,'SOURCE ZOOM | '+str(len(ix))+' pixels | IDs in JSON').save(V/zoom)
    # Deterministic overview adds raw RGB and both source overlays for both cameras.
    overview.extend([title(raw,cam+' RGB'),title(narrow,'old narrow '+str(int(flags.sum()))),title(full,'all source '+str(len(ix)))])
    data={'episode':ep,'frame':t,'camera':cam,'source_indices':ix.tolist(),'pixel_uv':uv.tolist(),'from_narrow':flags.tolist(),'rgb':rgb[uv[:,1],uv[:,0]].tolist(),'depth_m':dep[uv[:,1],uv[:,0]].tolist(),'crop_box':box,'image':main,'zoom':zoom}
    save(R/'sources'/(stem+'.json'),data);views.append({'camera':cam,'points':len(ix),'old_points':int(flags.sum()),'new_points':int((~flags).sum()),'image':main,'zoom':zoom})
   sheet=Image.new('RGB',(w*3,(h+30)*2),'#101820')
   for j,im in enumerate(overview):sheet.paste(im,((j%3)*w,(j//3)*(h+30)))
   over='images/'+ep+'_'+str(t)+'_overview.png';sheet.save(V/over)
  manifest.append({**row,'views':views,'overview':over,'joint_full_voxels':int(z['full_voxels'][t]),'joint_narrow_voxels':int(z['narrow_voxels'][t])})
 # Reconstruct selection independently from all six attempts and fixed supplemental frames.
 expect={}
 for a in json.loads((R/'attempts.json').read_text()):
  ep=a['episode'];z=np.load(B/'iter_033/cases'/(ep+'.npz'));s,e=a['start'],a['end'];qual=[i for i in range(s,e+1) if float(z['hand_to_world'][i,2,3]-z['hand_to_world'][s,2,3])>.025]
  first=qual[0] if qual else None;assert first==a['first_lift']
  expect.setdefault(ep,set()).update([s,e]+([first] if first is not None else []))
 expect['white_yellow_mug_demo_6'].update([55,68,82]);expect['tomato_sauce_demo_3'].update([125,136,148])
 assert {(e,t) for e,ts in expect.items() for t in ts}=={(r['episode'],r['frame']) for r in selection}
 save(R/'manifest.json',manifest);save(R/'input_slices_locked.json',slices)
 intro='<h1>第034轮：训练有限体积来源查看</h1><p>5训练episode、6尝试、23固定帧。全部颜色仅表示自动观测来源：青色原窄垫，品红全体积新增。不是模型预测、接触真值或成功率。空候选保留。未来帧只在监督侧诊断，不进入评分查询。</p><p>全体积可能混邻物；遮挡、低分辨率和密集编号不能确认逐点身份。本轮无新标签/训练/验证测试访问，第二步未验收。</p>'
 parts=[intro,'<nav>']
 for r in manifest:parts.append('<a href="#'+r['episode']+'_'+str(r['frame'])+'">'+r['episode']+' / '+str(r['frame'])+'</a> ')
 parts.append('</nav>')
 for r in manifest:
  key=r['episode']+'_'+str(r['frame']);parts.append('<section id="'+key+'"><h2>'+key+'</h2><p>'+html.escape(', '.join(r['reasons']))+' | 两相机合计1mm格：窄 '+str(r['joint_narrow_voxels'])+' / 全 '+str(r['joint_full_voxels'])+'</p>')
  parts.append('<a href="../experiments/region_pose/reviews/iter_034/'+r['overview']+'">双相机概览</a>')
  for v in r['views']:
   base='../experiments/region_pose/reviews/iter_034/';parts.append('<h3>'+v['camera']+'：旧 '+str(v['old_points'])+'，新增 '+str(v['new_points'])+' 个实际源像素</h3><a href="'+base+v['image']+'"><img loading="lazy" src="'+base+v['image']+'" alt="'+key+' '+v['camera']+' 完整RGB深度与来源"></a>')
   if v['zoom']:parts.append('<details><summary>固定局部放大（不代替完整图）</summary><img loading="lazy" src="'+base+v['zoom']+'" alt="固定来源放大"></details>')
  parts.append('</section>')
 page='<!doctype html><meta charset="utf-8"><title>034 训练有限体积来源查看</title><style>body{font-family:system-ui;max-width:1150px;margin:30px auto;background:#f4f5f6;color:#18232d;padding:20px}section{background:white;padding:18px;margin:22px 0}img{max-width:100%}nav a{display:inline-block;margin:5px}p{line-height:1.7}</style>'+''.join(parts)
 pagepath=P/'docs/第034轮_训练有限体积来源查看.html';assert not pagepath.exists();pagepath.write_text(page,encoding='utf-8')
 save(V/'manifest.json',manifest)
 assert all(sha(Path(k))==v for k,v in locks.items()) and all(sha(Path(k))==v for k,v in sources.items())
 assert all(sha(P/k)==v for k,v in oldlock.items())
 images=list((V/'images').glob('*.png'))
 save(R/'audits/completion_audit.json',{'inputs_unchanged':len(locks),'sources_unchanged':len(sources),'old033_lock_unchanged':len(oldlock),'selected_frames':len(selection),'main_images':len(selection)*2,'overview_images':len(selection),'zoom_images':sum(v['zoom'] is not None for r in manifest for v in r['views']),'total_images':len(images),'slices':len(slices),'max_pixel_error':max_pixel,'max_depth_error_m':max_depth,'selection_independent_equal':True,'source_pixel_rgb_depth_equal':True,'validation_read':False,'test_read':False,'new_labels':False})
 save(V/'immutable_locked.json',{str(p.relative_to(P)):sha(p) for p in [*images,V/'manifest.json',pagepath]})
 save(R/'stage_state.json',{'stage':'review_ready'})
 print(json.dumps({'review_ready':True,'frames':len(selection),'images':len(images)}))
except Exception as e:
 save(R/'failure.json',{'type':type(e).__name__,'message':str(e)});(R/'traceback_remote.log').write_text(traceback.format_exc());print(json.dumps({'failed':True,'type':type(e).__name__,'message':str(e)}));raise SystemExit(1)
