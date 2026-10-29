from pathlib import Path
import json,hashlib
import numpy as np,h5py
from PIL import Image,ImageDraw,ImageFont
P=Path('/mnt/nvme1/drh/D_VLA');R=P/'experiments/region_pose/iter_034';V=P/'experiments/region_pose/reviews/iter_034'
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
outputs=[];slices=json.loads((R/'input_slices_locked.json').read_text())
for row in json.loads((R/'manifest.json').read_text()):
 for v in row['views']:
  if not v['zoom']:continue
  ep,t,cam=row['episode'],row['frame'],v['camera'];j=json.loads((R/'sources'/(ep+'_'+str(t)+'_'+cam+'.json')).read_text())
  with h5py.File(row['rgbd'],'r') as f:rgb=f[cam]['rgb'][t]
  assert hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()==slices[ep+'/'+str(t)+'/'+cam+'/rgb']
  raw=Image.fromarray(rgb);mark=raw.copy();draw=ImageDraw.Draw(mark)
  for (u,vv),old in zip(j['pixel_uv'],j['from_narrow']):draw.point((u,vv),fill='#00ffff' if old else '#ff00ff')
  box=j['crop_box'];left=raw.crop(box);right=mark.crop(box);ww=box[2]-box[0];hh=box[3]-box[1];scale=min(480/ww,480/hh);size=(int(ww*scale),int(hh*scale))
  left=left.resize(size,Image.Resampling.NEAREST);right=right.resize(size,Image.Resampling.NEAREST)
  out=Image.new('RGB',(size[0]*2,size[1]+55),'#101820');out.paste(left,(0,55));out.paste(right,(size[0],55));d=ImageDraw.Draw(out);d.text((5,5),ep+' / '+str(t)+' / '+cam,font=font,fill='white');d.text((5,28),'RAW RGB                         SOURCE PIXELS ONLY - NO LABELS',font=font,fill='white')
  rel='images/'+ep+'_'+str(t)+'_'+cam+'_clean.png';out.save(V/rel);outputs.append({'episode':ep,'frame':t,'camera':cam,'file':rel,'source_count':len(j['source_indices'])})
page=P/'docs/第034轮_来源无编号放大.html'
page.write_text('<!doctype html><meta charset="utf-8"><h1>034 固定来源无编号放大</h1><p>原始来源、裁剪范围与像素不变。左原始RGB，右仅源像素颜色。青色窄垫、品红新增；不是物理接触真值，不因目视修改标签。</p>'+''.join('<h2>'+a['episode']+' '+str(a['frame'])+' '+a['camera']+'</h2><a href="图示/iter_034/'+Path(a['file']).name+'"><img style="max-width:100%" src="图示/iter_034/'+Path(a['file']).name+'"></a>' for a in outputs),encoding='utf-8')
(R/'audits/clean_zoom.json').write_text(json.dumps(outputs,indent=2))
(V/'clean_zoom_locked.json').write_text(json.dumps({str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [page,*[V/a['file'] for a in outputs]]},indent=2))
print(json.dumps({'clean_images':len(outputs)}))
