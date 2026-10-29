from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
R=Path('/mnt/nvme1/drh/D_VLA');W=R/'experiments/lcjp003/r2/v0_v1/v001';A=R/'docs/LCJP003_R2_V01_assets/coverage32'
lock=json.loads((W/'v1_sampling_lock.json').read_text());font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
done=[]
for start in range(0,100,5):
 cases=lock['cases'][start:start+5]
 if not all((A/f'coverage_case{ci}_view{v}_grid32.jpg').exists() for ci in range(start,start+5) for v in [0,1]):continue
 im=Image.new('RGB',(1024,1360),'#f1f5f9');d=ImageDraw.Draw(im)
 d.text((8,4),f"Cases {start}-{start+4} | {cases[0]['target']} | {cases[0]['demo']}",fill='#142f43',font=font)
 for j,name in enumerate(['external RAW','external GRID32','wrist RAW','wrist GRID32']):d.text((j*256+8,23),name,fill='#142f43',font=font)
 for row,c in enumerate(cases):
  ci=start+row;y=40+row*264
  for j,(v,kind) in enumerate([(0,'raw'),(0,'grid'),(1,'raw'),(1,'grid')]):
   name=f'coverage_case{ci}_raw{v}.png' if kind=='raw' else f'coverage_case{ci}_view{v}_grid32.jpg'
   with Image.open(A/name) as src:im.paste(src.resize((244,244)),(j*256+6,y+19))
   d.text((j*256+6,y),f"#{ci} f={c['frame']}",fill='#142f43',font=font)
 out=f'audit_{start:03d}_{start+4:03d}.png';im.save(A/out);done.append(out)
print(json.dumps({'contact_sheets':done,'cases_ready':5*len(done)}))
