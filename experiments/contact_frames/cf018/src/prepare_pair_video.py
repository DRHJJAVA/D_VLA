from pathlib import Path
from PIL import Image
import json,hashlib,time
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018')
for source,target,summary in [('prepared','prepared_pair','pair_summary.json'),('prepared_fine','prepared_pair_fine','pair_summary_fine.json')]:
 rows=[]
 for c in json.loads((R/'episode_manifest.json').read_text())['cases']:
  name=c['case'];n=c['frames']
  src=R/source/name;dest=R/target/name;dest.mkdir(parents=True,exist_ok=True)
  ids=list(range(n)) if source=='prepared_fine' else sorted(set(range(0,n,2))|{n-1})
  hashes={}
  for t in ids:
   a=Image.open(src/f'{t:04d}_agentview.jpg').convert('RGB')
   b=Image.open(src/f'{t:04d}_robot0_eye_in_hand.jpg').convert('RGB')
   assert a.size==(448,448) and b.size==(448,448)
   pair=Image.new('RGB',(896,448))
   pair.paste(a,(0,0));pair.paste(b,(448,0))
   p=dest/f'{t:04d}.jpg'
   pair.save(p,quality=92,subsampling=0)
   hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
  (dest/'images_sha256.json').write_text(json.dumps(hashes,indent=2))
  rows.append({'case':name,'frames':len(ids),'images':len(hashes)})
 (R/summary).write_text(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S%z'),'source':source,'target':target,'cases':rows,'layout':'896x448 left agentview right wrist'},ensure_ascii=False,indent=2))
print(json.dumps({'status':'paired','sampled_images':sum(x['images'] for x in json.loads((R/'pair_summary.json').read_text())['cases']),'fine_images':sum(x['images'] for x in json.loads((R/'pair_summary_fine.json').read_text())['cases'])}))
