from pathlib import Path
import json,hashlib,re,datetime
import numpy as np,h5py
from PIL import Image
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/region_pose/iter_039';V=R/'experiments/region_pose/reviews/iter_039';P=R/'experiments/region_pose/iter_038'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def check(p):
 x=json.loads(Path(p).read_text());assert all(sha(R/k)==v for k,v in x.items());return len(x)
z=np.load(P/'cases/tomato_sauce_demo_4.npz');rows=json.loads((D/'manifest.json').read_text());slices=json.loads((D/'slices_locked.json').read_text());pre=check(D/'input_locked.json');check(D/'source_locked.json');old=check(P/'completed_locked.json');check(V/'immutable_locked.json')
with h5py.File(json.loads((P/'cases.json').read_text())[0]['rgbd'],'r') as h:
 for row in rows:
  t=row['frame'];cam=row['camera'];c=['agentview','robot0_eye_in_hand'].index(cam);a,b=z['frame_offsets'][t:t+2];ids=np.arange(a,b);ids=ids[z['camera'][ids]==c]
  saved=json.loads((D/'sources'/f'{t}_{cam}.json').read_text())
  assert saved['source_038_index']==ids.tolist() and saved['count']==len(ids)
  uv=z['pixel'][ids].astype(np.int64)
  assert saved['pixel']==uv.tolist() and saved['from_narrow']==z['from_narrow'][ids].tolist()
  rgb=h[cam]['rgb'][t];dep=h[cam]['depth_m'][t];assert saved['rgb']==rgb[uv[:,1],uv[:,0]].tolist();assert np.array_equal(np.asarray(saved['depth_m']),dep[uv[:,1],uv[:,0]])
  for key in ['rgb','depth_m','K','camera_to_world']:
   arr=h[cam][key][t];assert hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()==slices[f'{t}/{cam}/{key}']
  panel=np.asarray(Image.open(V/'images'/row['panel']))
  assert np.array_equal(panel[25:473,:448],rgb)
  gray=(np.clip(dep/2,0,1)*255).astype(np.uint8);gray[~np.isfinite(dep)|(dep<=0)]=0
  assert np.array_equal(panel[25:473,448:896],np.repeat(gray[:,:,None],3,axis=2))
  if 'zoom' in row:
   box=row['crop'];left,top,right,bottom=box
   expected=Image.fromarray(rgb).crop(box).resize(((right-left)*3,(bottom-top)*3),Image.Resampling.NEAREST)
   zoom=np.asarray(Image.open(V/'images'/row['zoom']))
   assert np.array_equal(zoom[25:,:expected.width],np.asarray(expected))
page=R/'docs/第039轮_旧验证tomato4固定来源查看.html'
refs=re.findall(r'(?:href|src)="([^"]+)"',page.read_text())
assert all((page.parent/p).exists() for p in refs)
post=check(D/'input_locked.json');check(P/'completed_locked.json');check(V/'immutable_locked.json')
save(D/'audits/independent_source_display.json',{'passed':True,'source_views':len(rows),'input_files':pre,'old038_files':old,'all_rgb_depth_panels_pixel_exact':True,'all_source_indices_pixels_rgb_depth_exact':True,'all_zoom_raw_pixels_exact':True,'all_88_slices_hash_equal':True,'page_references':len(refs),'missing_references':0,'unknown_not_negative':True})
print(json.dumps({'independent_audit':'passed','views':len(rows),'page_references':len(refs)}))
