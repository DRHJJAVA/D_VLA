import runpy
runpy.run_path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018/src/normalize_and_review_v2.py')
runpy.run_path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018/src/prepare_fine_normalized_v2.py')
from pathlib import Path
import hashlib,json
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018')
cases=json.loads((R/'episode_manifest.json').read_text())['cases']
locked=json.loads((R/'predictions_locked.json').read_text())
assert locked['count']==184
for name,digest in locked['files'].items():
 p=R/'responses_for_fine'/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
out=[];invalid=0
for case in cases:
 name=case['case'];n=case['frames'];centers=set()
 for arm in ['A','B','C','D']:
  for p in sorted((R/'responses_for_fine').glob(f'{name}__{arm}__*.json')):
   x=json.loads(p.read_text())
   y=x.get('parsed')
   if not isinstance(y,dict):continue
   for kind in ['contact_intervals','unknown_intervals']:
    arr=y.get(kind,[])
    if not isinstance(arr,list):continue
    for item in arr:
     if not isinstance(item,dict):continue
     try:
      a=int(item['start_frame']);b=int(item['end_frame'])
     except Exception:
      invalid+=1;continue
     if a<0 or b<a or b>=n or a<x['frame_ids'][0] or b>x['frame_ids'][-1]:
      invalid+=1;continue
     centers.add(a);centers.add(b)
     for fld in ['start_possible_range','end_possible_range']:
      z=item.get(fld)
      if isinstance(z,list) and len(z)==2:
       for v in z:
        if isinstance(v,int) and 0<=v<n:centers.add(v)
 selected=set()
 for c in centers:
  lo=max(0,c-10);hi=min(n-1,c+10)
  start=min(max(0,lo),n-40)
  if start+39<hi:start=min(n-40,hi-39)
  selected.add(start)
 starts=sorted(selected)
 compact=[]
 for st in starts:
  if not compact or st>compact[-1]+20:compact.append(st)
  elif any(not (compact[-1]<=c-10 and c+10<=compact[-1]+39) for c in centers if st<=c<=st+39):
   compact.append(st)
 windows=[{'start':int(st),'end':int(st+39),'frame_ids':list(range(st,st+40))} for st in compact]
 out.append({'case':name,'centers':sorted(centers),'windows':windows})
result={'source_lock_sha256':hashlib.sha256((R/'predictions_locked.json').read_bytes()).hexdigest(),'cases':out,'invalid_boundary_count':invalid,'human_ranges_used':False}
(R/'fine_windows.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'cases':len(out),'total_fine_windows':sum(len(x['windows']) for x in out),'invalid_boundaries':invalid,'per_case':{x['case']:len(x['windows']) for x in out}},ensure_ascii=False))
