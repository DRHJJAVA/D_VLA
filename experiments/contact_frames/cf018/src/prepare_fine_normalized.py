from pathlib import Path
import json,copy,hashlib,datetime,zoneinfo
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018')
SRC=R/'responses';NORM=R/'normalized_responses';OUT=R/'responses_for_fine';OUT.mkdir(exist_ok=True)
counts={'files':0,'mapped_contact':0,'uncertain_contact':0,'mapped_unknown':0}
for p in sorted(SRC.glob('*.json')):
 d=json.loads(p.read_text());n=json.loads((NORM/p.name).read_text())['normalization']
 e=copy.deepcopy(d);q=e.get('parsed')
 if not isinstance(q,dict):continue
 contacts=[];unknown=[]
 for z in n['contact']:
  if z['status']=='mapped':
   w=copy.deepcopy(z['raw']);w['start_frame']=z['start_frame'];w['end_frame']=z['end_frame'];w['evidence_frame_ids']=z['evidence_frame_ids'];contacts.append(w);counts['mapped_contact']+=1
  else:
   unknown.append({'object':str(z['raw'].get('object','unknown')),'start_frame':d['frame_ids'][0],'end_frame':d['frame_ids'][-1],'reason':'unresolved frame-unit or contradictory output'});counts['uncertain_contact']+=1
 for z in n['unknown']:
  if z['status']=='mapped':
   w=copy.deepcopy(z['raw']);w['start_frame']=z['start_frame'];w['end_frame']=z['end_frame'];unknown.append(w);counts['mapped_unknown']+=1
  else:
   unknown.append({'object':'unknown','start_frame':d['frame_ids'][0],'end_frame':d['frame_ids'][-1],'reason':'unresolved frame-unit'})
 q['contact_intervals']=contacts;q['unknown_intervals']=unknown
 e['normalization_provenance']={'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'method':'exact input-frame list, no human intervals'}
 (OUT/p.name).write_text(json.dumps(e,ensure_ascii=False,indent=2));counts['files']+=1
counts['timestamp_beijing']=datetime.datetime.now(zoneinfo.ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
(R/'fine_normalization_audit.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2))
print(json.dumps(counts,ensure_ascii=False))
