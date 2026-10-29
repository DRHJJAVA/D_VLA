from pathlib import Path
import json,numpy as np
I=Path(__file__).resolve().parents[1];E=I/'evidence';M=E/'motion'
rows=json.loads((E/'base/protocol.json').read_text())['episodes'];frames=[];excluded=[]
for row in rows:
 result=json.loads((M/(row['id']+'.json')).read_text())
 for a in result['attempts']:
  eligible=a['accepted_contact_frames'][:20]
  if eligible:
   counts=lambda t:sum(result['frame_diagnostics'][t]['visible_pad_band_points'])
   t=max(eligible,key=counts)
   if counts(t)==0:
    excluded.append(dict(id=row['id'],attempt=a['attempt'],reason='held candidate but no visible finger-band reference; no fabricated target'));continue
   accepted=True
  else:t=(a['start_frame']+a['end_frame'])//2;accepted=False
  counts=result['frame_diagnostics'][t]['visible_pad_band_points'] if accepted else [0,0]
  frames.append(dict(id=row['id'],attempt=a['attempt'],frame=t,accepted_by_motion_visual_gate=accepted,visible_points_per_side=counts,both_sides_visible=bool(min(counts)>0),user_approved=False,selection_rule='max visible narrow-band count among first20 eligible frames; training target generation only'))
(M/'review_frames.json').write_text(json.dumps(frames,indent=2));(M/'excluded_events.json').write_text(json.dumps(excluded,indent=2))
print(json.dumps(dict(frames=len(frames),held=sum(x['accepted_by_motion_visual_gate'] for x in frames),excluded=len(excluded))),flush=True)
