#!/usr/bin/env python3
"""Select reproducible contact-event cases for required manual visual review."""
import argparse
import json
import random
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--events-dir',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=30)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    episodes=[x['episode'] for x in json.loads(a.labels.read_text())['labels']]
    selected=random.Random(a.seed).sample(episodes,min(a.count,len(episodes)))
    rows=[]
    for episode in selected:
        path=a.events_dir/f'alphabet_soup_{episode}_grasp_candidates.json'
        events=json.loads(path.read_text())['events']
        event=max((x for x in events if x['passes_numeric_filter']),key=lambda x:x['duration_frames'])
        left=np.asarray([x['position_world'] for x in event['left_contacts']],dtype=float)
        right=np.asarray([x['position_world'] for x in event['right_contacts']],dtype=float)
        rows.append({'episode':episode,'event':event['grasp_event_id'],
            'contact_frame':event['contact_frame'],'end_frame':event['end_frame'],
            'left_contact_count':len(left),'right_contact_count':len(right),
            'finger_contact_center_separation_m':float(np.linalg.norm(left.mean(axis=0)-right.mean(axis=0))),
            'lift_m':event['lift_m'],'relative_drift_m':event['max_relative_drift_m'],
            'video':f'data/rendered/alphabet_soup_{episode}_full_rgbd.h5',
            'probe':f'data/events/alphabet_soup_{episode}_contact_probe.json',
            'review_status':'pending_visual_surface_and_object_check'})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'seed':a.seed,'selected':len(rows),'records':rows},indent=2)+'\n')
    print(json.dumps({'selected':len(rows),'output':str(a.output),
                      'separation_range_m':[min(r['finger_contact_center_separation_m'] for r in rows),
                                            max(r['finger_contact_center_separation_m'] for r in rows)]}),flush=True)

if __name__=='__main__':main()
