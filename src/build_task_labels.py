#!/usr/bin/env python3
"""Build provisional grasp point labels from real simulator finger/object contacts."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--events-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--libero-root',type=Path,required=True)
    a=p.parse_args()
    a.events_dir.mkdir(parents=True,exist_ok=True)
    with h5py.File(a.input) as f:
        episodes=sorted(f['data'],key=lambda name:int(name.split('_')[-1]))
        lengths={name:len(f['data'][name]['states']) for name in episodes}
    rows=[]
    for episode in episodes:
        probe=a.events_dir/f'alphabet_soup_{episode}_contact_probe.json'
        candidate=a.events_dir/f'alphabet_soup_{episode}_grasp_candidates.json'
        if not probe.exists():
            subprocess.run([sys.executable,str(Path(__file__).with_name('probe_contacts.py')),
                            '--input',str(a.input),'--output',str(probe),
                            '--libero-root',str(a.libero_root),'--episode',episode],check=True)
        if not candidate.exists():
            subprocess.run([sys.executable,str(Path(__file__).with_name('extract_grasps.py')),
                            '--probe',str(probe),'--output',str(candidate),
                            '--object-prefix','alphabet_soup_1_g',
                            '--object-body','alphabet_soup_1_main',
                            '--object-asset-id','stable_hope_objects/alphabet_soup'],check=True)
        report=json.loads(candidate.read_text())
        accepted=[event for event in report['events'] if event['passes_numeric_filter']]
        if accepted:
            event=max(accepted,key=lambda x:x['duration_frames'])
            contacts=event['left_contacts']+event['right_contacts']
            world=np.asarray([c['position_world'] for c in contacts],dtype=np.float64)
            object_points=np.asarray([c['position_object'] for c in contacts],dtype=np.float64)
            row={'episode':episode,'num_frames':lengths[episode],
                 'target_world_xyz':world.mean(axis=0).tolist(),
                 'target_object_xyz':object_points.mean(axis=0).tolist(),
                 'contact_frame':event['contact_frame'],
                 'duration_frames':event['duration_frames'],
                 'lift_m':event['lift_m'],
                 'max_relative_drift_m':event['max_relative_drift_m'],
                 'contact_count':len(contacts),
                 'quality_flag':'provisional_unreviewed'}
            rows.append(row)
            print(json.dumps({'episode':episode,'accepted':True,'contacts':len(contacts)}),flush=True)
        else:
            print(json.dumps({'episode':episode,'accepted':False,'runs':len(report['events'])}),flush=True)
        a.output.write_text(json.dumps({'source':str(a.input),'expected_episodes':len(episodes),
            'labeled_episodes':len(rows),'labels':rows},indent=2)+'\n')
    print(json.dumps({'labeled':len(rows),'episodes':len(episodes),'output':str(a.output)}),flush=True)

if __name__=='__main__':
    main()
