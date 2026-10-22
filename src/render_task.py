#!/usr/bin/env python3
"""Render full RGB-D videos for all demonstrations in one LIBERO task HDF5."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--libero-root',type=Path,required=True)
    p.add_argument('--prefix',required=True)
    p.add_argument('--size',type=int,default=448)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with h5py.File(a.input) as f:
        demos=sorted(f['data'],key=lambda name:int(name.split('_')[-1]))
        counts={name:len(f['data'][name]['states']) for name in demos}
    result={'input':str(a.input),'expected':len(demos),'completed':[],'failed':[]}
    for name in demos:
        dest=a.output_dir/f'{a.prefix}_{name}_full_rgbd.h5'
        if dest.exists():
            with h5py.File(dest) as f:
                if int(f.attrs.get('num_frames',-1))==counts[name] and not bool(f.attrs.get('has_segmentation',True)):
                    result['completed'].append(name)
                    print(json.dumps({'episode':name,'status':'already_complete','frames':counts[name]}),flush=True)
                    continue
        command=[sys.executable,str(Path(__file__).with_name('render_rgbd_episode.py')),
                 '--input',str(a.input),'--output',str(dest),'--libero-root',str(a.libero_root),
                 '--episode',name,'--size',str(a.size)]
        print(json.dumps({'episode':name,'status':'rendering','frames':counts[name]}),flush=True)
        run=subprocess.run(command,check=False)
        if run.returncode:
            result['failed'].append({'episode':name,'exit_code':run.returncode})
            print(json.dumps(result),flush=True)
            raise SystemExit(run.returncode)
        result['completed'].append(name)
        (a.output_dir/f'{a.prefix}_render_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
