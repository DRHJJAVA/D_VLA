#!/usr/bin/env python3
"""Wait for split-isolated point samples, build graphs, then train and evaluate."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

root=Path('/mnt/nvme1/drh/D_VLA')
p=argparse.ArgumentParser()
p.add_argument('--timeout',type=int,default=10800)
p.add_argument('--epochs',type=int,default=8)
a=p.parse_args()
log=root/'logs/prepare_all_splits.log'
start=time.monotonic()
while time.monotonic()-start<a.timeout:
    if log.exists() and 'All point sample partitions complete' in log.read_text(errors='replace'):
        break
    time.sleep(20)
else:raise TimeoutError('Single-frame sample preparation did not finish')
for partition in ('train','val','test'):
    print('Building graphs for',partition,flush=True)
    subprocess.run([sys.executable,str(root/'src/build_point_graphs.py'),
        '--sample-dir',str(root/'data/samples/single_frame_pilot'/partition),
        '--output-dir',str(root/'data/samples/single_frame_pilot/graphs'/partition),
        '--min-samples','1'],check=True)
print('Training held-out PointNet++',flush=True)
subprocess.run([sys.executable,str(root/'src/train_pointnet2_heldout.py'),
    '--samples',str(root/'data/samples/single_frame_pilot'),
    '--output-dir',str(root/'outputs/alphabet_soup_pointnet2_heldout'),
    '--epochs',str(a.epochs)],check=True)
