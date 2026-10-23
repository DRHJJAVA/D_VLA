#!/usr/bin/env python3
"""Prepare split-isolated single-frame samples at a consistent point count."""
import argparse
import subprocess
import sys
from pathlib import Path

root = Path('/mnt/nvme1/drh/D_VLA')
p = argparse.ArgumentParser()
p.add_argument('--points', type=int, default=4096)
p.add_argument('--output-dir', type=Path, default=root/'data/samples/single_frame_pilot')
a = p.parse_args()
for partition in ('train', 'val', 'test'):
    print(f'Preparing {partition} at {a.points} points', flush=True)
    command = [sys.executable, str(root/'src/prepare_point_samples.py'),
        '--raw', str(root/'data/raw/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5'),
        '--rendered-dir', str(root/'data/rendered'),
        '--events-dir', str(root/'data/events'),
        '--labels', str(root/'data/events/alphabet_soup_training_labels.json'),
        '--split', str(root/'data/splits/alphabet_soup_episode_split.json'),
        '--output-dir', str(a.output_dir),
        '--libero-root', '/home/drh/project/ECV-VLA/vendor/LIBERO',
        '--partition', partition, '--limit-episodes', '0', '--points', str(a.points)]
    subprocess.run(command, check=True)
print('All point sample partitions complete', flush=True)
