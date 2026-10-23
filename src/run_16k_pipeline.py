#!/usr/bin/env python3
"""Resume-safe, resource-aware provisional 16k point experiment."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/mnt/nvme1/drh/D_VLA')
SAMPLES = ROOT/'data/samples/single_frame_16k'
OUTPUT = ROOT/'outputs/alphabet_soup_pointnet2_16k_provisional'
PYTHON = sys.executable

def gpu_ready():
    result = subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    free, utilization = [int(x.strip()) for x in result.stdout.splitlines()[0].split(',')]
    return free >= 20000 and utilization <= 70, free, utilization

def wait_for_gpu(max_seconds=21600):
    start = time.monotonic()
    consecutive = 0
    while time.monotonic() - start < max_seconds:
        ready, free, utilization = gpu_ready()
        consecutive = consecutive + 1 if ready else 0
        print(json.dumps({'stage':'gpu_wait','free_mib':free,'utilization_percent':utilization,
                          'consecutive_ready':consecutive}), flush=True)
        if consecutive >= 2:
            return
        time.sleep(30)
    raise TimeoutError('Shared GPU did not become available within 6 hours')

def run(command, env=None):
    print(json.dumps({'stage':'command','argv':command}), flush=True)
    subprocess.run(command, check=True, env=env)

def main():
    wait_for_gpu()
    env = os.environ.copy()
    env.update({'LIBERO_CONFIG_PATH':str(ROOT/'configs/libero'),
                'MUJOCO_GL':'egl', 'PYOPENGL_PLATFORM':'egl'})
    run([PYTHON,str(ROOT/'src/prepare_all_splits.py'), '--points','16384',
         '--output-dir',str(SAMPLES)],env)
    split = json.loads((ROOT/'data/splits/alphabet_soup_episode_split.json').read_text())
    for partition in ('train','val','test'):
        expected = len(split[partition])*4
        samples = list((SAMPLES/partition).glob('*.npz'))
        if len(samples) != expected:
            raise RuntimeError(f'{partition}: expected {expected} samples, found {len(samples)}')
        run([PYTHON,str(ROOT/'src/build_point_graphs.py'), '--sample-dir',str(SAMPLES/partition),
             '--output-dir',str(SAMPLES/'graphs'/partition), '--min-samples',str(expected)])
        graphs = list((SAMPLES/'graphs'/partition).glob('*.npz'))
        if len(graphs) != expected:
            raise RuntimeError(f'{partition}: expected {expected} graphs, found {len(graphs)}')
    wait_for_gpu()
    run([PYTHON,str(ROOT/'src/train_pointnet2_heldout.py'), '--samples',str(SAMPLES),
         '--output-dir',str(OUTPUT), '--epochs','8'],env)
    if not (OUTPUT/'summary.json').exists():
        raise RuntimeError('Training ended without summary.json')
    print('16k provisional point pipeline complete', flush=True)

if __name__ == '__main__':
    main()
