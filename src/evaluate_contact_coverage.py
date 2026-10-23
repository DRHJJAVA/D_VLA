#!/usr/bin/env python3
"""Evaluate contact coverage at fixed visible-object point budgets."""
import argparse
import json
import math
from pathlib import Path
import sys
import h5py
import numpy as np
import torch

ROOT = Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0, str(ROOT / 'src'))
from pointnet2_grasp import PointNet2Grasp
from train_pointnet2_grasp import tensors

CAMERAS = ('agentview', 'robot0_eye_in_hand')
OBJECT = 'alphabet_soup_1_main'
BUDGETS = (0.1, 0.2, 0.3)


def unique_contacts(event):
    points = np.asarray([x['position_object'] for side in ('left_contacts', 'right_contacts')
                         for x in event[side]], dtype=np.float32)
    voxels = np.round(points / 0.003).astype(np.int32)
    _, first = np.unique(voxels, axis=0, return_index=True)
    return points[np.sort(first)]


def visible_contacts(world, video, frame, tolerance):
    visible = np.zeros(len(world), dtype=bool)
    for camera in CAMERAS:
        group = video[camera]
        transform = group['camera_to_world'][frame]
        camera_points = (world - transform[:3, 3]) @ transform[:3, :3]
        z = camera_points[:, 2]
        K = group['K'][frame]
        safe_z = np.where(z > 0, z, 1)
        u = np.rint(K[0, 0] * camera_points[:, 0] / safe_z + K[0, 2]).astype(int)
        v = np.rint(K[1, 1] * camera_points[:, 1] / safe_z + K[1, 2]).astype(int)
        depth = group['depth_m'][frame]
        inside = (z > 0) & (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
        indices = np.flatnonzero(inside)
        if len(indices):
            measured = depth[v[indices], u[indices]]
            visible[indices] |= np.isfinite(measured) & (np.abs(measured - z[indices]) <= tolerance)
    return visible


def coverage(selected, contacts, tolerance):
    if len(contacts) == 0 or len(selected) == 0:
        return None
    distances = np.linalg.norm(contacts[:, None, :] - selected[None, :, :], axis=-1)
    return float(np.mean(np.min(distances, axis=1) <= tolerance))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--partition', choices=('val', 'test'), default='test')
    p.add_argument('--random-repeats', type=int, default=100)
    p.add_argument('--sample-root', type=Path, default=ROOT/'data/samples/single_frame_pilot')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/alphabet_soup_pointnet2_heldout/best.pt')
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/alphabet_soup_pointnet2_heldout')
    p.add_argument('--coverage-tolerance-m', type=float, default=None)
    a = p.parse_args()
    if a.coverage_tolerance_m is not None and a.coverage_tolerance_m <= 0:
        raise ValueError('Coverage tolerance must be positive')
    torch.set_num_threads(4)
    sample_root = a.sample_root
    checkpoint = torch.load(a.checkpoint,
                            map_location='cpu', weights_only=False)
    model = PointNet2Grasp()
    model.load_state_dict(checkpoint['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    rng = np.random.default_rng(2026)
    rows = []
    files = sorted((sample_root / a.partition).glob('*.npz'))
    if not files:
        raise RuntimeError('No samples in partition')
    for path in files:
        with np.load(path) as f:
            item = {key: f[key].copy() for key in ('xyz', 'dino_feat', 'target', 'valid_mask')}
            object_ids = f['object_ids'].copy()
            episode = str(f['episode_id'])
            frame = int(f['frame_id'])
            sigma = float(f['sigma_m'])
        with np.load(sample_root / 'graphs' / a.partition / path.name) as f:
            graph = {key: f[key].copy() for key in f.files}
        xyz, feature, target, valid, graph_t = tensors(item, graph, device)
        with torch.inference_mode():
            score = torch.sigmoid(model(xyz, feature, graph_t)).detach().cpu().numpy()
        score = score.reshape(-1)
        object_points = item['xyz'][object_ids == 1]
        object_scores = score[object_ids == 1]
        if len(object_points) == 0:
            raise RuntimeError(f'No target points: {path.name}')
        report = json.loads((ROOT / 'data/events' / f'alphabet_soup_{episode}_grasp_candidates.json').read_text())
        events = [x for x in report['events'] if x['passes_numeric_filter']]
        event = max(events, key=lambda x: x['duration_frames'])
        contact_object = unique_contacts(event)
        probe = json.loads((ROOT / 'data/events' / f'alphabet_soup_{episode}_contact_probe.json').read_text())
        pose = probe['records'][frame]['target_body_poses'][OBJECT]
        position = np.asarray(pose['position_world'], dtype=np.float32)
        rotation = np.asarray(pose['rotation_world'], dtype=np.float32)
        contact_world = contact_object @ rotation.T + position
        with h5py.File(ROOT / 'data/rendered' / f'alphabet_soup_{episode}_full_rgbd.h5') as video:
            visible = visible_contacts(contact_world, video, frame, max(0.01, 2 * sigma))
        visible_world = contact_world[visible]
        tolerance = sigma if a.coverage_tolerance_m is None else a.coverage_tolerance_m
        row = {'episode': episode, 'frame': frame, 'object_points': len(object_points),
               'contacts': len(contact_world), 'visible_contacts': len(visible_world),
               'visible_fraction': float(np.mean(visible)), 'sigma_m': sigma,
               'visibility_depth_tolerance_m': max(0.01, 2 * sigma),
               'coverage_tolerance_m': tolerance,
               'sampling_ceiling': coverage(object_points, visible_world, tolerance)}
        for q in BUDGETS:
            key = str(int(q * 100))
            n = max(1, math.ceil(q * len(object_points)))
            best = np.argsort(-object_scores, kind='stable')[:n]
            row[f'coverage_at_{key}'] = coverage(object_points[best], visible_world, tolerance)
            if len(visible_world):
                row[f'random_at_{key}'] = float(np.mean([
                    coverage(object_points[rng.choice(len(object_points), size=n, replace=False)],
                             visible_world, tolerance) for _ in range(a.random_repeats)]))
            else:
                row[f'random_at_{key}'] = None
        rows.append(row)
    episodes = {}
    for episode in sorted({r['episode'] for r in rows}):
        group = [r for r in rows if r['episode'] == episode]
        episodes[episode] = {}
        for key in ('visible_fraction', 'sampling_ceiling', *[f'coverage_at_{int(q*100)}' for q in BUDGETS],
                    *[f'random_at_{int(q*100)}' for q in BUDGETS]):
            values = [r[key] for r in group if r[key] is not None]
            episodes[episode][key] = float(np.mean(values)) if values else None
    summary = {'partition': a.partition, 'best_epoch': checkpoint['epoch'],
               'evaluation': 'held-out event contacts transformed to observation-frame object pose',
               'visibility': 'two-camera depth agreement; tolerance max(1 cm, 2 sigma)',
               'coverage_tolerance': 'sample sigma' if a.coverage_tolerance_m is None else a.coverage_tolerance_m,
               'budget_denominator': 'offline target-object points',
               'object_ids_as_model_input': False, 'label_quality': 'provisional_unreviewed',
               'samples': len(rows), 'episodes': len(episodes),
               'samples_with_visible_contacts': sum(r['visible_contacts'] > 0 for r in rows),
               'mean_visible_fraction': float(np.mean([r['visible_fraction'] for r in rows])),
               'episode_metrics': {key: float(np.mean([v[key] for v in episodes.values() if v[key] is not None]))
                                   if any(v[key] is not None for v in episodes.values()) else None
                                   for key in next(iter(episodes.values()))},
               'per_episode': episodes, 'per_sample': rows}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'sigma' if a.coverage_tolerance_m is None else f'{a.coverage_tolerance_m:.3f}m'
    out = a.output_dir / f'contact_coverage_{a.partition}_{suffix}.json'
    out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in ('partition','samples','episodes',
        'samples_with_visible_contacts','mean_visible_fraction','episode_metrics')}, indent=2))

if __name__ == '__main__':
    main()
