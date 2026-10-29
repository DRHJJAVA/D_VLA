def audit_feature_reuse(key, points, pixel, full, sel, rgb, crop, feat, xyz, Rh, th, pose):
    old = B.parent / 'iter_016/samples' / (key + '.npz')
    rec = {'key': key, 'existing': old.exists(), 'mode': 'fresh_current_frame'}
    if old.exists():
        with np.load(old) as z:
            pairs = {'xyz_world': points.astype(np.float32), 'pixel': pixel.astype(np.int64), 'input_rgb_pixel': full.astype(np.int64), 'causal_fps_indices': sel, 'input_crop_box': np.array(crop), 'rgb': rgb[pixel[:, 1], pixel[:, 0]], 'dino_feat': feat.astype(np.float32), 'xyz': xyz.astype(np.float32), 'hand_R': Rh.astype(np.float32), 'hand_t': th.astype(np.float32), 'pose': pose.astype(np.float32)}
            checks = {k: bool(np.array_equal(v, z[k])) for (k, v) in pairs.items()}
            rec['exact_checks'] = checks
            rec['feature_max_abs_diff'] = float(np.max(np.abs(feat - z['dino_feat']))) if feat.shape == z['dino_feat'].shape else None
            if all(checks.values()):
                feat = z['dino_feat'].copy()
                rec['mode'] = 'exact_verified_reuse'
    (B / 'audits' / (key + '_feature_reuse.json')).write_text(json.dumps(rec, indent=2))
    return feat
from pathlib import Path
import os, sys, json, hashlib
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np, h5py, torch
from torch.nn import functional as F
R = Path('/mnt/nvme1/drh/D_VLA')
B = Path(__file__).resolve().parents[1]
P = B.parent
sys.path[:0] = [str(R / 'src/v3'), str(R / 'src')]
from stage2_extract_dino import model, preproc
from supervision import targets
from build_point_graphs import fps, neighbors, interpolate

def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()

def dump(p, d):
    p.write_text(json.dumps(d, indent=2))

def buildgraph(x):
    i1 = fps(x, min(512, len(x)))
    x1 = x[i1]
    i2 = fps(x1, min(128, len(x1)))
    x2 = x1[i2]
    i3 = fps(x2, min(32, len(x2)))
    x3 = x2[i3]
    (a, b) = interpolate(x3, x2)
    (c, d) = interpolate(x2, x1)
    (e, f) = interpolate(x1, x)
    return dict(idx1=i1, idx2=i2, idx3=i3, nb1=neighbors(x, x1, 0.02, k=min(32, len(x))), nb2=neighbors(x1, x2, 0.04, k=min(32, len(x1))), nb3=neighbors(x2, x3, 0.08, k=min(32, len(x2))), fp3_idx=a, fp3_weight=b, fp2_idx=c, fp2_weight=d, fp1_idx=e, fp1_weight=f)

def main():
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.15)
    cfg = json.loads((B / 'config.json').read_text())
    refs = {c['id']: c for c in json.loads((P / 'iter_003/evidence/base/protocol.json').read_text())['episodes']}
    rows = []
    skips = []
    net = None
    for split in ['train', 'val']:
        if split == 'val' and (not (B / 'train_preparation_freeze.json').exists()):
            dump(B / 'train_preparation_freeze.json', {'train_rows': rows.copy(), 'skips': skips.copy(), 'source_lock_sha256': sha(B / 'source_locked.json')})
        for c in json.loads((P / ('iter_027' if split == 'train' else 'iter_028') / 'summary.json').read_text())['rows']:
            key = c['event'] + '_query' + str(c['query'])
            source = (P / 'iter_027/train' if split == 'train' else P / 'iter_028/val') / (key + '.npz')
            row = dict(c, key=key, query_frame=c['query'], split=split, target_source=str(source))
            if not c['eligible']:
                skips.append(dict(key=key, split=split, reason='insufficient unique sources before sampling'))
                continue
            key = row['key']
            ident = row['episode']
            t = row['query_frame']
            dest = B / 'samples' / (key + '.npz')
            graph = B / 'graphs' / (key + '.npz')
            cl = B / 'audits' / (key + '_sample_locked.json')
            if cl.exists():
                rec = json.loads(cl.read_text())
                for (f, h) in rec['files'].items():
                    assert sha(B / f) == h
                rows.append(rec['row'])
                continue
            assert not dest.exists() and (not graph.exists()), 'inspect partial preparation instead of overwrite'
            orig = np.load(source)
            world = orig['query_xyz']
            full = orig['query_pixel'].astype(np.int64)
            sel = fps(world, cfg['max_points']) if len(world) > cfg['max_points'] else np.arange(len(world))
            points = world[sel]
            pixel = full[sel]
            (target_all, known_all, conf_all, source_idx) = targets(row, orig)
            target = target_all[sel]
            known = known_all[sel]
            conf = conf_all[sel]
            sample_sources = source_idx[sel]
            pos = int(target.sum())
            neg = int((known & ~target).sum())
            unique_pos = len(np.unique(sample_sources[target]))
            unique_neg = len(np.unique(sample_sources[known & ~target]))
            if len(points) < 128 or unique_pos < 3 or unique_neg < 16:
                skips.append({'key': key, 'split': split, 'points': len(points), 'positive': pos, 'negative': neg, 'unique_positive': unique_pos, 'unique_negative': unique_neg, 'reason': 'insufficient sampled source support'})
                continue
            motion = np.load(P / 'iter_003/evidence/motion' / (ident + '.npz'))
            H = motion['hand_to_world'][t]
            q = motion['measured_finger_q'][t]
            attempt = int(row['event'].rsplit('attempt', 1)[1])
            start = json.loads((P / 'iter_003/evidence/motion' / (ident + '.json')).read_text())['attempts'][attempt - 1]['start_frame']
            assert t < start
            with h5py.File(refs[ident]['rgbd'], 'r') as f:
                g = f['agentview']
                rgb = g['rgb'][t]
                Tc = g['camera_to_world'][t]
                center = points.mean(0)
                xyz = (points - center) @ Tc[:3, :3]
                Rh = Tc[:3, :3].T @ H[:3, :3]
                th = (H[:3, 3] - center) @ Tc[:3, :3]
                pose = np.r_[th / 0.1, Rh[:, 0], Rh[:, 1], (q[0] - q[1] - 0.001) / 0.1]
                im = np.full_like(rgb, 127)
                im[full[:, 1], full[:, 0]] = rgb[full[:, 1], full[:, 0]]
                (x1, y1) = full.min(0)
                (x2, y2) = full.max(0) + 1
                if net is None:
                    net = model()
                with torch.inference_mode():
                    patch = net.forward_features(preproc(im[None, y1:y2, x1:x2]))[:, 1:, :].reshape(1, 32, 32, 384).permute(0, 3, 1, 2)
                gx = (pixel[:, 0] - x1 + 0.5) / (x2 - x1) * 2 - 1
                gy = (pixel[:, 1] - y1 + 0.5) / (y2 - y1) * 2 - 1
                grid = torch.as_tensor(np.stack([gx, gy], 1)[None, None], dtype=patch.dtype, device=patch.device)
                with torch.inference_mode():
                    feat = F.grid_sample(patch, grid, align_corners=False).squeeze(0).squeeze(1).T.cpu().numpy()
                    feat = audit_feature_reuse(key, points, pixel, full, sel, rgb, [x1, y1, x2, y2], feat, xyz, Rh, th, pose)
                assert feat.shape == (len(points), 384) and np.isfinite(feat).all()
                np.savez_compressed(dest, xyz=xyz.astype(np.float32), xyz_world=points.astype(np.float32), dino_feat=feat.astype(np.float16), pixel=pixel.astype(np.int32), input_rgb_pixel=full.astype(np.int32), input_crop_box=np.array([x1, y1, x2, y2], np.int32), rgb=rgb[pixel[:, 1], pixel[:, 0]], pose=pose.astype(np.float32), hand_R=Rh.astype(np.float32), hand_t=th.astype(np.float32), target=target, known=known, confidence=conf, query_frame=t, event=np.asarray(row['event']), episode=np.asarray(ident), split=np.asarray(split), query_rgbd=np.asarray(refs[ident]['rgbd']), coordinate_basis_camera_to_world=Tc, center_world=center, causal_fps_indices=sel, reference_source_index=sample_sources, offset=c['offset'])
            gr = buildgraph(xyz)
            np.savez_compressed(graph, **gr)
            for (k, v) in gr.items():
                if k.endswith('weight'):
                    assert np.isfinite(v).all() and np.allclose(v.sum(1), 1)
            rr = {**row, 'points': len(points), 'positive': pos, 'negative': neg, 'unknown': int((~known).sum()), 'unique_positive': unique_pos, 'unique_negative': unique_neg, 'full_rgb_pixels': len(full), 'closed_command_start_frame': start}
            dump(cl, {'row': rr, 'files': {str(f.relative_to(B)): sha(f) for f in [dest, graph]}})
            rows.append(rr)
            print(json.dumps(rr), flush=True)
    complete = {}
    for split in ['train', 'val']:
        complete[split] = {ev for ev in {r['event'] for r in rows if r['split'] == split} if {r['offset'] for r in rows if r['split'] == split and r['event'] == ev} == {15, 10, 5, 1}}
    kept = [r for r in rows if r['event'] in complete[r['split']]]
    for r in rows:
        if r not in kept:
            skips.append(dict(key=r['key'], split=r['split'], reason='event lacks all four eligible queries; excluded from both arms'))
    tr = {r['episode'] for r in kept if r['split'] == 'train'}
    va = {r['episode'] for r in kept if r['split'] == 'val'}
    assert tr.isdisjoint(va)
    gate = len(tr) >= cfg['min_train_events'] and len(va) >= cfg['min_val_events']
    manifest = {'rows': kept, 'prepared_rows': rows, 'skipped': skips, 'train_episodes': len(tr), 'val_episodes': len(va), 'training_gate_passed': gate, 'input_mask_independent_of_target': True, 'sampling_independent_of_target': True, 'source_only_trainval': True}
    dump(B / 'manifest.json', manifest)
    print('SAMPLES_COMPLETE', len(kept), 'TRAINING_GATE', gate, flush=True)
if __name__ == '__main__':
    main()
