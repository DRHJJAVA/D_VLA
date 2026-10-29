from pathlib import Path
import sys, json, hashlib
import numpy as np
from scipy.spatial import cKDTree
R = Path('/mnt/nvme1/drh/D_VLA')
B = Path(__file__).resolve().parents[1]
P = B.parent

def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()

def dump(p, d):
    p.write_text(json.dumps(d, indent=2))

def apply(p, T):
    return p @ T[:3, :3].T + T[:3, 3]

def rigid(a, b):
    ac = a.mean(0)
    bc = b.mean(0)
    (u, s, v) = np.linalg.svd((a - ac).T @ (b - bc))
    rot = v.T @ u.T
    if np.linalg.det(rot) < 0:
        v[-1] *= -1
        rot = v.T @ u.T
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = bc - rot @ ac
    return T

def inside(p, q, margin):
    lo = np.array([-0.008, q[1] + 0.0005, 0.0854]) - margin
    hi = np.array([0.008, q[0] - 0.0005, 0.1014]) + margin
    return np.all(p >= lo, axis=-1) & np.all(p <= hi, axis=-1)

def main(split):
    assert split in ['train', 'val']
    out = B / 'targets' / split
    for (f, h) in json.loads((B / 'source_locked.json').read_text()).items():
        assert sha(R / f) == h
    for (f, h) in json.loads((B / (split + '_input_locked.json')).read_text()).items():
        assert sha(R / f) == h
    if (out / 'locked.json').exists():
        for (f, h) in json.loads((out / 'locked.json').read_text()).items():
            assert sha(out / f) == h
        print('REUSE_TARGETS', split)
        return
    if split == 'val':
        fr = json.loads((B / 'pre_validation_freeze.json').read_text())
        assert fr['train_summary'] == sha(B / 'targets/train/summary.json') and fr['source_lock'] == sha(B / 'source_locked.json')
    rows = []
    for c in json.loads((P / 'iter_024' / split / 'summary.json').read_text())['cases']:
        name = c['event'] + '_query' + str(c['query'])
        dest = out / (name + '.npz')
        diag = out / (name + '.json')
        lock = out / (name + '_locked.json')
        if lock.exists():
            for (f, h) in json.loads(lock.read_text()).items():
                assert sha(out / f) == h
            rows.append(json.loads(diag.read_text()))
            continue
        z = np.load(P / 'iter_024' / split / (name + '.npz'))
        cr = np.load(P / ('iter_022' if split == 'train' else 'iter_023') / split / 'correspondences' / (name + '.npz'))
        query = z['query_xyz']
        frames = z['selected_frames']
        positives = []
        negatives = []
        possible = []
        posweights = []
        negweights = []
        spreadrows = []
        records = []
        for (j, t) in enumerate(frames):
            q = z['finger_q_' + str(t)]
            Q = z['query_to_current_hand_' + str(t)]
            H = z['hand_to_world_' + str(t)]
            mask = cr['inlier_' + str(t)]
            a = cr['query_xyz_' + str(t)][mask]
            b = cr['bridge_xyz_' + str(t)][mask]
            alternatives = []
            for k in range(len(a)):
                aa = np.delete(a, k, 0)
                bb = np.delete(b, k, 0)
                if len(aa) < 7:
                    continue
                scale = np.linalg.svd(aa - aa.mean(0), compute_uv=False) / np.sqrt(len(aa))
                if scale[1] < 0.003:
                    continue
                alternatives.append(np.linalg.inv(H) @ rigid(aa, bb))
            positions = np.stack([apply(query, qq) for qq in [Q] + alternatives])
            spread = np.linalg.norm(positions - positions[0], axis=-1).max(axis=0)
            robust_inside = inside(positions, q, 0.001).all(axis=0)
            may_inside = inside(positions, q, 0.004).any(axis=0)
            stable = (spread <= 0.002) & (len(alternatives) >= 7)
            source = (z['frame'] == t) & z['supported_candidate']
            dist = cKDTree(z['mapped_query_xyz'][source]).query(query)[0] if source.any() else np.full(len(query), np.inf)
            pos = (dist <= 0.003) & robust_inside & stable
            neg = z['query_visible_at_frames'][j] & ~may_inside & stable
            positives.append(pos)
            negatives.append(neg)
            possible.append(may_inside)
            spreadrows.append(spread)
            posweights.append(np.where(pos, np.exp(-spread / 0.003) * np.exp(-dist / 0.003), 0))
            negweights.append(np.where(neg, np.exp(-spread / 0.003), 0))
            records.append({'frame': int(t), 'valid_leave_one_out_fits': len(alternatives), 'positive_supported_points': int(pos.sum()), 'visible_stable_outside_points': int(neg.sum()), 'possible_inside_expanded4_points': int(may_inside.sum())})
        ps = np.sum(positives, axis=0)
        ns = np.sum(negatives, axis=0)
        ever = np.any(possible, axis=0)
        positive = ps >= 2
        negative = (ns >= 2) & ~ever
        conflict = positive & negative
        positive &= ~conflict
        negative &= ~conflict
        assert not conflict.any()
        known = positive | negative
        confidence = np.where(positive, np.sum(posweights, axis=0) / np.maximum(ps, 1), np.where(negative, np.sum(negweights, axis=0) / np.maximum(ns, 1), 0)).astype(np.float32)
        assert np.all(confidence[~known] == 0) and np.all(confidence[known] > 0)
        np.savez_compressed(dest, query_xyz=query, query_pixel=z['query_pixel'], query_rgb=z['query_rgb'], selected_frames=frames, target=positive.astype(np.float32), known=known, confidence=confidence, positive_frames=np.array(positives), negative_frames=np.array(negatives), possible_inside_frames=np.array(possible), position_sensitivity_m=np.array(spreadrows), positive_support_count=ps, negative_support_count=ns, visible_at_frames=z['query_visible_at_frames'])
        row = {'key': name, 'episode': c['id'], 'event': c['event'], 'query_frame': c['query'], 'split': split, 'points': len(query), 'positive': int(positive.sum()), 'negative': int(negative.sum()), 'unknown': int((~known).sum()), 'conflicts': int(conflict.sum()), 'frames': frames.tolist(), 'frame_diagnostics': records, 'target_source': str(dest), 'physical_contact_ground_truth': False}
        dump(diag, row)
        dump(lock, {dest.name: sha(dest), diag.name: sha(diag)})
        rows.append(row)
        print(json.dumps(row), flush=True)
    dump(out / 'summary.json', {'split': split, 'rows': rows, 'events': len(rows), 'unknown_not_negative': True})
    dump(out / 'locked.json', {f.name: sha(f) for f in out.iterdir() if f.is_file() and f.name != 'locked.json'})
if __name__ == '__main__':
    main(sys.argv[1])
