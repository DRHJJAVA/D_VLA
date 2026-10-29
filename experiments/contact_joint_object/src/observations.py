from pathlib import Path
import os, sys, json, time, hashlib, traceback
import numpy as np, h5py, cv2, torch
from scipy.spatial import cKDTree
from sampling import fps
from scipy.optimize import linear_sum_assignment
P = Path('/mnt/nvme1/drh/D_VLA')
B = P / 'experiments/contact_joint_object'
sys.path[:0] = [str(B / 'dependencies/python'), str(B / 'dependencies/sam2'), str(P / 'src/v3'), str(P / 'third_party/co_tracker_010')]
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from stage2_extract_dino import model, preproc
from cotracker.predictor import CoTrackerPredictor
CAMS = ['agentview', 'robot0_eye_in_hand']

def back(dep, K, C, uv):
    z = dep[uv[:, 1], uv[:, 0]]
    x = np.column_stack(((uv[:, 0] - K[0, 2]) * z / K[0, 0], (uv[:, 1] - K[1, 2]) * z / K[1, 1], z))
    return x @ C[:3, :3].T + C[:3, 3]

def match_cost(a, b):
    ds = cKDTree(a['xyz']).query(b['xyz'][::max(1, len(b['xyz']) // 128)])[0]
    return float(np.median(ds) / 0.02 + np.linalg.norm(a['desc'] - b['desc']))

def store(g, k, a):
    g.create_dataset(k, data=a, compression='gzip', compression_opts=1)

class Processor:

    def __init__(self):
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(0.45)
        torch.manual_seed(2026)
        sam = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', str(B / 'dependencies/sam21_tiny.pt'), device='cuda')
        self.sam = SAM2AutomaticMaskGenerator(sam, points_per_side=8, points_per_batch=64)
        self.dino = model().eval()
        self.cot = CoTrackerPredictor(checkpoint=str(P / 'experiments/region_pose/iter_010/weights/scaled_offline.pth'), offline=True, v2=False, window_len=60).cuda().eval()

    @torch.inference_mode()
    def run(self, row):
        assert row['split'] in ['train', 'dev']
        dest = B / 'cache/observations' / (row['id'] + '.h5')
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            return {'id': row['id'], 'reused': True}
        raw = B / 'cache/rgbd' / (row['id'] + '.h5')
        partial = dest.with_suffix('.partial.h5')
        previous = {}
        nextid = 0
        queries = [[], []]
        track_object = [[], []]
        counts = []
        with h5py.File(raw, 'r') as h, h5py.File(partial, 'w') as out:
            out.attrs.update(id=row['id'], split=row['split'], frames=row['frames'], contact_labels=False, proposal_method='SAM2.1 current-frame automatic masks; causal association; no simulator object labels')
            frame_root = out.create_group('frames')
            for t in range(row['frames']):
                frame = frame_root.create_group(str(t))
                objects = []
                label_imgs = []
                for (ci, cam) in enumerate(CAMS):
                    im = h[cam + '/rgb'][t]
                    dep = h[cam + '/depth_m'][t]
                    K = h[cam + '/K'][t]
                    C = h[cam + '/camera_to_world'][t]
                    robot = h[cam + '/robot_mask'][t].astype(bool)
                    masks = self.sam.generate(im)
                    ft = self.dino.forward_features(preproc(im[None]))[:, 1:].reshape(32, 32, 384).float().cpu().numpy().astype('f2')
                    store(frame, 'features_' + str(ci), ft)
                    accepted = []
                    for (mi, m) in enumerate(masks):
                        mask = m['segmentation'] & ~robot & np.isfinite(dep) & (dep > 0)
                        uv = np.argwhere(mask)[:, ::-1].astype(np.int64)
                        if len(uv) < 16:
                            continue
                        xyz = back(dep, K, C, uv)
                        ix = fps(xyz, min(2048, len(xyz)))
                        uv = uv[ix]
                        xyz = xyz[ix].astype('f4')
                        patch = np.clip(((uv + 0.5) / 256 * 32).astype(int), 0, 31)
                        desc = ft[patch[:, 1], patch[:, 0]].astype('f4').mean(0)
                        desc /= max(np.linalg.norm(desc), 1e-06)
                        uu = np.clip(uv + [1, 0], 0, 255)
                        vv = np.clip(uv + [0, 1], 0, 255)
                        a = back(dep, K, C, uu) - xyz
                        b = back(dep, K, C, vv) - xyz
                        n = np.cross(a, b)
                        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-08)
                        accepted.append({'xyz': xyz, 'uv': uv, 'camera': np.full(len(uv), ci, 'i1'), 'normal': n.astype('f4'), 'desc': desc, 'mask': mask, 'sam_quality': float(m['predicted_iou']), 'members': [(ci, mask)]})
                    for obj in accepted:
                        candidates = []
                        if ci == 1:
                            for (j, old) in enumerate(objects):
                                if not all((c == 0 for (c, _) in old['members'])):
                                    continue
                                da = cKDTree(old['xyz']).query(obj['xyz'])[0]
                                db = cKDTree(obj['xyz']).query(old['xyz'])[0]
                                if np.mean(da < 0.004) > 0.5 and np.mean(db < 0.004) > 0.5:
                                    candidates.append(j)
                        if len(candidates) == 1:
                            old = objects[candidates[0]]
                            for k in ['xyz', 'uv', 'camera', 'normal']:
                                old[k] = np.concatenate([old[k], obj[k]])
                            old['members'] += obj['members']
                            old['desc'] = (old['desc'] + obj['desc']) / 2
                            ix = fps(old['xyz'], 2048)
                            for k in ['xyz', 'uv', 'camera', 'normal']:
                                old[k] = old[k][ix]
                        else:
                            objects.append(obj)
                oldids = list(previous)
                cost = np.array([[match_cost(previous[i], o) for o in objects] for i in oldids]) if oldids and objects else np.zeros((0, len(objects)))
                assign = {}
                if cost.size:
                    (a, b) = linear_sum_assignment(cost)
                    for (i, j) in zip(a, b):
                        alternatives = np.delete(cost[:, j], i)
                        margin = float(alternatives.min() - cost[i, j]) if len(alternatives) else 10.0
                        if cost[i, j] < 2 and margin > 0.2:
                            assign[j] = oldids[i]
                current = {}
                ids = []
                for (j, obj) in enumerate(objects):
                    oid = assign.get(j)
                    if oid is None:
                        oid = nextid
                        nextid += 1
                    ids.append(oid)
                    g = frame.create_group('object_' + str(oid))
                    for k in ['xyz', 'uv', 'camera', 'normal', 'desc']:
                        store(g, k, obj[k])
                    g.attrs['sam_quality'] = obj['sam_quality']
                    g.attrs['association_cost'] = float(cost[oldids.index(oid), j]) if oid in oldids else np.nan
                    g.attrs['new_identity'] = oid not in oldids
                    maskids = []
                    for (ci, mask) in obj['members']:
                        store(g, 'mask_' + str(ci), np.packbits(mask, axis=None))
                        if t % 16 == 0:
                            vu = np.argwhere(mask)[:, ::-1]
                            chosen = vu[np.linspace(0, len(vu) - 1, min(8, len(vu))).astype(int)]
                            for (u, v) in chosen:
                                queries[ci].append([t, float(u), float(v)])
                                track_object[ci].append(oid)
                    current[oid] = obj
                previous = current
                counts.append(len(objects))
                frame.attrs['object_ids'] = np.array(ids, dtype='i4')
                if t % 16 == 0:
                    out.flush()
            tr = out.create_group('tracks')
            for (ci, cam) in enumerate(CAMS):
                g = tr.create_group(str(ci))
                q = np.asarray(queries[ci], dtype='f4').reshape(-1, 3)
                store(g, 'queries', q)
                store(g, 'seed_object_id', np.asarray(track_object[ci], dtype='i4'))
                if len(q) == 0:
                    continue
                video = torch.from_numpy(h[cam + '/rgb'][:]).permute(0, 3, 1, 2)[None].float().cuda()
                tracks = []
                visible = []
                for start in range(0, len(q), 256):
                    (xy, v) = self.cot(video, queries=torch.from_numpy(q[start:start + 256])[None].cuda(), backward_tracking=False)
                    tracks.append(xy[0].cpu().numpy())
                    visible.append(v[0].cpu().numpy())
                xy = np.concatenate(tracks, axis=1)
                vis = np.concatenate(visible, axis=1)
                world = np.full((len(xy), len(q), 3), np.nan, 'f4')
                valid = np.zeros((len(xy), len(q)), bool)
                for t in range(len(xy)):
                    uv = np.rint(xy[t]).astype(np.int64)
                    good = (vis[t] > 0.9) & (q[:, 0] <= t) & (uv[:, 0] >= 0) & (uv[:, 0] < 256) & (uv[:, 1] >= 0) & (uv[:, 1] < 256)
                    ix = np.flatnonzero(good)
                    dep = h[cam + '/depth_m'][t]
                    robot = h[cam + '/robot_mask'][t]
                    ix = ix[(robot[uv[ix, 1], uv[ix, 0]] == 0) & np.isfinite(dep[uv[ix, 1], uv[ix, 0]]) & (dep[uv[ix, 1], uv[ix, 0]] > 0)]
                    if len(ix):
                        world[t, ix] = back(dep, h[cam + '/K'][t], h[cam + '/camera_to_world'][t], uv[ix])
                        valid[t, ix] = True
                store(g, 'uv', xy)
                store(g, 'visible', vis)
                store(g, 'world', world)
                store(g, 'valid', valid)
                del video
            out.attrs['object_counts'] = np.asarray(counts, 'i4')
            out.attrs['tracker_query_chunk'] = 256
        partial.replace(dest)
        return {'id': row['id'], 'frames': row['frames'], 'object_frame_proposals': sum(counts), 'identities': nextid, 'query_counts': list(map(len, queries)), 'bytes': dest.stat().st_size}

def main():
    proc = Processor()
    m = json.loads((B / 'manifests/episodes.json').read_text())['episodes']
    rows = [r for r in m if r['split'] in ['train', 'dev']]
    if '--smoke' in sys.argv:
        rows = [rows[0]]
    result = []
    start = time.time()
    for row in rows:
        q = B / 'cache/rgbd' / (row['id'] + '.audit.json')
        while not q.exists():
            if (B / 'failure.json').exists():
                raise RuntimeError('upstream rendering failed')
            time.sleep(5)
        result.append(proc.run(row))
        (B / 'manifests/observation_progress.json').write_text(json.dumps({'completed': len(result), 'total': len(rows), 'elapsed_seconds': time.time() - start, 'episodes': result}, indent=2))
    (B / 'manifests' / ('observation_smoke.json' if '--smoke' in sys.argv else 'observation_complete.json')).write_text(json.dumps({'ok': True, 'episodes': result}, indent=2))
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        (B / 'logs/observations_traceback.log').write_text(traceback.format_exc())
        (B / 'observation_failure.json').write_text(json.dumps({'error': type(e).__name__, 'detail': str(e)[:250]}))
