from pathlib import Path
import json, time, traceback
import numpy as np, h5py, torch
from geometry import separation_lower_bound
from sampling import fps
from scipy.spatial import cKDTree
from data import B, to_device, CAMS, mask
from network import solve_motion, apply_twist

@torch.no_grad()
def refresh_episode(teacher, ds, row, epoch, point_limit=128, prune=True):
    started = time.time()
    teacher.eval()
    dest = B / 'cache/online' / ('epoch_%03d' % epoch) / (row['id'] + '.h5')
    dest.parent.mkdir(parents=True, exist_ok=True)
    (op, rp) = ds.paths(row)
    n = row['frames']
    records = []
    tracks = []
    sigma = ds.sigma
    with h5py.File(op, 'r') as obs, h5py.File(rp, 'r') as rendered:
        for ci in [0, 1]:
            tr = obs['tracks/' + str(ci)]
            if 'world' not in tr:
                tracks.append(None)
                continue
            tracks.append({'world': tr['world'][:], 'valid': tr['valid'][:], 'lo': np.zeros(tr['valid'].shape, 'f4'), 'hi': np.ones(tr['valid'].shape, 'f4'), 'supported': np.zeros(tr['valid'].shape, bool)})
        for t in range(n):
            ff = obs['frames/' + str(t)]
            for oid in ff.attrs['object_ids']:
                obj = ff['object_' + str(oid)]
                allxyz = obj['xyz'][:]
                selected = fps(allxyz.astype(float), min(point_limit, len(allxyz)))
                x = allxyz[selected]
                bound = np.exp(-0.5 * (separation_lower_bound(rendered, t, x) / (sigma * 4)) ** 2)
                if prune and np.max(bound) < 1e-06:
                    raw = {'xyz': x, 'source_index': selected}
                    lo = np.zeros(len(x), 'f4')
                    hi = bound.astype('f4')
                    records.append({'t': t, 'oid': int(oid), 'index': selected, 'xyz': x, 'lo': lo, 'hi': hi, 'contact': np.zeros((len(x), 2), 'f4'), 'reliability': 0.0, 'motion_error': None, 'free_error': None, 'numerically_pruned': True})
                else:
                    raw = ds.get(row, t, int(oid), ci=t % 2, point_limit=point_limit)
                    b = to_device(raw)
                    o = teacher(b)
                    c = o['contact'].cpu().numpy().max(1)
                    if len(b['motion_source']):
                        xi = solve_motion(b, o)
                        xf = solve_motion(b, o, free=True)
                        center = b['xyz'].mean(0)
                        ep = ((apply_twist(b['motion_source'], xi, center) - b['motion_target']) / sigma).square().sum(1).median().item()
                        ef = ((apply_twist(b['motion_source'], xf, center) - b['motion_target']) / sigma).square().sum(1).median().item()
                        reliability = (1 - np.exp(-max(0, ef - ep))) / (1 + ep)
                    else:
                        ep = None
                        ef = None
                        reliability = 0.0
                    lo = c * reliability
                    hi = lo + (1 - reliability)
                    d = np.maximum(raw['sdf'].min(1), 0)
                    separation_upper = np.exp(-0.5 * (d / (sigma * 4)) ** 2)
                    hi = np.minimum(hi, separation_upper)
                    lo = np.minimum(lo, hi)
                    records.append({'t': t, 'oid': int(oid), 'index': raw['source_index'], 'xyz': raw['xyz'], 'lo': lo.astype('f4'), 'hi': hi.astype('f4'), 'contact': o['contact'].cpu().numpy(), 'reliability': reliability, 'motion_error': ep, 'free_error': ef})
                obj = ff['object_' + str(oid)]
                for (ci, tr) in enumerate(tracks):
                    if tr is None:
                        continue
                    ix = np.flatnonzero(tr['valid'][t])
                    if not len(ix):
                        continue
                    pixel = np.rint(obs['tracks/' + str(ci) + '/uv'][t, ix]).astype(int)
                    inside = (pixel >= 0).all(1) & (pixel < 256).all(1)
                    ix = ix[inside]
                    pixel = pixel[inside]
                    m = mask(obj, ci)
                    ix = ix[m[pixel[:, 1], pixel[:, 0]]]
                    if not len(ix):
                        continue
                    (dist, j) = cKDTree(raw['xyz']).query(tr['world'][t, ix])
                    mapping = np.exp(-0.5 * (dist / sigma) ** 2)
                    low = lo[j] * mapping
                    high = hi[j] * mapping + (1 - mapping)
                    old = tr['supported'][t, ix]
                    tr['lo'][t, ix] = np.where(old, np.minimum(tr['lo'][t, ix], low), low)
                    tr['hi'][t, ix] = np.where(old, np.maximum(tr['hi'][t, ix], high), high)
                    tr['supported'][t, ix] = True
        metadata = json.loads(rendered.attrs['robot_metadata'])
        fids = [metadata['body_names'].index(n) for n in ['gripper0_leftfinger', 'gripper0_rightfinger']]
        for (ci, trajectory) in enumerate(tracks):
            if trajectory is None:
                continue
            for tt in range(1, n):
                ids = np.flatnonzero(trajectory['valid'][tt] & trajectory['valid'][tt - 1] & trajectory['supported'][tt] & trajectory['supported'][tt - 1])
                if not len(ids):
                    continue
                continuity = []
                for fid in fids:
                    x0 = (trajectory['world'][tt - 1, ids] - rendered['robot/body_position'][tt - 1, fid]) @ rendered['robot/body_rotation'][tt - 1, fid]
                    x1 = (trajectory['world'][tt, ids] - rendered['robot/body_position'][tt, fid]) @ rendered['robot/body_rotation'][tt, fid]
                    continuity.append(np.exp(-0.5 * np.sum((x1 - x0) ** 2, axis=1) / sigma ** 2))
                continuation = np.max(continuity, axis=0)
                prior = trajectory['lo'][tt - 1, ids] * continuation
                trajectory['lo'][tt, ids] = np.minimum(trajectory['hi'][tt, ids], np.maximum(trajectory['lo'][tt, ids], prior))
        for tr in tracks:
            if tr is None:
                continue
            Dl = tr['lo'].sum(0) * 0.05
            Dh = tr['hi'].sum(0) * 0.05
            T = n * 0.05
            tr['Yl'] = 1 - np.exp(-3 * Dl / T)
            tr['Yh'] = 1 - np.exp(-3 * Dh / T)
            absent = ~tr['supported'].any(0)
            tr['Yl'][absent] = 0
            tr['Yh'][absent] = 1
        partial = dest.with_suffix('.partial.h5')
        active = 0
        total = 0
        with h5py.File(partial, 'w') as out:
            out.attrs.update(epoch=epoch, id=row['id'], frames=n, teacher_point_limit=point_limit, all_video_frames_visited=True, uses_contact_labels=False)
            for rec in records:
                t = rec['t']
                xyz = rec['xyz']
                low = np.zeros(len(xyz), 'f4')
                high = np.ones(len(xyz), 'f4')
                mapped = np.zeros(len(xyz), bool)
                for (ci, tr) in enumerate(tracks):
                    if tr is None:
                        continue
                    ids = np.flatnonzero(tr['valid'][t] & tr['supported'][t])
                    if not len(ids):
                        continue
                    (dist, j) = cKDTree(tr['world'][t, ids]).query(xyz)
                    w = np.exp(-0.5 * (dist / sigma) ** 2)
                    sel = ids[j]
                    ll = w * tr['Yl'][sel]
                    hh = w * tr['Yh'][sel] + 1 - w
                    low = np.where(mapped, np.minimum(low, ll), ll)
                    high = np.where(mapped, np.maximum(high, hh), hh)
                    mapped |= w > 1e-06
                low[~mapped] = 0
                high[~mapped] = 1
                g = out.require_group(str(t)).create_group(str(rec['oid']))
                for (name, val) in [('index', rec['index']), ('xyz', xyz), ('low', low), ('high', high), ('contact', rec['contact']), ('contact_reliability', np.full_like(rec['contact'], rec['reliability']))]:
                    g.create_dataset(name, data=val, compression='gzip', compression_opts=1)
                g.attrs['reliability'] = rec['reliability']
                g.attrs['motion_error'] = np.nan if rec['motion_error'] is None else rec['motion_error']
                g.attrs['free_error'] = np.nan if rec['free_error'] is None else rec['free_error']
                active += int((high - low < 0.999).sum())
                total += len(low)
            out.attrs['active_score_points'] = active
            out.attrs['total_points'] = total
            out.attrs['scope'] = 'online estimates from EMA learned contact and physical observation likelihood; not contact truth'
        partial.replace(dest)
    return {'id': row['id'], 'frames': n, 'object_frames': len(records), 'target_points': total, 'active_score_points': active, 'path': str(dest), 'elapsed_seconds': time.time() - started, 'numerically_pruned_object_frames': sum((bool(r.get('numerically_pruned', False)) for r in records))}

def target_for(raw, epoch, device='cuda'):
    path = B / 'cache/online' / ('epoch_%03d' % epoch) / (raw['id'] + '.h5')
    if not path.exists():
        return None
    key = str(raw['frame']) + '/' + str(raw['object_id'])
    with h5py.File(path, 'r') as h:
        if key not in h:
            return None
        g = h[key]
        (dist, j) = cKDTree(g['xyz'][:]).query(raw['xyz'])
        w = np.exp(-0.5 * (dist / float(raw['measurement_sigma'])) ** 2).astype('f4')
        low = w * g['low'][:][j]
        high = w * g['high'][:][j] + 1 - w
        return {'low': torch.from_numpy(low).to(device), 'high': torch.from_numpy(high).to(device), 'contact': torch.from_numpy(g['contact'][:][j]).to(device), 'contact_reliability': torch.from_numpy(g['contact_reliability'][:][j] * w[:, None]).to(device)}
