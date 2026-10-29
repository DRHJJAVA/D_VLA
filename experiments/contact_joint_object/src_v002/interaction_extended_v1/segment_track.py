import os, sys, json, time, hashlib, argparse, traceback, contextlib
from pathlib import Path
from collections import OrderedDict
import numpy as np
import h5py
from PIL import Image
BASE = Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object')
sys.path[:0] = [str(BASE / 'dependencies/python'), str(BASE / 'dependencies/sam2')]
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ['USE_TF'] = '0'
import torch
torch.set_num_threads(4)
from scipy.optimize import linear_sum_assignment
from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import sam2.sam2_video_predictor as vpmod
CAMERAS = ['agentview', 'robot0_eye_in_hand']
EPISODES = ['two_objects', 'drawer_pull', 'drawer_push', 'stove_switch']
PROTOCOL = {'method': 'class_agnostic_dual_view_sam21_video', 'points_per_side': 32, 'redetect_every': 16, 'minimum_pixels': 16, 'duplicate_mask_iou': 0.85, 'refresh_mask_iou': 0.35, 'retire_empty_frames': 32, 'depth_filter': 'finite_positive', 'robot_filter': 'provided_robot_only_mask', 'background_policy': 'retain_all_mark_large_area_only', 'causal': 'H5 loader rejects frame greater than allowed; current-frame prompts; forward only', 'model': 'sam21_tiny', 'autocast': 'bfloat16', 'contact_labels': False, 'task_text_used': False, 'simulator_object_pose_used': False}

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for q in iter(lambda : f.read(4 * 1024 * 1024), b''):
            h.update(q)
    return h.hexdigest()

def now():
    from datetime import datetime, timezone, timedelta
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

def write_json(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    q.replace(p)

class CausalFrames:

    def __init__(self, rgb, size):
        self.rgb = rgb
        self.size = size
        self.allowed = 0
        self.highest_read = -1
        self.denied = 0
        self.read_count = 0
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        i = int(i)
        if i > self.allowed:
            self.denied += 1
            raise RuntimeError('CAUSAL_FUTURE_READ')
        self.highest_read = max(self.highest_read, i)
        self.read_count += 1
        x = np.array(Image.fromarray(self.rgb[i]).convert('RGB').resize((self.size, self.size))).astype(np.float32) / 255.0
        return (torch.from_numpy(x).permute(2, 0, 1) - self.mean) / self.std

def clean(mask, dep, robot):
    return np.asarray(mask, dtype=bool) & ~robot.astype(bool) & np.isfinite(dep) & (dep > 0)

def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return float(inter / max(1, np.logical_or(a, b).sum()))

def bbox(m):
    (y, x) = np.nonzero(m)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if len(x) else [0, 0, 0, 0]

def safe_failure(e):
    return {'time': now(), 'type': type(e).__name__, 'frames': [{'file': Path(f.filename).name, 'line': f.lineno, 'function': f.name} for f in traceback.extract_tb(e.__traceback__)]}

def run(args):
    out = BASE / 'runs/v002/preflight/P1_interaction_extended_v1'
    if args.smoke:
        out = out / 'smoke_segmentation'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'segmentation').mkdir(exist_ok=True)
    protocol = dict(PROTOCOL)
    protocol['source_sha256'] = sha(__file__)
    protocol['model_sha256'] = sha(BASE / 'dependencies/sam21_tiny.pt')
    write_json(out / 'segmentation_protocol.json', {'time': now(), **protocol})
    model = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml', str(BASE / 'dependencies/sam21_tiny.pt'), device='cuda', apply_postprocessing=True)
    model.eval()
    amg = SAM2AutomaticMaskGenerator(model, points_per_side=32, points_per_batch=64)
    summaries = []
    started = time.time()
    selected = [EPISODES[args.episode]] if args.episode is not None else EPISODES
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for ep in selected:
            dst = out / 'segmentation' / (ep + '.h5')
            if dst.exists():
                with h5py.File(dst, 'r') as old:
                    if old.attrs.get('complete', False) and old.attrs.get('source_sha256', '') == protocol['source_sha256']:
                        summaries.append({'episode': ep, 'reused_complete': True, 'frames': int(old.attrs['frames'])})
                        continue
                raise RuntimeError('OUTPUT_EXISTS_NOT_COMPLETE_OR_SOURCE_CHANGED')
            rgbpath = BASE / 'runs/v002/preflight/P1_interaction_extended_v1/rgbd' / (ep + '.h5')
            partial = dst.with_suffix('.partial.h5')
            if partial.exists():
                raise RuntimeError('PARTIAL_EXISTS_REQUIRES_REVIEW')
            input_hash = sha(rgbpath)
            epstart = time.time()
            epstats = []
            with h5py.File(rgbpath, 'r') as src, h5py.File(partial, 'w') as target:
                total = len(src[CAMERAS[0]]['rgb'])
                T = min(total, 2) if args.smoke else total
                target.attrs.update({'episode': ep, 'split': 'exploratory', 'demo': 'demo_0', 'frames': T, 'source_frames': total, 'input_sha256': input_hash, 'source_sha256': protocol['source_sha256'], 'model_sha256': protocol['model_sha256'], 'protocol_json': json.dumps(PROTOCOL), 'complete': False})
                frames_group = target.create_group('frames')
                for (camidx, cam) in enumerate(CAMERAS):
                    lazy = CausalFrames(src[cam]['rgb'], model.image_size)
                    original = vpmod.load_video_frames

                    def loader(**kwargs):
                        return (lazy, src[cam]['rgb'].shape[1], src[cam]['rgb'].shape[2])
                    vpmod.load_video_frames = loader
                    try:
                        state = model.init_state(video_path='causal_h5', offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=False)
                    finally:
                        vpmod.load_video_frames = original
                    records = {}
                    next_id = 1
                    num_auto = 0
                    num_refresh = 0
                    mask_count = 0
                    max_tracks = 0
                    for t in range(T):
                        frame_start = time.time()
                        lazy.allowed = t
                        dep = src[cam]['depth_m'][t]
                        robot = src[cam]['robot_mask'][t]
                        pred = {}
                        if state['obj_ids']:
                            generator = model.propagate_in_video(state, start_frame_idx=t, max_frame_num_to_track=0, reverse=False)
                            for (ft, ids, logits) in generator:
                                assert ft == t
                                raw = (logits[:, 0] > 0).cpu().numpy()
                                pred = {int(k): clean(m, dep, robot) for (k, m) in zip(ids, raw)}
                        for k in records:
                            records[k]['source'] = 'video'
                        det_stats = {'raw': 0, 'valid': 0, 'duplicate': 0, 'discarded_after_robot_depth': 0}
                        if t % PROTOCOL['redetect_every'] == 0:
                            auto = amg.generate(src[cam]['rgb'][t])
                            det_stats['raw'] = len(auto)
                            det = []
                            for a in sorted(auto, key=lambda z: -(float(z['predicted_iou']) * float(z['stability_score']))):
                                m = clean(a['segmentation'], dep, robot)
                                if int(m.sum()) < 16:
                                    det_stats['discarded_after_robot_depth'] += 1
                                    continue
                                if any((iou(m, x['mask']) > PROTOCOL['duplicate_mask_iou'] for x in det)):
                                    det_stats['duplicate'] += 1
                                    continue
                                det.append({'mask': m, 'quality': float(a['predicted_iou']), 'stability': float(a['stability_score']), 'raw_area': int(a['area'])})
                            det_stats['valid'] = len(det)
                            num_auto += len(det)
                            keys = [k for k in pred if pred[k].sum() >= 16]
                            matched = {}
                            if keys and det:
                                mat = np.array([[iou(pred[k], a['mask']) for a in det] for k in keys])
                                (rr, cc) = linear_sum_assignment(1 - mat)
                                for (r, c) in zip(rr, cc):
                                    if mat[r, c] >= PROTOCOL['refresh_mask_iou']:
                                        matched[int(c)] = keys[int(r)]
                            for (j, a) in enumerate(det):
                                if j in matched:
                                    k = matched[j]
                                    num_refresh += 1
                                else:
                                    k = next_id
                                    next_id += 1
                                    records[k] = {'born': t, 'last_visible': t, 'quality': a['quality'], 'stability': a['stability']}
                                records[k].update({'last_detection': t, 'quality': a['quality'], 'stability': a['stability'], 'source': 'refreshed' if j in matched else 'automatic'})
                                model.add_new_mask(state, frame_idx=t, obj_id=k, mask=a['mask'])
                                pred[k] = a['mask']
                            if state['obj_ids']:
                                for (ft, ids, logits) in model.propagate_in_video(state, start_frame_idx=t, max_frame_num_to_track=0, reverse=False):
                                    assert ft == t
                                    masks = (logits[:, 0] > 0).cpu().numpy()
                                    for (k, m) in zip(ids, masks):
                                        pred[int(k)] = clean(m, dep, robot)
                        cg = frames_group.require_group(str(t)).create_group(str(camidx))
                        cg.attrs.update({'camera': cam, 'frame_index': t, 'detection_stats_json': json.dumps(det_stats), 'causal_highest_rgb_frame_read': lazy.highest_read})
                        og = cg.create_group('objects')
                        retired = []
                        for k in list(records):
                            m = pred.get(k, np.zeros(dep.shape, dtype=bool))
                            area = int(m.sum())
                            r = records[k]
                            if area >= 16:
                                r['last_visible'] = t
                                idx = state['obj_id_to_idx'][k]
                                od = state['output_dict_per_obj'][idx]
                                current = od['cond_frame_outputs'].get(t, od['non_cond_frame_outputs'].get(t, {}))
                                presence = current.get('object_score_logits')
                                presence = float(presence.detach().float().reshape(-1)[0].cpu()) if presence is not None else float('nan')
                                g = og.create_group(str(k))
                                g.create_dataset('mask', data=np.packbits(m.reshape(-1)), compression='gzip', compression_opts=1)
                                g.attrs.update({'local_id': k, 'area': area, 'bbox': bbox(m), 'sam_quality_at_last_detection': r['quality'], 'sam_stability_at_last_detection': r['stability'], 'source': r['source'], 'current_frame_redetected': bool(r['last_detection'] == t), 'sam_presence_logit_current': presence, 'born': r['born'], 'last_detection': r['last_detection'], 'large_area_background_candidate': bool(area > 0.45 * m.size)})
                                mask_count += 1
                            elif t - r['last_visible'] >= PROTOCOL['retire_empty_frames']:
                                retired.append(k)
                        for k in retired:
                            model.remove_object(state, obj_id=k, strict=True, need_output=False)
                            records.pop(k)
                        max_tracks = max(max_tracks, len(records))
                        cg.attrs['empty_output'] = bool(len(og) == 0)
                        cg.attrs['active_track_count'] = len(records)
                        cg.attrs['retired_ids_json'] = json.dumps(retired)
                        target.flush()
                        write_json(out / 'segmentation_progress.json', {'time': now(), 'episode': ep, 'camera': cam, 'frame': t, 'frames': T, 'candidate_count': len(og), 'active_tracks': len(records), 'frame_seconds': time.time() - frame_start, 'elapsed_seconds': time.time() - started, 'causal_denied': lazy.denied, 'highest_rgb_read': lazy.highest_read})
                    epstats.append({'camera': cam, 'frames': T, 'mask_instances': mask_count, 'automatic_masks': num_auto, 'refreshes': num_refresh, 'max_active_tracks': max_tracks, 'rgb_read_count': lazy.read_count, 'highest_rgb_read': lazy.highest_read, 'causal_denied': lazy.denied})
                    del state, lazy
                    torch.cuda.empty_cache()
                target.attrs['complete'] = True
            partial.replace(dst)
            row = {'episode': ep, 'path': str(dst), 'frames': T, 'seconds': time.time() - epstart, 'cameras': epstats, 'input_sha256': input_hash, 'output_sha256': sha(dst)}
            summaries.append(row)
            write_json(out / 'segmentation' / f'{ep}.complete.json', {'time': now(), **row})
            write_json(out / 'segmentation_episode_progress.json', {'time': now(), 'episodes': summaries})
        write_json(out / 'segmentation_complete.json', {'time': now(), 'smoke': args.smoke, 'seconds': time.time() - started, 'episodes': summaries, 'source_sha256': protocol['source_sha256']})
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--episode', type=int, choices=range(3))
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as e:
        out = BASE / 'runs/v002/preflight/P1_interaction_extended_v1'
        if args.smoke:
            out = out / 'smoke_segmentation'
        write_json(out / 'segmentation_failure.json', safe_failure(e))
        print(json.dumps(safe_failure(e)))
        sys.exit(1)
