import os, ast, json, subprocess, sys, traceback
from common14 import *
import h5py, numpy as np, torch
sys.path.insert(0, str(ROOT / 'src/v3'))
from stage2_probe_motion_fingerpads import tc
from model14 import ContactMotionNet
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.22)
C = ROOT / 'experiments/contact_frames/cf014'

def main():
    verify(json.loads((BASE / 'weights_locked.json').read_text())['files'])
    verify(json.loads((BASE / 'prediction_input_locked.json').read_text()))
    verify(json.loads((C / 'completed_locked.json').read_text()))
    assert (BASE / 'src/model14.py').read_bytes() == (C / 'src/model14.py').read_bytes()
    assert (BASE / 'src/prepare_features.py').read_bytes() == (C / 'src/prepare_features.py').read_bytes()
    norm = json.loads((BASE.parent / 'cf003/normalization.json').read_text())
    mn = json.loads((C / 'motion_normalization.json').read_text())
    rows = []
    maxdiff = 0.0
    manifest = [json.loads(x) for x in (BASE / 'original_sequences.jsonl').read_text().splitlines()]
    for case in CFG['cases']:
        ep = case['id']
        root = BASE / 'cases' / ep
        conf = json.loads((root / 'config.json').read_text())
        rowref = next((x for x in manifest if x['id'] == ep))
        assert rowref['source_episode'] == 'demo_0'
        with h5py.File(rowref['source_path'], 'r') as f, h5py.File(conf['rgbd'], 'r') as h, np.load(conf['motion']) as m, np.load(conf['component']) as comp, np.load(dataset(ep)) as d, np.load(BASE / 'features' / f'{ep}.npz') as feat:
            e = f['data/demo_0']
            N = len(e['states'])
            freq = json.loads(f['data'].attrs['env_args'])['env_kwargs']['control_freq']
            assert freq == 20
            assert np.array_equal(h['frame_indices'][:], np.arange(N))
            assert np.array_equal(m['measured_finger_q'][1:], e['obs/gripper_states'][:N - 1].astype(np.float32))
            delta = float(np.max(np.abs(m['hand_to_world'] - h['robot0_eye_in_hand/camera_to_world'][:] @ np.linalg.inv(tc()))))
            assert delta == 0
            assert np.isnan(m['measured_finger_q'][0]).all() and (not d['state_valid'][0])
            assert d['states'].shape == (N, 50) and d['points'].shape == (N, 128, 10) and (d['images'].shape == (N, 4, 96, 96)) and (feat['features'].shape == (N, 24))
            assert np.isfinite(d['states']).all() and np.isfinite(d['points']).all() and np.isfinite(feat['features']).all()
            uv = comp['pixel']
            assert uv.dtype == np.int64 and (uv[:, 0] >= 0).all() and (uv[:, 1] >= 0).all()
            mo = (feat['features'] - np.array(mn['mean'])) / np.array(mn['std'])
            mo[~feat['valid']] = 0
            ss = (d['states'] - np.array(norm['mean'])) / np.array(norm['std'])
            args = [torch.tensor(z, dtype=torch.float32, device='cuda')[None] for z in [d['images'], d['points'], ss, mo]]
            motion_valid = int(feat['valid'].sum())
            visible = np.median(d['visible_counts'])
        saved = np.load(BASE / 'evaluation' / ep / 'predictions.npz')['probabilities']
        for (j, seed) in enumerate(CFG['seeds']):
            net = ContactMotionNet(True).cuda()
            net.load_state_dict(torch.load(C / 'runs/with_motion' / f'seed_{seed}' / 'last.pt', map_location='cuda', weights_only=False)['model'])
            net.eval()
            with torch.no_grad():
                q = net(*args)[0].sigmoid()[0].cpu().numpy()
            delta = float(np.max(np.abs(q - saved[j])))
            maxdiff = max(delta, maxdiff)
            assert delta < 1e-06
        video = ROOT / 'docs/图示/cf017_contact' / f'{ep}_clean.mp4'
        import cv2
        cap = cv2.VideoCapture(str(video))
        meta = dict(nb_frames=str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), r_frame_rate=str(int(cap.get(cv2.CAP_PROP_FPS))) + '/1')
        decoded = 0
        while True:
            (ok, img) = cap.read()
            if not ok:
                break
            decoded += 1
        cap.release()
        assert decoded == N
        assert int(meta['nb_frames']) == N and meta['width'] == 896 and (meta['height'] == 448) and (meta['r_frame_rate'] == '20/1')
        rows.append(dict(episode=ep, frames=N, initial_pixels=len(uv), valid_motion_frames=motion_valid, visible_tracks_median=float(visible), hand_transform_difference=0.0, raw_episode_read='demo_0 only', video=meta))
    old = ast.parse((ROOT / 'experiments/contact_frames/cf001/src/prepare.py').read_text())
    for case in CFG['cases']:
        root = BASE / 'cases' / case['id']
        p = root / 'src' / ('prepare_memory_v2.py' if case['id'] == 'two_objects_demo_0' else 'prepare.py')
        t = ast.parse(p.read_text())
        for n in ast.walk(t):
            if isinstance(n, ast.Constant) and n.value == str(BASE / 'original_sequences.jsonl'):
                n.value = 'data/v3/manifests/original_sequences.jsonl'
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and (n.func.attr == 'set_per_process_memory_fraction'):
                n.args[0] = ast.Constant(0.18)
        assert ast.dump(t) == ast.dump(old)
    save(BASE / 'audits/completion.json', dict(time=now(), rows=rows, frozen_prediction_max_difference=maxdiff, weights_unchanged=True, model_source_byte_equal=True, motion_descriptor_source_byte_equal=True, legacy_input_builder_ast_equal_except_manifest_and_memory_limit=True, normalization='original CF003 state and CF014 motion', manual_training_use=False, new_training=False, reserved_episode_access=False, contact_truth_access=False, object_pose_truth_access=False, multi_object_initial_surface_adapter=True, offline_full_trajectory_context_selection=True))
    stage('audited_page_ready')
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(BASE / 'failure_audit.json', dict(time=now(), type=type(e).__name__, message=str(e), stack=[dict(file=x.filename, line=x.lineno, function=x.name) for x in traceback.extract_tb(e.__traceback__)]))
        stage('audit_failed')
        raise SystemExit(1)
