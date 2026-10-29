import pathlib, json, datetime, hashlib, importlib.util, random, traceback
import numpy as np, torch
ROOT = pathlib.Path('/mnt/nvme1/drh/D_VLA')
BASE = ROOT / 'experiments/contact_frames'
R = BASE / 'cf009/persistent_evidence'

def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')

def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def module(p, name):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

def main():
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.15)
    cfg = json.loads((R / 'config.json').read_text())
    net = module(R / 'src/model.py', 'model8')
    control = module(BASE / 'cf005/src/model_all_train_frames.py', 'model5')
    old = module(BASE / 'cf003/src/train_and_evaluate.py', 'context3')
    norm = json.loads((BASE / 'cf003/normalization.json').read_text())
    mean = np.asarray(norm['mean'], np.float32)
    std = np.asarray(norm['std'], np.float32)
    lock = {}
    data = {}
    contexts = {}
    tensors = {}
    for ep in cfg['train_episodes']:
        p = R / 'data' / ep / 'artifacts/dataset.npz'
        lock[str(p)] = sha(p)
        d = dict(np.load(p))
        data[ep] = d
        contexts[ep] = old.context(d)
        tensors[ep] = {k: torch.as_tensor(d[k], device='cuda') for k in ['images', 'points', 'target', 'weight', 'near_target', 'near_weight', 'dy', 'dynvalid']}
        tensors[ep]['states'] = torch.as_tensor((d['states'] - mean) / std, device='cuda')
    for seed in cfg['seeds']:
        for p in [R / 'plans' / ('seed_' + str(seed) + '.json'), BASE / 'cf005/runs/all_train_frames' / ('seed_' + str(seed)) / 'initial.pt']:
            lock[str(p)] = sha(p)
    for p in [R / 'config.json', R / 'src/model.py', R / 'src/train.py', BASE / 'cf003/normalization.json']:
        lock[str(p)] = sha(p)
    save(R / 'training_inputs_locked.json', lock)
    pre = []
    for seed in cfg['seeds']:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = net.ContactFrameNet(50).cuda()
        ck = torch.load(BASE / 'cf005/runs/all_train_frames' / ('seed_' + str(seed)) / 'initial.pt', map_location='cpu', weights_only=False)
        missing = model.load_state_dict(ck['model'], strict=False)
        assert set(missing.missing_keys) == {'interaction.0.weight', 'interaction.0.bias', 'interaction.2.weight', 'interaction.2.bias'} and (not missing.unexpected_keys)
        base = control.ContactFrameNet(50).cuda().eval()
        base.load_state_dict(ck['model'])
        model.eval()
        ep = cfg['train_episodes'][0]
        dt = tensors[ep]
        ix = torch.as_tensor(contexts[ep][[5, 20, 60]], device='cuda')
        with torch.no_grad():
            (e, aux, near) = model(dt['images'][ix], dt['points'][ix], dt['states'][ix])
            (oldnear, oldaux) = base(dt['images'][ix], dt['points'][ix], dt['states'][ix])
            assert torch.equal(near, oldnear) and torch.equal(aux, oldaux)
        del base
        dest = R / 'runs/evidence' / ('seed_' + str(seed))
        dest.mkdir(parents=True)
        torch.save({'model': model.state_dict(), 'seed': seed, 'shared_initial_hash': ck['initial_tensor_hash']}, dest / 'initial.pt')
        pre.append({'seed': seed, 'shared_initial_hash': ck['initial_tensor_hash'], 'original_outputs_equal': True, 'parameters': sum((x.numel() for x in model.parameters()))})
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        plan = json.loads((R / 'plans' / ('seed_' + str(seed) + '.json')).read_text())
        assert len(plan) == 600
        hist = []
        for (step, item) in enumerate(plan, 1):
            ep = item['episode']
            ids = np.asarray(item['frames'])
            ix = torch.as_tensor(contexts[ep][ids], device='cuda')
            dt = tensors[ep]
            (e, aux, near) = model(dt['images'][ix], dt['points'][ix], dt['states'][ix])
            yy = dt['target'][ids]
            ww = dt['weight'][ids]
            a = (torch.nn.functional.binary_cross_entropy_with_logits(e, yy, reduction='none') * ww).mean()
            b = (torch.nn.functional.binary_cross_entropy_with_logits(near, dt['near_target'][ids], reduction='none') * dt['near_weight'][ids]).mean()
            c = (torch.nn.functional.smooth_l1_loss(aux, dt['dy'][ids], reduction='none').mean(-1) * dt['dynvalid'][ids]).mean()
            loss = a + 0.5 * b + 0.2 * c
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            hist.append({'step': step, 'episode': ep, 'evidence_bce': float(a.detach()), 'proximity_bce': float(b.detach()), 'aux': float(c.detach()), 'loss': float(loss.detach())})
            if step % 50 == 0:
                save(dest / 'history.json', hist)
                save(R / 'stage_state.json', {'stage': 'training', 'seed': seed, 'step': step, 'time': now()})
                print(json.dumps({'seed': seed, 'step': step}), flush=True)
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'seed': seed, 'steps': 600, 'state_mean': mean, 'state_std': std, 'config': cfg, 'shared_initial_hash': ck['initial_tensor_hash'], 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, dest / 'last.pt')
        del model, opt
        torch.cuda.empty_cache()
    save(R / 'audits/preflight.json', pre)
    save(R / 'evaluation_weights_locked.json', {str(p): sha(p) for p in R.glob('runs/*/seed_*/last.pt')})
    save(R / 'pre_review_freeze.json', {'time': now(), 'training_complete': True, 'supervision_sha': sha(R / 'train_supervision_summary.json'), 'config_sha': sha(R / 'config.json')})
    del tensors
    torch.cuda.empty_cache()
    rows = []
    for ep in cfg['train_episodes'] + cfg['fixed_review_episodes']:
        train = ep in data
        src = R / 'data' / ep / 'artifacts/dataset.npz' if train else BASE / 'cf005_frozen_review_003' / ep / 'artifacts/dataset.npz' if ep in ['tomato_sauce_demo_3', 'alphabet_soup_demo_1'] else BASE / 'cf005/evaluation_all_train_frames' / ep / 'artifacts/dataset.npz'
        d = dict(np.load(src))
        n = len(d['states'])
        if train:
            ix = old.context(d)
        else:
            predsrc = BASE / 'cf005_frozen_review_003' / ep / 'artifacts/frame_predictions.npz' if ep in ['tomato_sauce_demo_3', 'alphabet_soup_demo_1'] else BASE / 'cf005/evaluation_all_train_frames' / ep / 'artifacts/frame_predictions.npz'
            ix = np.load(predsrc)['context_indices']
        probs = []
        nearprobs = []
        states = (d['states'] - mean) / std
        for seed in cfg['seeds']:
            ck = torch.load(R / 'runs/evidence' / ('seed_' + str(seed)) / 'last.pt', map_location='cpu', weights_only=False)
            m = net.ContactFrameNet(50).cuda().eval()
            m.load_state_dict(ck['model'])
            pr = []
            nr = []
            with torch.no_grad():
                for t in range(0, n, 16):
                    k = ix[t:t + 16]
                    (e, aux, near) = m(torch.as_tensor(d['images'][k], device='cuda'), torch.as_tensor(d['points'][k], device='cuda'), torch.as_tensor(states[k], device='cuda'))
                    pr.append(torch.sigmoid(e).cpu().numpy())
                    nr.append(torch.sigmoid(near).cpu().numpy())
            probs.append(np.concatenate(pr))
            nearprobs.append(np.concatenate(nr))
            del m
            torch.cuda.empty_cache()
        probs = np.stack(probs)
        nearprobs = np.stack(nearprobs)
        rel = d['geometry_reliable'] & d['state_valid']
        sep = rel & (d['distance'] - 2 * d['sigma'] > 0.006)
        pos = (probs.min(0) >= 0.8) & rel & ~sep
        neg = (nearprobs.max(0) <= 0.2) & sep
        unknown = ~(pos | neg)
        out = R / 'evaluation' / ep / 'artifacts'
        out.mkdir(parents=True)
        (out / 'dataset.npz').symlink_to(src.resolve())
        np.savez_compressed(out / 'frame_predictions.npz', probabilities=probs, proximity_probabilities=nearprobs, mean=probs.mean(0), low=probs.min(0), high=probs.max(0), contact_candidate=pos, observed_separation_candidate=neg, unknown=unknown, context_indices=ix, geometry_separation_evidence=sep)
        rows.append({'episode': ep, 'training': train, 'frames': n, 'candidate': int(pos.sum()), 'separation': int(neg.sum()), 'unknown': int(unknown.sum()), 'raw_all_high': int((probs.min(0) >= 0.8).sum())})
    assert all((sha(pathlib.Path(p)) == h for (p, h) in lock.items()))
    save(R / 'summary.json', {'rows': rows, 'time': now(), 'score_semantics': 'learned interaction evidence support, not calibrated physical contact probability'})
    save(R / 'stage_state.json', {'stage': 'training_evaluation_complete', 'time': now()})
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        (R / 'failure.log').write_text(traceback.format_exc())
        save(R / 'failure.json', {'type': type(e).__name__, 'time': now()})
        save(R / 'stage_state.json', {'stage': 'failed', 'type': type(e).__name__, 'time': now()})
        raise SystemExit(1)