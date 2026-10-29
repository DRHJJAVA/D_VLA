from common14 import *
import numpy as np, torch, random, traceback
from torch.nn import functional as F
from model14 import ContactMotionNet
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.22)
DEVICE = 'cuda'

def load_data():
    norm = json.loads((OLD.parent / 'cf003/normalization.json').read_text())
    mn = json.loads((BASE / 'motion_normalization.json').read_text())
    data = {}
    inputs = [OLD.parent / 'cf003/normalization.json', BASE / 'motion_normalization.json']
    stats = []
    for ep in CFG['train_episodes']:
        p = dataset(ep)
        fp = BASE / 'features' / f'{ep}.npz'
        gp = OLD / 'evaluation' / ep / 'artifacts/frame_predictions.npz'
        inputs.extend([p, fp, gp])
        with np.load(p) as d, np.load(fp) as v, np.load(gp) as g:
            sep = g['geometry_separation_evidence'].astype(bool)
            w = d['weight'] * d['interaction_support'] * d['near_target']
            w = np.where(sep, d['near_weight'], w).astype('float32')
            y = np.where(sep, 0, 1).astype('float32')
            mo = (v['features'] - np.asarray(mn['mean'])) / np.asarray(mn['std'])
            mo[~v['valid']] = 0
            states = (d['states'] - np.asarray(norm['mean'])) / np.asarray(norm['std'])
            arrays = dict(images=d['images'], points=d['points'], states=states, motion=mo, target=y, weight=w, near_target=d['near_target'], near_weight=d['near_weight'], dy=d['dy'], dynvalid=d['dynvalid'].astype('float32'))
            assert all((np.isfinite(a).all() for a in arrays.values()))
            data[ep] = {k: torch.tensor(a, dtype=torch.float32, device=DEVICE) for (k, a) in arrays.items()}
            out = BASE / 'features' / f'{ep}_supervision.npz'
            assert not out.exists()
            np.savez_compressed(out, target=y, weight=w, separation=sep, positive_weight=np.where(sep, 0, w), unknown=w == 0)
            stats.append(dict(episode=ep, frames=len(w), positive=int(((y == 1) & (w > 0)).sum()), negative=int(((y == 0) & (w > 0)).sum()), unknown=int((w == 0).sum()), positive_mass=float(w[y == 1].sum()), negative_mass=float(w[y == 0].sum())))
    for seed in CFG['seeds']:
        inputs.extend([OLD / 'plans' / f'seed_{seed}.json', OLD / 'runs/full_history' / f'seed_{seed}' / 'initial.pt'])
    inputs.extend((BASE / 'src').glob('*.py'))
    inputs.append(BASE / 'config.json')
    save(BASE / 'training_input_locked.json', lock(inputs))
    save(BASE / 'audits/supervision.json', dict(rows=stats, manual_ranges_read=False))
    return (data, norm)

def forward(net, d):
    return net(d['images'][None], d['points'][None], d['states'][None], d['motion'][None])

def losses(out, d, ids):
    (logits, dy, near) = out
    main = (F.binary_cross_entropy_with_logits(logits[0, ids], d['target'][ids], reduction='none') * d['weight'][ids]).mean()
    prox = (F.binary_cross_entropy_with_logits(near[0, ids], d['near_target'][ids], reduction='none') * d['near_weight'][ids]).mean()
    dyn = (F.smooth_l1_loss(dy[0, ids], d['dy'][ids], reduction='none').mean(-1) * d['dynvalid'][ids]).mean()
    return (main + 0.5 * prox + 0.2 * dyn, main, prox, dyn)

def main():
    stage('training_preflight')
    (data, norm) = load_data()
    ep = CFG['train_episodes'][0]
    d = data[ep]
    torch.manual_seed(2026)
    a = ContactMotionNet(False).to(DEVICE)
    b = ContactMotionNet(True).to(DEVICE)
    initial = torch.load(OLD / 'runs/full_history/seed_2026/initial.pt', map_location=DEVICE, weights_only=False)
    a.base.load_state_dict(initial['model'])
    b.load_state_dict(a.state_dict())
    a.eval()
    b.eval()
    with torch.no_grad():
        oa = forward(a, d)
        ob = forward(b, d)
        initdiff = max((float((x - y).abs().max()) for (x, y) in zip(oa, ob)))
        dd = dict(d)
        dd['motion'] = torch.randn_like(d['motion']) * 100
        invariant = max((float((x - y).abs().max()) for (x, y) in zip(oa, forward(a, dd))))
        base = a.base(d['images'][None], d['points'][None], d['states'][None]) if False else None
        prefix = {k: v[:40] for (k, v) in d.items()}
        op = forward(b, prefix)
        prefixdiff = max((float((x[:, :40] - y).abs().max()) for (x, y) in zip(ob, op)))
    logits = torch.randn_like(d['weight'], requires_grad=True)
    (F.binary_cross_entropy_with_logits(logits, d['target'], reduction='none') * d['weight']).mean().backward()
    unknown_grad = float(logits.grad[d['weight'] == 0].abs().max())
    assert initdiff == 0 and invariant == 0 and (prefixdiff < 1e-05) and (unknown_grad == 0)
    save(BASE / 'audits/preflight.json', dict(initial_arm_max_difference=initdiff, zero_arm_invariance=invariant, temporal_prefix_difference=prefixdiff, unknown_main_gradient=unknown_grad, parameter_count=sum((p.numel() for p in a.parameters())), manual_read=False))
    del a, b, oa, ob, op
    torch.cuda.empty_cache()
    for seed in CFG['seeds']:
        plan = json.loads((OLD / 'plans' / f'seed_{seed}.json').read_text())
        assert len(plan) == 600 and all((x['episode'] in data and len(x['frames']) == 16 for x in plan))
        for arm in CFG['arms']:
            run = BASE / 'runs' / arm / f'seed_{seed}'
            assert not run.exists()
            run.mkdir(parents=True)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            net = ContactMotionNet(arm == 'with_motion').to(DEVICE)
            init = torch.load(OLD / 'runs/full_history' / f'seed_{seed}' / 'initial.pt', map_location=DEVICE, weights_only=False)
            net.base.load_state_dict(init['model'])
            assert all((torch.equal(v, init['model'][k]) for (k, v) in net.base.state_dict().items()))
            torch.save(dict(model=net.state_dict(), seed=seed), run / 'initial.pt')
            opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.0001)
            hist = []
            net.train()
            for (step, item) in enumerate(plan, 1):
                d = data[item['episode']]
                ids = torch.tensor(item['frames'], device=DEVICE)
                opt.zero_grad(set_to_none=True)
                (total, mainloss, prox, dyn) = losses(forward(net, d), d, ids)
                assert torch.isfinite(total)
                total.backward()
                grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                assert torch.isfinite(grad)
                opt.step()
                hist.append(dict(step=step, episode=item['episode'], loss=float(total.detach()), main=float(mainloss.detach()), near=float(prox.detach()), dynamics=float(dyn.detach())))
                if step % 100 == 0:
                    save(run / 'history.json', hist)
                    save(BASE / 'stage_state.json', dict(stage='training', arm=arm, seed=seed, step=step, time=now()))
                    print(json.dumps(dict(arm=arm, seed=seed, step=step, loss=hist[-1]['loss'])), flush=True)
            torch.save(dict(model=net.state_dict(), optimizer=opt.state_dict(), seed=seed, steps=600, state_mean=norm['mean'], state_std=norm['std'], torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), config=CFG), run / 'last.pt')
            save(run / 'history.json', hist)
            del net, opt
            torch.cuda.empty_cache()
    assert verify(json.loads((BASE / 'training_input_locked.json').read_text())) is not False
    checkpoints = list((BASE / 'runs').glob('*/*/last.pt'))
    assert len(checkpoints) == 3
    save(BASE / 'weights_frozen.json', dict(time=now(), checkpoints=lock(checkpoints), selection='all seeds last step 600; no review selection', manual_ranges_read=False))
    stage('weights_frozen')
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(BASE / 'failure_train.json', dict(time=now(), type=type(e).__name__, message=str(e), stack=[dict(file=x.filename, line=x.lineno, function=x.name) for x in traceback.extract_tb(e.__traceback__)]))
        stage('training_failed')
        raise
