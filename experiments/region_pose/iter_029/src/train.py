from pathlib import Path
import json, sys, time, random, hashlib
import numpy as np, torch
from torch.nn import functional as F
from model import RegionPoseNet
O = Path(__file__).resolve().parents[1]

def loadrows(manifest, device):
    rows = []
    for r in manifest['rows']:
        with np.load(O / 'samples' / (r['key'] + '.npz')) as z:
            d = {k: torch.as_tensor(z[k], dtype=torch.float32, device=device) for k in ['xyz', 'dino_feat', 'pose', 'hand_R', 'hand_t', 'target', 'confidence']}
            d['known'] = torch.as_tensor(z['known'], dtype=torch.bool, device=device)
        with np.load(O / 'graphs' / (r['key'] + '.npz')) as z:
            d['graph'] = {k: torch.as_tensor(z[k], dtype=torch.float32 if k.endswith('weight') else torch.long, device=device) for k in z.files}
        rows.append((r, d))
    return rows

def forward(net, d):
    return net(d['xyz'], d['dino_feat'], d['graph'], d['pose'], d['hand_R'], d['hand_t'])

def loss(logit, d):
    valid = d['known']
    pos = valid & (d['target'] > 0.5)
    neg = valid & ~pos
    b = F.binary_cross_entropy_with_logits(logit, d['target'], reduction='none')
    cw = d['confidence']
    p = (b[pos] * cw[pos]).sum() / cw[pos].sum().clamp_min(1e-08)
    n = (b[neg] * cw[neg]).sum() / cw[neg].sum().clamp_min(1e-08)
    ip = torch.flatnonzero(pos) if hasattr(torch, 'flatnonzero') else torch.nonzero(pos).flatten()
    ix = torch.nonzero(neg).flatten()
    ip = ip[torch.linspace(0, len(ip) - 1, min(64, len(ip)), device=ip.device).long()]
    ix = ix[torch.linspace(0, len(ix) - 1, min(64, len(ix)), device=ix.device).long()]
    rank = F.softplus(logit[ix][None, :] - logit[ip][:, None]).mean()
    return 0.5 * (p + n) + 0.1 * rank

def metrics(score, d):
    y = d['target'].detach().cpu().numpy().astype(bool)
    known = d['known'].cpu().numpy()
    score = np.asarray(score)
    ids = np.flatnonzero(known)
    order = ids[np.argsort(-score[ids], kind='stable')]
    yy = y[order]
    ap = float((np.cumsum(yy) / np.arange(1, len(yy) + 1) * yy).sum() / max(1, yy.sum()))
    out = dict(ap=ap, valid_fraction=float(known.mean()), positive_points=int(y.sum()))
    for b in [0.1, 0.2, 0.3]:
        k = max(1, int(np.ceil(b * len(ids))))
        out['coverage' + str(int(b * 100))] = float(y[order[:k]].sum() / max(1, y.sum()))
        out['random' + str(int(b * 100))] = k / len(ids)
    for b in [0.1, 0.2, 0.3]:
        count = max(1, int(np.ceil(b * len(score))))
        ix = np.argsort(-score, kind='stable')[:count]
        suffix = str(int(b * 100))
        out['all_coverage' + suffix] = float(y[ix].sum() / max(1, y.sum()))
        out['all_unknown_fraction' + suffix] = float((~known[ix]).mean())
        out['all_known_negative_fraction' + suffix] = float((known[ix] & ~y[ix]).mean())
    return out

def evaluate(net, rows, variant):
    net.eval()
    out = []
    with torch.inference_mode():
        for (r, d) in rows:
            (a, b) = forward(net, d)
            s = a if variant == 'object_only' else b
            m = metrics(s.cpu().numpy(), d)
            m.update(key=r['key'], episode=r['episode'], event=r['event'], offset=r['offset'], loss=float(loss(s, d)))
            out.append(m)
    keys = ['loss', 'ap', 'coverage10', 'coverage20', 'coverage30', 'random10', 'random20', 'random30', 'valid_fraction']
    episodes = []
    for ep in sorted({r['episode'] for r in out}):
        events = []
        for ev in sorted({r['event'] for r in out if r['episode'] == ep}):
            rr = [r for r in out if r['event'] == ev]
            events.append({k: float(np.mean([r[k] for r in rr])) for k in keys})
        episodes.append(dict(episode=ep, **{k: float(np.mean([r[k] for r in events])) for k in keys}))
    return dict(**{k: float(np.mean([r[k] for r in episodes])) for k in keys}, episodes=episodes, rows=out, offsets=[dict(offset=o, **{k: float(np.mean([r[k] for r in out if r['offset'] == o])) for k in keys}) for o in [15, 10, 5, 1]])

def atomicjson(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)

def main():
    torch.set_num_threads(4)
    cfg = json.loads((O / 'config.json').read_text())
    manifest = json.loads((O / 'manifest.json').read_text())
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(0.15)
    rows = loadrows(manifest, device)
    tr = [r for r in rows if r[0]['split'] == 'train']
    va = [r for r in rows if r[0]['split'] == 'val']
    assert {r[0]['episode'] for r in tr}.isdisjoint({r[0]['episode'] for r in va})
    sourcefiles = list((O / 'src').glob('*.py')) + [O / 'config.json', O / 'manifest.json', O / '设计方案.md'] + list((O / 'samples').glob('*.npz')) + list((O / 'graphs').glob('*.npz')) + (list((O / 'plans').glob('*.json')) + list((O / 'initialization').glob('*')))
    lock = {str(p.relative_to(O)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcefiles}
    if (O / 'training_inputs_locked.json').exists():
        assert json.loads((O / 'training_inputs_locked.json').read_text()) == lock
    else:
        (O / 'training_inputs_locked.json').write_text(json.dumps(lock, indent=2))
    torch.manual_seed(cfg['seed'])
    net = RegionPoseNet().to(device)
    (r, d) = tr[0]
    (a, b) = forward(net, d)
    altered = dict(d, hand_t=d['hand_t'] + 0.02, pose=d['pose'] + 0.1)
    (a2, b2) = forward(net, altered)
    assert a.shape == d['target'].shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.allclose(a, a2) and (not torch.allclose(b, b2))
    l = loss(a, d) + loss(b, d)
    l.backward()
    assert torch.isfinite(l) and net.conditional[0].weight.grad.abs().sum() > 0
    (O / 'audits/model_interface.json').write_text(json.dumps(dict(output_shape=list(a.shape), base_pose_invariant=True, conditional_geometry_sensitive=True, finite_gradient=True)))
    del net
    torch.cuda.empty_cache()
    allsummary = []
    for run in cfg['runs']:
        tr = [r for r in rows if r[0]['split'] == 'train' and (run['group'] == 'expanded10' or r[0]['episode'] in cfg['control_episodes'])]
        assert len({r[0]['event'] for r in tr}) == run['events']
        variant = run['variant']
        group = run['group']
        run_name = group + '_' + variant
        out = O / 'runs' / run_name
        out.mkdir(exist_ok=True)
        if (out / 'done.json').exists():
            allsummary.append(json.loads((out / 'done.json').read_text()))
            continue
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        net = RegionPoseNet().to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        init_hash = hashlib.sha256(b''.join((v.detach().cpu().numpy().tobytes() for v in net.state_dict().values()))).hexdigest()
        assert init_hash == json.loads((O / 'initialization/locked.json').read_text())['state_sha256']
        (out / 'initialization.json').write_text(json.dumps({'state_sha256': init_hash, 'seed': cfg['seed']}))
        history = []
        start = 1
        best = float('inf')
        elapsed = 0
        if (out / 'last.pt').exists():
            ck = torch.load(out / 'last.pt', weights_only=False, map_location=device)
            assert ck['input_lock'] == lock
            net.load_state_dict(ck['state'])
            opt.load_state_dict(ck['optimizer'])
            history = ck['history']
            start = ck['epoch'] + 1
            best = ck['best']
            random.setstate(ck['rng_python'])
            np.random.set_state(ck['rng_numpy'])
            torch.set_rng_state(ck['rng_cpu'].cpu())
            torch.cuda.set_rng_state(ck['rng_cuda'].cpu())
            elapsed = history[-1]['elapsed_s']
        events = sorted({r[0]['event'] for r in tr})
        clock = time.monotonic()
        for epoch in range(start, run['epochs'] + 1):
            net.train()
            order = events.copy()
            random.shuffle(order)
            trainloss = []
            selected_queries = []
            for ev in order:
                offset = 1 if group == 'late' else [15, 10, 5, 1][(epoch - 1 + events.index(ev)) % 4]
                items = [r for r in tr if r[0]['event'] == ev and r[0]['offset'] == offset]
                assert len(items) == 1
                selected_queries.append({'event': ev, 'offset': offset, 'key': items[0][0]['key']})
                opt.zero_grad(set_to_none=True)
                ls = []
                for (r, d) in items:
                    (a, b) = forward(net, d)
                    v = loss(a, d) if variant == 'object_only' else loss(b, d) + 0.5 * loss(a, d)
                    (v / len(items)).backward()
                    ls.append(float(v.detach()))
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                trainloss.append(np.mean(ls))
            val = evaluate(net, va, variant)
            row = dict(variant=variant, group=group, run=run_name, selected_queries=selected_queries, optimizer_steps=len(events) * epoch, epoch=epoch, train_loss=float(np.mean(trainloss)), val_loss=val['loss'], val_ap=val['ap'], val_coverage20=val['coverage20'], random20=val['random20'], elapsed_s=round(elapsed + time.monotonic() - clock, 2))
            history.append(row)
            atomicjson(out / 'history.json', history)
            print(json.dumps(row), flush=True)
            if val['loss'] < best:
                best = val['loss']
                tmp = out / 'best.tmp.pt'
                torch.save(dict(state=net.state_dict(), epoch=epoch, config=cfg, variant=variant, group=group, val=val, input_lock=lock), tmp)
                tmp.replace(out / 'best.pt')
            tmp = out / 'last.tmp.pt'
            torch.save(dict(state=net.state_dict(), optimizer=opt.state_dict(), epoch=epoch, best=best, history=history, input_lock=lock, rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()), tmp)
            tmp.replace(out / 'last.pt')
        ck = torch.load(out / 'best.pt', weights_only=False, map_location=device)
        net.load_state_dict(ck['state'])
        net.eval()
        val = evaluate(net, va, variant)
        pred = out / 'val_predictions'
        pred.mkdir(exist_ok=True)
        baselines = []
        shuffled = []
        with torch.inference_mode():
            for (k, (r, d)) in enumerate(va):
                (a, b) = forward(net, d)
                score = a if variant == 'object_only' else b
                np.savez_compressed(pred / (r['key'] + '.npz'), xyz=d['xyz'].cpu().numpy(), score=score.sigmoid().cpu().numpy(), base_score=a.sigmoid().cpu().numpy(), target=d['target'].cpu().numpy(), known=d['known'].cpu().numpy())
                center = -torch.linalg.vector_norm(d['xyz'], dim=1).cpu().numpy()
                local = (d['xyz'] - d['hand_t']) @ d['hand_R']
                gap = d['pose'][-1] * 0.1
                lo = local.new_tensor([-0.008, 0, 0.0854])
                hi = local.new_tensor([0.008, 0, 0.1014])
                lo[1] = -gap / 2
                hi[1] = gap / 2
                distance = torch.linalg.vector_norm(torch.maximum(lo - local, torch.zeros_like(local)) + torch.maximum(local - hi, torch.zeros_like(local)), dim=1)
                baselines.append(dict(key=r['key'], center=metrics(center, d), current_gripper_box=metrics(-distance.cpu().numpy(), d)))
                others = []
                if not others:
                    continue
                other = others[0]
                sd = dict(d, pose=other['pose'], hand_R=other['hand_R'], hand_t=other['hand_t'])
                (_, sw) = forward(net, sd)
                shuffled.append(dict(key=r['key'], mean_abs_score_change=float((sw.sigmoid() - b.sigmoid()).abs().mean()), metrics_against_original_event_target=metrics(sw.cpu().numpy(), d)))
        atomicjson(out / 'baselines.json', baselines)
        atomicjson(out / 'pose_swap_probe.json', shuffled)
        done = dict(variant=variant, group=group, run=run_name, initialization_sha256=init_hash, optimizer_steps=run['epochs'] * len(events), status='complete', best_epoch=ck['epoch'], validation=val, train_episodes=len({r[0]['episode'] for r in tr}), val_episodes=len({r[0]['episode'] for r in va}), scope=cfg['scope'], warning='automatic region agreement only, not contact truth, success rate or generalization', checkpoint_sha256=hashlib.sha256((out / 'best.pt').read_bytes()).hexdigest())
        atomicjson(out / 'done.json', done)
        allsummary.append(done)
        print(json.dumps(dict(variant=variant, event='complete', best_epoch=ck['epoch'], val_ap=val['ap'], val_coverage20=val['coverage20'])), flush=True)
        del net, opt
        torch.cuda.empty_cache()
    atomicjson(O / 'summary.json', allsummary)
    print('ALL_TRAINING_COMPLETE', flush=True)
if __name__ == '__main__':
    main()
