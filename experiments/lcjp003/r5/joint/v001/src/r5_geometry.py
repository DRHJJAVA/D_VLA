import torch
from torch.nn import functional as F
def fps(x, n=64):
    chosen = []
    distance = torch.full((len(x),), float('inf'), device=x.device)
    i = torch.argmax(((x - x.mean(0)) ** 2).sum(-1))
    for _ in range(n):
        chosen.append(i)
        distance = torch.minimum(distance, ((x - x[i]) ** 2).sum(-1))
        i = distance.argmax()
    return torch.stack(chosen)

def points_from_depth(d, prob, teacher_mask=None):
    with torch.no_grad():
        z = d['depth']
        dev = z.device
        (yy, xx) = torch.meshgrid(torch.arange(256, device=dev), torch.arange(256, device=dev), indexing='ij')
        k = d['K']
        p = torch.stack([(xx[None] + 0.375 - k[:, 0, 2, None, None]) * z / k[:, 0, 0, None, None], (yy[None] + 0.875 - k[:, 1, 2, None, None]) * z / k[:, 1, 1, None, None], z], -1)
        world = torch.einsum('vij,vhwj->vhwi', d['T'][:, :3, :3], p) + d['T'][:, :3, 3, None, None].permute(0, 2, 3, 1)
        dx = torch.roll(world, -1, 2) - torch.roll(world, 1, 2)
        dy = torch.roll(world, -1, 1) - torch.roll(world, 1, 1)
        normal = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        eligible = torch.isfinite(world).all(-1) & (z > 0) & ~d['robot']
        mask = (prob.detach() > 0.5) & eligible
        if teacher_mask is not None:
            mask |= (teacher_mask > 0.5) & eligible
        flat = torch.nonzero(mask.flatten()).flatten()
        count = len(flat)
        if count:
            ids = flat[torch.linspace(0, count - 1, min(count, 2048), device=dev).long()]
            valid = torch.arange(2048, device=dev) < len(ids)
            ids = ids[torch.arange(2048, device=dev) % len(ids)]
        else:
            ids = torch.zeros(2048, dtype=torch.long, device=dev)
            valid = torch.zeros(2048, dtype=torch.bool, device=dev)
        xyz = world.reshape(-1, 3)[ids]
        norm = normal.reshape(-1, 3)[ids]
        xyz = torch.nan_to_num(xyz)
        norm = torch.nan_to_num(norm)
        rel = (xyz - d['ee']) @ d['rotation']
        normrel = norm @ d['rotation']
        extent = xyz[valid].amax(0) - xyz[valid].amin(0) if valid.any() else xyz.new_ones(3)
        center = xyz[valid].mean(0) if valid.any() else d['ee']
        unit = (xyz - center) / extent.clamp_min(0.0001)
        rgb = d['rgb'].reshape(-1, 3)[ids].float() / 255.0
        geom = torch.cat([rel, unit, normrel, rgb], -1) * valid[:, None]
        cam = ids // 65536
        pix = ids % 65536
        uv = torch.stack([pix % 256, pix // 256], -1)
        anchors = fps(rel, 64)
        dist = torch.cdist(rel[anchors], rel)
        (dd, kn) = dist.topk(32, largest=False)
        radius = torch.linalg.vector_norm(extent).clamp(0.016, 1.0) / 8
        neighbor = (dd <= radius) & valid[kn] & ((norm[anchors, None] * norm[kn]).sum(-1) > 0.25)
        neighbor[:, 0] = valid[anchors]
        (di, inter) = torch.cdist(rel, rel[anchors]).topk(3, largest=False)
        iw = 1 / (di + 1e-06)
        iw = iw * ((norm[:, None] * norm[anchors][inter]).sum(-1) > 0.25) * (di <= radius)
        iw = iw / iw.sum(-1, keepdim=True).clamp_min(1e-08)
    return {'xyz': xyz, 'rel': rel, 'normal': norm, 'geom': geom, 'ids': ids, 'cam': cam, 'uv': uv, 'valid': valid, 'anchors': anchors, 'knn': kn, 'neighbor': neighbor, 'interp': inter, 'iw': iw, 'extent': extent, 'point_count': count}