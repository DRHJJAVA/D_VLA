import torch
import math

def drift_loss(gen,pos,temperatures=(.02,.05,.2)):
    # Port of the official JAX drift_loss with unit weights and no fixed negatives.
    # Each batch item is one observation condition; no actions from other scenes are positives.
    with torch.no_grad():
        old=gen.detach().float(); targets=torch.cat([old,pos.detach().float()],1)
        g=old.shape[1]; dim=old.shape[-1]
        sq=old.square().sum(-1)[...,None]+targets.square().sum(-1)[:,None]-2*old@targets.transpose(-1,-2)
        dist=sq.clamp_min(1e-8).sqrt()
        scale=dist.mean(); scale_input=(scale/math.sqrt(dim)).clamp_min(1e-3)
        x=old/scale_input; y=targets/scale_input
        dn=dist/scale.clamp_min(1e-3)
        mask=torch.nn.functional.pad(torch.eye(g,device=gen.device),(0,pos.shape[1]))
        dn=dn+100*mask[None]
        force=torch.zeros_like(x)
        for temp in temperatures:
            z=-dn/temp
            affinity=(z.softmax(-1)*z.softmax(-2)).clamp_min(1e-6).sqrt()
            neg,positive=affinity[:,:,:g],affinity[:,:,g:]
            coeff=torch.cat([-neg*positive.sum(-1,keepdim=True),positive*neg.sum(-1,keepdim=True)],-1)
            f=coeff@y-coeff.sum(-1,keepdim=True)*x
            force=force+f/f.square().mean().clamp_min(1e-8).sqrt()
        goal=x+force
    return ((gen/scale_input-goal)**2).mean((-1,-2))
