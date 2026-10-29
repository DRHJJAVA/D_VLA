
"""Conditional action-space Drifting, paper Algorithm 2 and Appendix A.6."""
from __future__ import annotations
import math
import torch
PAPER_URL="https://arxiv.org/html/2602.04770v2"
TEMPERATURES=(0.02,0.05,0.2)

def conditional_drifting_loss(pred, target_actions, action_valid, temperatures=TEMPERATURES):
    """pred [B,G,H,A], target [B,H,A], valid [B,H] or [B,H,A]."""
    if pred.ndim != 4 or pred.shape[1] < 2:
        raise ValueError("Requires at least two generated samples per condition")
    if action_valid.ndim==2:
        action_valid=action_valid.unsqueeze(-1).expand_as(target_actions)
    if target_actions.shape != pred[:,0].shape or action_valid.shape!=target_actions.shape:
        raise ValueError("Action schema mismatch")
    terms=[]; diagnostics=[]
    for b in range(pred.shape[0]):
        mask=action_valid[b].reshape(-1).bool()
        if not bool(mask.any()):
            diagnostics.append({"condition":b,"valid_dimensions":0,"skipped":True})
            continue
        x=pred[b].reshape(pred.shape[1],-1)[:,mask].float()
        y=target_actions[b].reshape(1,-1)[:,mask].float()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise FloatingPointError("Nonfinite valid action")
        with torch.no_grad():
            xd=x.detach(); yd=y.detach()
            raw=torch.cdist(xd,torch.cat([yd,xd],dim=0))
            scale=(raw.mean()/math.sqrt(x.shape[-1])).clamp_min(1e-6)
        xn=x/scale
        with torch.no_grad():
            x0=xd/scale; y0=yd/scale
            candidates=torch.cat([y0,x0],dim=0)
            distances=torch.cdist(x0,candidates)
            distances[:,1:]+=torch.eye(x.shape[0],device=x.device)*1e6
            drift_sum=torch.zeros_like(x0)
            norms=[]
            for tau in temperatures:
                logits=-distances/(float(tau)*math.sqrt(x.shape[-1]))
                affinity=(logits.softmax(-1)*logits.softmax(-2)).sqrt()
                ap,an=affinity[:,:1],affinity[:,1:]
                wp=ap*an.sum(-1,keepdim=True)
                wn=an*ap.sum(-1,keepdim=True)
                v=wp@y0-wn@x0
                rms=v.square().mean().sqrt().clamp_min(1e-6)
                drift_sum+=v/rms
                norms.append(float(rms))
            drifted=(xn.detach()+drift_sum).detach()
        loss=(xn-drifted).square().mean()
        terms.append(loss)
        diagnostics.append({"condition":b,"valid_dimensions":int(mask.sum()),"scale":float(scale),"drift_rms":norms,"loss":float(loss.detach())})
    if not terms:
        raise ValueError("No valid target dimensions")
    return torch.stack(terms).mean(),diagnostics
