
"""Frozen official spatial/text features for the LCJP003 P0 baseline."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

def parameter_digest(module):
    h=hashlib.sha256()
    for name,p in module.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

class FrozenSmolFeatures:
    def __init__(self, weights, device="cuda"):
        from transformers import AutoModelForImageTextToText,AutoTokenizer,AutoImageProcessor
        self.device=torch.device(device)
        self.model=AutoModelForImageTextToText.from_pretrained(
            str(weights),local_files_only=True,torch_dtype=torch.float32,
            attn_implementation="eager").to(self.device).eval()
        self.model.requires_grad_(False)
        inner=self.model.model
        self.vision=inner.vision_model
        self.text=inner.text_model
        self.text.layers=self.text.layers[:16]
        self.text.config.num_hidden_layers=16
        self.tokenizer=AutoTokenizer.from_pretrained(str(weights),local_files_only=True)
        proc=AutoImageProcessor.from_pretrained(str(weights),local_files_only=True)
        self.image_mean=torch.tensor(proc.image_mean,device=self.device).reshape(1,3,1,1)
        self.image_std=torch.tensor(proc.image_std,device=self.device).reshape(1,3,1,1)
        self.image_size=512
        self.audit={"backend":"SmolVLM2-500M-Video-Instruct",
            "visual_channels":768,"text_channels":960,"text_layers":16,
            "image_size":512,"weights":str(weights),"requires_grad_count":sum(p.requires_grad for p in self.model.parameters())}
        self.audit["frozen_digest_before"]=parameter_digest(self.model)
    @torch.inference_mode()
    def image_grid(self, rgb):
        """rgb [2,H,W,3] uint8; grid [2,C,32,32]."""
        t=torch.from_numpy(np.asarray(rgb).copy()).to(self.device).permute(0,3,1,2).float()/255
        t=F.interpolate(t,size=(512,512),mode="bilinear",align_corners=False)
        t=(t-self.image_mean)/self.image_std
        out=self.vision(pixel_values=t,return_dict=True).last_hidden_state.float()
        side=int(round(out.shape[1]**0.5))
        if side*side!=out.shape[1] or out.shape[-1]!=768:
            raise ValueError("Unexpected spatial vision schema")
        return out.transpose(1,2).reshape(2,768,side,side)
    @torch.inference_mode()
    def text_features(self,instruction):
        data=self.tokenizer(instruction,return_tensors="pt",truncation=True,max_length=48)
        ids=data.input_ids.to(self.device);mask=data.attention_mask.to(self.device)
        out=self.text(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
        return out[0].float().cpu().numpy(),mask[0].bool().cpu().numpy()
    @torch.inference_mode()
    def point_features(self,rgb,projected_uv,view_visible):
        """T,2,H,W,3; T,N,2,2; T,N,2. Fuse only visible views."""
        rgb=np.asarray(rgb);uv=np.asarray(projected_uv);valid=np.asarray(view_visible).astype(bool)
        rows=[]
        for t in range(rgb.shape[0]):
            grid=self.image_grid(rgb[t])
            n=uv.shape[1];h,w=rgb.shape[2:4]
            coord=torch.from_numpy(uv[t].copy()).to(self.device).permute(1,0,2).float()
            vis=torch.from_numpy(valid[t].copy()).to(self.device).T
            coord=torch.nan_to_num(coord)
            coord[...,0]=2*(coord[...,0]+.5)/w-1
            coord[...,1]=2*(coord[...,1]+.5)/h-1
            samples=F.grid_sample(grid,coord.unsqueeze(2),mode="bilinear",padding_mode="zeros",align_corners=False)
            samples=samples.squeeze(-1).permute(0,2,1)
            fused=(samples*vis[...,None]).sum(0)/vis.sum(0).clamp_min(1)[:,None]
            fused=torch.where(vis.any(0)[:,None],fused,torch.zeros_like(fused))
            if not torch.isfinite(fused).all():raise FloatingPointError("Nonfinite frozen feature")
            rows.append(fused.cpu().numpy())
        return np.stack(rows)
    def finish_audit(self):
        self.audit["frozen_digest_after"]=parameter_digest(self.model)
        self.audit["unchanged"]=self.audit["frozen_digest_before"]==self.audit["frozen_digest_after"]
        self.audit["any_gradient"]=any(p.grad is not None for p in self.model.parameters())
        return self.audit
