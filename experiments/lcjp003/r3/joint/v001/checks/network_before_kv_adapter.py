import math
import torch
from torch import nn
from torch.nn import functional as F

class Cross(nn.Module):
    def __init__(self):
        super().__init__()
        self.qn=nn.LayerNorm(256); self.kn=nn.LayerNorm(256)
        self.att=nn.MultiheadAttention(256,4,batch_first=True,dropout=0)
        self.fn=nn.LayerNorm(256)
        self.ff=nn.Sequential(nn.Linear(256,512),nn.GELU(),nn.Linear(512,256))
    def forward(self,q,kv,mask=None):
        q=q+self.att(self.qn(q),self.kn(kv),self.kn(kv),key_padding_mask=mask,need_weights=False)[0]
        return q+self.ff(self.fn(q))

class LoRA(nn.Module):
    def __init__(self,base):
        super().__init__(); self.base=base
        self.a=nn.Linear(base.in_features,16,bias=False); self.b=nn.Linear(16,base.out_features,bias=False)
        nn.init.kaiming_uniform_(self.a.weight,a=math.sqrt(5)); nn.init.zeros_(self.b.weight)
    @property
    def weight(self): return self.base.weight
    @property
    def bias(self): return self.base.bias
    def forward(self,x): return self.base(x)+self.b(self.a(x))

def mlp(*dims,ln=False):
    layers=[]
    for i,(a,b) in enumerate(zip(dims,dims[1:])):
        layers.append(nn.Linear(a,b))
        if i<len(dims)-2:
            if ln: layers.append(nn.LayerNorm(b))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)

class JointNetwork(nn.Module):
    def __init__(self,expert):
        super().__init__(); self.expert=expert
        for p in expert.parameters(): p.requires_grad_(False)
        for layer in expert.layers:
            for name in ['q_proj','k_proj','v_proj','o_proj']:
                setattr(layer.self_attn,name,LoRA(getattr(layer.self_attn,name)))
        self.point=mlp(396,128,256,ln=True)
        self.neighbor=mlp(259,256,256,ln=True)
        self.box=nn.Linear(6,256)
        self.hist_proj=nn.Linear(27,256); self.hist_pos=nn.Parameter(torch.randn(16,256)*.01)
        self.hist=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,4,512,dropout=0,batch_first=True,norm_first=True,activation='gelu'),2,enable_nested_tensor=False)
        self.finger=mlp(6,128,256); self.finger_id=nn.Parameter(torch.randn(2,256)*.01)
        self.text=nn.Sequential(nn.Linear(960,256),nn.LayerNorm(256))
        self.object_text=Cross(); self.object_hist=Cross(); self.null=nn.Parameter(torch.zeros(1,1,256))
        self.role=mlp(256,128,1); self.destination=mlp(256,128,1)
        self.part_self=Cross(); self.part_finger=Cross(); self.part_context=Cross()
        self.score=mlp(512,256,128,1,ln=True)
        self.part_proj=nn.Sequential(nn.Linear(266,256),nn.LayerNorm(256))
        self.queries=nn.Parameter(torch.randn(1,8,256)*.02)
        self.base_read=Cross(); self.part_read=Cross()
        self.base_norm=nn.LayerNorm(256); self.part_norm=nn.LayerNorm(256)
        self.gate=mlp(774,128,1); nn.init.zeros_(self.gate[-1].weight); nn.init.constant_(self.gate[-1].bias,math.log(.05/.95))
        self.prefix_proj=nn.Linear(256,720); self.plan_proj=nn.Linear(7,720)
        self.noise_proj=nn.Linear(6,720); self.action_pos=nn.Parameter(torch.randn(16,720)*.01)
        self.motion=nn.Sequential(nn.Linear(720,256),nn.SiLU(),nn.Linear(256,6))
        self.grip=nn.Sequential(nn.Linear(720,128),nn.SiLU(),nn.Linear(128,2))
        self.aux=mlp(2304,512,256,21)

    def encode(self,d,score_override=None):
        x=d['x']; valid=d['valid']; k,n,_=x.shape
        p=self.point(x); p=p.masked_fill(~valid[...,None],0)
        idx=d['knn']; anchors=d['anchors']
        batches=torch.arange(k,device=x.device)[:,None,None]
        local=p[batches,idx]
        xyz=d['xyz']; centers=xyz[torch.arange(k,device=x.device)[:,None],anchors]
        rel=xyz[batches,idx]-centers[:,:,None,:]
        neighbor_valid=valid[batches,idx]
        groups=self.neighbor(torch.cat([local,rel],-1)).masked_fill(~neighbor_valid[...,None],-1e4).amax(2)
        groups=groups+self.box(d['box'])[:,None,:]
        obj=groups.mean(1)[None]
        h=self.hist_proj(d['h'])[None]+self.hist_pos[None]
        causal=torch.triu(torch.ones(16,16,device=x.device,dtype=torch.bool),diagonal=1)
        hist=self.hist(h,mask=causal)
        text=self.text(d['words'])[None]
        finger=self.finger(d['fingers'])[None]+self.finger_id.repeat_interleave(32,0)[None]
        obj=self.object_hist(self.object_text(obj,text),hist)
        roles=self.role(torch.cat([obj,self.null],1))[0,:,0]
        destinations=self.destination(torch.cat([obj,self.null],1))[0,:,0]
        g=self.part_self(groups,groups)
        g=self.part_finger(g,finger.expand(k,-1,-1))
        g=self.part_context(g,torch.cat([hist,text],1).expand(k,-1,-1))
        interp=d['interp']; iw=d['interp_weight']
        dense=(g[torch.arange(k,device=x.device)[:,None,None],interp]*iw[...,None]).sum(2)
        logits=self.score(torch.cat([p,dense],-1))[...,0].masked_fill(~valid,-1e9)
        alpha=logits.softmax(-1)
        if score_override=='uniform': alpha=valid.float()/valid.sum(-1,keepdim=True)
        elif score_override=='shuffle':
            alpha=torch.stack([a.roll(137) for a in alpha])*valid
            alpha=alpha/alpha.sum(-1,keepdim=True).clamp_min(1e-9)
        mass=(alpha[batches,idx]*neighbor_valid).sum(-1)
        w=alpha[batches,idx]*neighbor_valid; w=w/w.sum(-1,keepdim=True).clamp_min(1e-9)
        lf=(local*w[...,None]).sum(2)
        normals=d['normal'][batches,idx]; normals=(normals*w[...,None]).sum(2)
        ext=rel.amax(2)-rel.amin(2)
        part_tokens=self.part_proj(torch.cat([lf,centers,normals,ext,mass[...,None]],-1))
        top=mass.topk(8,dim=-1).indices
        selected=[]
        available=torch.ones_like(mass,dtype=torch.bool)
        for j in range(8):
            soft=(mass.clamp_min(1e-12).log().masked_fill(~available,-1e9)).softmax(-1)
            hard=F.one_hot(top[:,j],64).to(soft)
            pick=hard+soft-soft.detach() if self.training else hard
            selected.append(torch.einsum('kg,kgd->kd',pick,part_tokens))
            available=available.scatter(1,top[:,j,None],False)
        selected=torch.stack(selected,1)
        queries=self.queries.expand(k,-1,-1)
        # No dense point sequence enters this branch.
        base=self.base_norm(self.base_read(queries,obj[0,:,None,:]))
        part=self.part_norm(self.part_read(queries,selected))
        pi=roles.softmax(-1); ent=-(pi*pi.clamp_min(1e-12).log()).sum()/math.log(k+1)
        ae=-(alpha*alpha.clamp_min(1e-12).log()).sum(-1)/valid.sum(-1).float().clamp_min(2).log()
        stats=torch.stack([pi[:-1],ent.expand(k),ae,valid.float().mean(-1),d['dual_visible'],(1-pi[-1]).expand(k)],-1)
        gate=self.gate(torch.cat([base.mean(1),part.mean(1),hist[:,-1].expand(k,-1),stats],-1)).sigmoid()
        mix=(1-gate[...,None])*base+gate[...,None]*part
        aux=self.aux(torch.cat([part.flatten(1),hist[:,-1].expand(k,-1)],-1))
        return dict(roles=roles,destinations=destinations,alpha=alpha,logits=logits,gate=gate,base=base,part=part,mix=mix,aux=aux,hist=hist,finger=finger,text=text,obj=obj,top=top)

    def generate(self,e,noise,object_index=None,plan=None,lambda_override=None):
        if object_index is None: object_index=int(e['roles'].argmax())
        k=e['mix'].shape[0]
        if object_index==k:
            mix=self.base_norm(self.base_read(self.queries,e['obj']))[0]
        else:
            mix=e['mix'][object_index]
            if lambda_override is not None: mix=(1-lambda_override)*e['base'][object_index]+lambda_override*e['part'][object_index]
        destination=int(e['destinations'].argmax())
        dest=e['obj'][:,destination:destination+1] if destination<k else self.null
        seq=torch.cat([mix[None],e['hist'],e['finger'],e['text'],dest],1)
        prefix=self.prefix_proj(seq)
        if plan is None: plan=torch.zeros(16,7,device=noise.device)
        prefix=torch.cat([prefix,self.plan_proj(plan)[None]],1)
        g=noise.shape[0]; prefix=prefix.expand(g,-1,-1)
        action=self.noise_proj(noise)+self.action_pos
        tok=torch.cat([prefix,action],1); length=tok.shape[1]; pn=prefix.shape[1]
        mask=torch.zeros(g,1,length,length,device=tok.device,dtype=tok.dtype)
        mask[:,:,:pn,pn:]=torch.finfo(tok.dtype).min
        out=self.expert(inputs_embeds=tok,attention_mask=mask,use_cache=False).last_hidden_state[:,-16:]
        return self.motion(out),self.grip(out)
