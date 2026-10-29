import ast, pathlib, sys, math, hashlib, io, copy
import torch
from torch import nn

ROOT = pathlib.Path('/mnt/nvme1/drh/D_VLA')
OLD = ROOT / 'experiments/lcjp003/r2/v0_v1/v001'
HERE = pathlib.Path(__file__).parent

def load_legacy():
    sys.path.insert(0, str(OLD))
    path = OLD / 'v0_corrected_check.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.Try)]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Front')
    enc = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'encode')
    assert isinstance(enc.body[-1], ast.Return)
    enc.body[-1] = ast.parse('return prefix, tokenvalid.reshape(C,-1), scores, role_logits, tokens.reshape(C,-1,256), hist, word').body[0]
    ns = {'__file__':str(path), '__name__':'fusion_legacy_readonly'}
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'), ns)
    def local_save(name, data):
        import json
        (HERE / ('inherited_'+name)).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    ns['save'] = local_save
    return ns

LEGACY = load_legacy()

def causal_history(sequence, current_index, window=16):
    assert current_index >= 0 and current_index < len(sequence)
    index = torch.arange(current_index-window+1,current_index+1,device=sequence.device).clamp_min(0)
    return sequence[index]

def parameter_digest(module):
    h=hashlib.sha256()
    for name, p in module.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class FusionFront(LEGACY['Front']):
    def __init__(self):
        super().__init__()
        self.shared_queries=nn.Parameter(torch.randn(32,256)*.02)
        self.base_attention=nn.MultiheadAttention(256,4,dropout=0,batch_first=True)
        self.part_attention=nn.MultiheadAttention(256,4,dropout=0,batch_first=True)
        self.base_projection=nn.Linear(256,256)
        self.part_projection=nn.Linear(256,256)
        self.base_norm=nn.LayerNorm(256)
        self.part_norm=nn.LayerNorm(256)
        self.coarse_geometry=nn.Sequential(nn.Linear(12,256),nn.GELU(),nn.Linear(256,256))
        self.gate=nn.Sequential(nn.Linear(770,128),nn.GELU(),nn.Linear(128,1))
        nn.init.normal_(self.gate[-1].weight, std=.001)
        nn.init.constant_(self.gate[-1].bias, math.log(.25/.75))
        # Interface-test fixtures only; not a V2 training configuration.
        self.register_buffer('warmup_initial_cap',torch.tensor(.2))
        self.register_buffer('warmup_updates',torch.tensor(0,dtype=torch.long))
        self.register_buffer('warmup_length',torch.tensor(100,dtype=torch.long))

    def cap(self):
        fraction=(self.warmup_updates.float()/self.warmup_length.clamp_min(1).float()).clamp(0,1)
        return self.warmup_initial_cap+(1-self.warmup_initial_cap)*fraction

    def encode(self,x,valid,h,words,lambda_override=None,part_only=False):
        _,token_mask,scores,roles,tokens,hist,word=super().encode(x,valid,h,words)
        b=x.shape[0]
        object_valid=valid.flatten(1).any(1)
        # Only 3D coordinates (feature positions 384:387), never visual channels,
        # are used to assemble coarse scene/context geometry.
        xyz=x[...,384:387].masked_fill(~valid[...,None],0).flatten(1,2)
        mask=valid.flatten(1)
        count=mask.sum(1,keepdim=True).clamp_min(1)
        mean=xyz.sum(1)/count
        low=xyz.masked_fill(~mask[...,None],float('inf')).amin(1)
        high=xyz.masked_fill(~mask[...,None],float('-inf')).amax(1)
        low=torch.where(object_valid[:,None],low,torch.zeros_like(low))
        high=torch.where(object_valid[:,None],high,torch.zeros_like(high))
        variance=((xyz-mean[:,None]).square()*mask[...,None]).sum(1)/count
        geometry=self.coarse_geometry(torch.cat([mean,low,high,variance.clamp_min(1e-12).sqrt()],-1))
        geometry=geometry.masked_fill(~object_valid[:,None],0)
        history=hist.expand(b,-1,-1)
        scene=geometry[None].expand(b,-1,-1)
        common=torch.cat([history,word,scene],1)
        common_valid=torch.cat([torch.ones(b,history.shape[1]+word.shape[1],dtype=torch.bool,device=x.device),
                                object_valid[None].expand(b,-1)],1)
        base_memory=torch.cat([common,geometry[:,None]],1)
        base_valid=torch.cat([common_valid,object_valid[:,None]],1)
        q=self.shared_queries[None].expand(b,-1,-1)
        base=self.base_attention(q,base_memory,base_memory,key_padding_mask=~base_valid,need_weights=False)[0]
        # All-invalid part rows get a zero safe sentinel internally; they have
        # zero output and zero effective lambda after attention.
        safe=token_mask.clone()
        has_part=token_mask.any(1)
        safe[~has_part,0]=True
        clean=tokens.masked_fill(~token_mask[...,None],0)
        part=self.part_attention(q,clean,clean,key_padding_mask=~safe,need_weights=False)[0]
        base=self.base_norm(self.base_projection(base))
        part=self.part_norm(self.part_projection(part))
        part=part.masked_fill(~has_part[:,None,None],0)
        role_prob=roles[:,0].masked_fill(~object_valid,-1e4).softmax(0)
        visible=valid.float().flatten(1).mean(1)
        common_pool=(common*common_valid[...,None]).sum(1)/common_valid.sum(1,keepdim=True).clamp_min(1)
        gate_input=torch.cat([common_pool,base.mean(1),part.mean(1),role_prob[:,None],visible[:,None]],-1)
        lam=self.cap()*torch.sigmoid(self.gate(gate_input))
        if lambda_override is not None:
            lam=torch.full_like(lam,float(lambda_override))
        lam=lam*has_part[:,None]
        mixed=(1-lam[...,None])*base+lam[...,None]*part
        selected=part if part_only else mixed
        selected_valid=has_part[:,None].expand(b,32) if part_only else torch.ones(b,32,dtype=torch.bool,device=x.device)
        prefix_valid=torch.cat([common_valid,selected_valid],1)
        prefix=self.adapter(torch.cat([common,selected],1)).masked_fill(~prefix_valid[...,None],0)
        return {
            'prefix':prefix,'prefix_valid':prefix_valid,'scores':scores,'roles':roles,
            'lambda':lam,'base':base,'part':part,'mixed':mixed,'common':common,
            'common_valid':common_valid,'has_part':has_part,'role_prob':role_prob,
            'gate_input':gate_input,'shared_query_ptr':q.data_ptr()
        }

def expert_forward(front,backbone,encoded,noise,time_scalar,target):
    b=noise.shape[0];length=encoded['prefix'].shape[1];horizon=noise.shape[1]
    noisy=(1-time_scalar)*target+time_scalar*noise
    times=torch.stack([time_scalar.reshape(-1),1-time_scalar.reshape(-1)],-1)
    suffix=front.motion_in(noisy)+front.time(times)[:,None]
    valid=encoded['prefix_valid']
    total=length+horizon
    attention=torch.ones(b,total,total,dtype=torch.bool,device=noise.device)
    attention[:,:length,length:]=False
    attention[:,:,:length]&=valid[:,None,:]
    positions=torch.arange(total,device=noise.device)[None].expand(b,-1)
    out,_=backbone(attention_mask=attention,position_ids=positions,
                   inputs_embeds=[encoded['prefix'],suffix],use_cache=False,fill_kv_cache=False)
    hidden=out[1]
    motion=front.motion_out(hidden)
    grip=front.grip(hidden)
    return motion,grip

def action_loss(motion,grip_logits,noise,target,gripper):
    mse=(motion-(noise-target)).square().mean((1,2))
    ce=torch.nn.functional.cross_entropy(grip_logits.transpose(1,2),gripper,reduction='none').mean(1)
    return mse+ce
