
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F

class NoValidObservationError(ValueError):
    pass

def mlp(widths, final_activation=True):
    layers=[]
    for i,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
        layers.append(nn.Linear(a,b))
        if i<len(widths)-2 or final_activation:
            layers.extend([nn.LayerNorm(b),nn.GELU()])
    return nn.Sequential(*layers)

@dataclass(frozen=True)
class ModelConfig:
    semantic_dim: int = 768
    state_dim: int = 18
    action_dim: int = 7
    history_length: int = 16
    action_horizon: int = 16
    max_points: int = 2048
    neighbours: int = 32
    radius1: float = 0.1
    radius2: float = 0.2
    temperature: float = 1.0
    action_layers: int = 16
    action_width: int = 720
    query_heads: int = 15
    kv_heads: int = 5
    head_dim: int = 64
    action_ffn: int = 2048

@torch.no_grad()
def farthest_points(x, count):
    n=x.shape[0]
    count=min(n,max(1,int(count)))
    distances=torch.full((n,),float('inf'),device=x.device)
    result=torch.empty(count,dtype=torch.long,device=x.device)
    farthest=((x-x.mean(0))**2).sum(-1).argmax()
    for j in range(count):
        result[j]=farthest
        d=((x-x[farthest])**2).sum(-1)
        distances=torch.minimum(distances,d)
        farthest=distances.argmax()
    return result

@torch.no_grad()
def neighbourhood(x,centres,radius,max_neighbours):
    distances=torch.cdist(centres.float(),x.float())
    count=min(x.shape[0],max_neighbours)
    d,index=distances.topk(count,largest=False,dim=1)
    mask=d<=radius
    # Every FPS centre is an observed point and therefore has a zero-distance neighbour.
    if not bool(mask.any(-1).all()):
        raise ValueError('empty_geometric_neighbourhood')
    return index,mask

def interpolate(source_xyz, source_features, target_xyz):
    if source_xyz.shape[0]==1:
        return source_features.expand(target_xyz.shape[0],-1)
    with torch.no_grad():
        dist,index=torch.cdist(target_xyz.float(),source_xyz.float()).topk(min(3,source_xyz.shape[0]),largest=False,dim=1)
        inverse=1.0/dist.clamp_min(1e-8)
        weights=inverse/inverse.sum(-1,keepdim=True)
    return (source_features[index]*weights.to(source_features.dtype)[...,None]).sum(1)

class ObjectPointNet(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.sa1=mlp([131,128,128,256])
        self.sa2=mlp([259,256,256,512])
        self.fp2=mlp([768,256,256])
        self.fp1=mlp([384,256,256])

    def one_object(self,xyz,features):
        i1=farthest_points(xyz,math.ceil(len(xyz)/4))
        p1=xyz[i1]
        near1,valid1=neighbourhood(xyz,p1,self.cfg.radius1,self.cfg.neighbours)
        h1=self.sa1(torch.cat([xyz[near1]-p1[:,None],features[near1]],-1))
        f1=h1.masked_fill(~valid1[...,None],-torch.inf).amax(1)
        i2=farthest_points(p1,math.ceil(len(p1)/4))
        p2=p1[i2]
        near2,valid2=neighbourhood(p1,p2,self.cfg.radius2,self.cfg.neighbours)
        h2=self.sa2(torch.cat([p1[near2]-p2[:,None],f1[near2]],-1))
        f2=h2.masked_fill(~valid2[...,None],-torch.inf).amax(1)
        up1=self.fp2(torch.cat([interpolate(p2,f2,p1),f1],-1))
        return self.fp1(torch.cat([interpolate(p1,up1,xyz),features],-1))

    def forward(self,xyz,features,object_id,valid):
        batch,steps,points,_=features.shape
        frames=[]
        for bi in range(batch):
            times=[]
            for ti in range(steps):
                ids=object_id[bi,ti]
                mask=valid[bi,ti]
                out=features.new_zeros(points,256)
                for oid in torch.unique(ids[mask]).tolist():
                    if oid<0:
                        raise ValueError('valid_point_has_no_object')
                    ii=torch.where(mask & (ids==oid))[0]
                    value=self.one_object(xyz[bi,ti,ii],features[bi,ti,ii])
                    out=out.index_copy(0,ii,value)
                times.append(out)
            frames.append(torch.stack(times))
        return torch.stack(frames)

class RMSNorm(nn.Module):
    def __init__(self,width,eps=1e-5):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(width))
        self.eps=eps
    def forward(self,x):
        var=x.float().square().mean(-1,keepdim=True)
        return (x.float()*torch.rsqrt(var+self.eps)).to(x.dtype)*self.weight

class GroupedAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.hq,self.hkv,self.d=cfg.query_heads,cfg.kv_heads,cfg.head_dim
        self.q=nn.Linear(cfg.action_width,self.hq*self.d,bias=False)
        self.k=nn.Linear(cfg.action_width,self.hkv*self.d,bias=False)
        self.v=nn.Linear(cfg.action_width,self.hkv*self.d,bias=False)
        self.out=nn.Linear(self.hq*self.d,cfg.action_width,bias=False)
    def forward(self,x,memory,bias=None):
        bs,s,_=x.shape;m=memory.shape[1]
        q=self.q(x).view(bs,s,self.hq,self.d).transpose(1,2)
        k=self.k(memory).view(bs,m,self.hkv,self.d).transpose(1,2)
        v=self.v(memory).view(bs,m,self.hkv,self.d).transpose(1,2)
        k=k.repeat_interleave(self.hq//self.hkv,dim=1)
        v=v.repeat_interleave(self.hq//self.hkv,dim=1)
        logits=q.float() @ k.float().transpose(-2,-1)/math.sqrt(self.d)
        if bias is not None:
            logits=logits+bias[:,None,None,:].float()
        attention=logits.softmax(-1).to(v.dtype)
        value=(attention@v).transpose(1,2).reshape(bs,s,self.hq*self.d)
        return self.out(value)

class ExpertBlock(nn.Module):
    def __init__(self,cfg,cross):
        super().__init__()
        self.cross=cross
        self.norm1=RMSNorm(cfg.action_width)
        self.attention=GroupedAttention(cfg)
        self.norm2=RMSNorm(cfg.action_width)
        self.gate=nn.Linear(cfg.action_width,cfg.action_ffn,bias=False)
        self.up=nn.Linear(cfg.action_width,cfg.action_ffn,bias=False)
        self.down=nn.Linear(cfg.action_ffn,cfg.action_width,bias=False)
    def forward(self,x,memory,bias):
        h=self.norm1(x)
        x=x+self.attention(h,memory,bias) if self.cross else x+self.attention(h,h)
        h=self.norm2(x)
        return x+self.down(F.silu(self.gate(h))*self.up(h))

class ActionExpert(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.text_projection=nn.Linear(960,cfg.action_width)
        self.part_projection=nn.Linear(256,cfg.action_width)
        self.state_projection=nn.Linear(256,cfg.action_width)
        self.noise_projection=nn.Linear(cfg.action_dim,cfg.action_width)
        self.position=nn.Parameter(torch.zeros(1,cfg.action_horizon+1,cfg.action_width))
        nn.init.normal_(self.position,std=0.02)
        self.blocks=nn.ModuleList([ExpertBlock(cfg,bool(i%2)) for i in range(cfg.action_layers)])
        self.norm=RMSNorm(cfg.action_width)
        self.output=nn.Linear(cfg.action_width,cfg.action_dim)

    def forward(self,parts,beta,part_valid,text,text_valid,current_state,noise):
        memory=torch.cat([self.text_projection(text),self.part_projection(parts)],1)
        safe_beta=beta.clamp_min(torch.finfo(beta.dtype).tiny)
        bias=torch.cat([torch.zeros_like(text_valid,dtype=beta.dtype),safe_beta.log()],1)
        mask=torch.cat([text_valid,part_valid],1)
        bias=bias.masked_fill(~mask,-torch.inf)
        x=torch.cat([self.state_projection(current_state)[:,None],self.noise_projection(noise)],1)
        x=x+self.position
        for block in self.blocks:
            x=block(x,memory,bias)
        return self.output(self.norm(x[:,1:]))

class LCJP003(nn.Module):
    REQUIRED=('semantic','xyz_base','xyz_norm','relation','object_id','point_valid',
              'state','history_valid','text_features','text_valid','group_id')
    def __init__(self,cfg=ModelConfig()):
        super().__init__()
        self.cfg=cfg
        if (cfg.action_layers,cfg.action_width,cfg.query_heads,cfg.kv_heads,cfg.head_dim,cfg.action_ffn)!=(16,720,15,5,64,2048):
            raise ValueError('architecture_must_match_LCJP003')
        if cfg.history_length!=16 or cfg.action_horizon!=16:
            raise ValueError('temporal_shapes_must_match_LCJP003')
        self.visual_adapter=mlp([cfg.semantic_dim,128])
        self.point_encoder=ObjectPointNet(cfg)
        self.relation_encoder=mlp([16,64,128])
        self.state_encoder=mlp([cfg.state_dim,128,256])
        self.language_projection=nn.Linear(960,256)
        self.history_projection=nn.Linear(512,256)
        layer=nn.TransformerEncoderLayer(256,4,1024,dropout=0.0,activation='gelu',batch_first=True,norm_first=True)
        self.history_encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.query_mlp=mlp([768,256,256])
        self.point_score=mlp([640,256,1],final_activation=False)
        self.part_projection=nn.Sequential(nn.Linear(270,256),nn.LayerNorm(256))
        self.group_score=mlp([512,128,1],final_activation=False)
        self.action_expert=ActionExpert(cfg)
        freqs=torch.exp(-math.log(10000.0)*torch.arange(128)/128)
        self.register_buffer('time_frequencies',freqs,persistent=False)

    def parameter_groups(self):
        return {
            'visual_adapter':self.visual_adapter,
            'point_encoder':self.point_encoder,
            'relation_encoder':self.relation_encoder,
            'state_encoder':self.state_encoder,
            'language_projection':self.language_projection,
            'history_projection':self.history_projection,
            'history_encoder':self.history_encoder,
            'query_mlp':self.query_mlp,
            'point_score':self.point_score,
            'part_projection':self.part_projection,
            'group_score':self.group_score,
            'action_expert':self.action_expert,
        }

    def _validate(self,batch):
        missing=[k for k in self.REQUIRED if k not in batch]
        if missing: raise KeyError('missing_model_input:'+','.join(missing))
        bs,ts,n,cv=batch['semantic'].shape
        if ts!=16 or cv!=self.cfg.semantic_dim or n>self.cfg.max_points:
            raise ValueError('input_feature_shape_mismatch')
        expected={'xyz_base':(bs,ts,n,3),'xyz_norm':(bs,ts,n,3),'relation':(bs,ts,n,16),
                  'object_id':(bs,ts,n),'point_valid':(bs,ts,n),
                  'state':(bs,ts,self.cfg.state_dim),'history_valid':(bs,ts),'group_id':(bs,n)}
        for k,shape in expected.items():
            if tuple(batch[k].shape)!=shape: raise ValueError('shape_mismatch:'+k)
        if batch['text_features'].shape[0]!=bs or batch['text_features'].shape[-1]!=960:
            raise ValueError('text_shape_mismatch')
        if tuple(batch['text_valid'].shape)!=tuple(batch['text_features'].shape[:2]):
            raise ValueError('text_mask_mismatch')
        if not bool(batch['text_valid'].bool().any(-1).all()):
            raise ValueError('no_valid_language_tokens')
        if not bool(batch['history_valid'][:,-1].bool().all()):
            raise NoValidObservationError('current_time_invalid')
        valid=batch['point_valid'].bool() & batch['history_valid'].bool()[...,None]
        if not bool(valid[:,-1].any(-1).all()):
            raise NoValidObservationError('no_valid_current_points')
        if bool((batch['object_id'][valid]<0).any()):
            raise ValueError('valid_point_without_object')
        if bool((batch['group_id'][valid[:,-1]]<0).any()):
            raise ValueError('valid_current_point_without_group')
        return valid

    def encode(self,batch,return_diagnostics=True,uniform=False):
        valid=self._validate(batch)
        bs,ts,n,_=batch['semantic'].shape
        clean=lambda x: torch.where(valid[...,None],x,torch.zeros((),dtype=x.dtype,device=x.device))
        semantic=clean(batch['semantic'].detach())
        xyz=clean(batch['xyz_base'])
        xyz_norm=clean(batch['xyz_norm'])
        relation=clean(batch['relation'])
        state=torch.where(batch['history_valid'].bool()[...,None],batch['state'],torch.zeros_like(batch['state']))
        text=torch.where(batch['text_valid'].bool()[...,None],batch['text_features'].detach(),torch.zeros_like(batch['text_features']))
        for name,x in [('semantic',semantic),('xyz_base',xyz),('xyz_norm',xyz_norm),('relation',relation),('state',state),('text_features',text)]:
            if not bool(torch.isfinite(x).all()): raise ValueError('nonfinite_valid_input:'+name)
        f0=self.visual_adapter(semantic)
        feats=self.point_encoder(xyz_norm,f0,batch['object_id'],valid)
        states=self.state_encoder(state)
        tm=batch['text_valid'].bool()
        pooled=(text*tm[...,None]).sum(1)/tm.sum(1,keepdim=True)
        language=self.language_projection(pooled)
        objects=batch['object_id']
        query_frames=[];present_frames=[]
        time=batch.get('time_delta')
        if time is None:
            time=torch.arange(ts,device=semantic.device,dtype=semantic.dtype)[None].expand(bs,-1)-(ts-1)
        if tuple(time.shape)!=(bs,ts):raise ValueError('time_delta_shape')
        current_queries=[]
        for bi in range(bs):
            current=torch.unique(objects[bi,-1,valid[bi,-1]]).tolist()
            qmap={}
            for oid in current:
                obs=[];times=[]
                for ti in range(ts):
                    vm=valid[bi,ti] & (objects[bi,ti]==oid)
                    if bool(vm.any()):
                        pooled_object=feats[bi,ti,vm].mean(0)
                        obs.append(self.history_projection(torch.cat([pooled_object,states[bi,ti]],-1)))
                        times.append(time[bi,ti])
                if not obs: raise NoValidObservationError('empty_object_history')
                h=torch.stack(obs)
                phase=torch.stack(times).to(h.dtype)[:,None]*self.time_frequencies[None].to(h.dtype)
                h=h+torch.cat([phase.sin(),phase.cos()],-1)
                causal=torch.ones(len(obs),len(obs),device=h.device,dtype=torch.bool).triu(1)
                history=self.history_encoder(h[None],mask=causal)[0,-1]
                qmap[int(oid)]=self.query_mlp(torch.cat([language[bi],states[bi,-1],history],-1))
            point_q=feats.new_zeros(n,256)
            for oid,q in qmap.items():
                ii=torch.where(valid[bi,-1] & (objects[bi,-1]==oid))[0]
                point_q=point_q.index_copy(0,ii,q.expand(len(ii),-1))
            current_queries.append(point_q)
            query_frames.append(qmap)
        queries=torch.stack(current_queries)
        current_feats=feats[:,-1]
        rel=self.relation_encoder(relation[:,-1])
        point_logits=self.point_score(torch.cat([current_feats,rel,queries],-1)).squeeze(-1)
        group_lists=[]
        alpha_rows=[]
        for bi in range(bs):
            groups=torch.unique(batch['group_id'][bi,valid[bi,-1]]).tolist()
            row=point_logits.new_zeros(n);meta=[]
            for gid in groups:
                ii=torch.where(valid[bi,-1] & (batch['group_id'][bi]==gid))[0]
                oids=torch.unique(objects[bi,-1,ii])
                if oids.numel()!=1:raise ValueError('part_group_crosses_objects')
                weights=torch.ones_like(point_logits[bi,ii])/len(ii) if uniform else (point_logits[bi,ii]/self.cfg.temperature).softmax(0)
                row=row.index_copy(0,ii,weights)
                meta.append((int(gid),int(oids.item()),ii))
            alpha_rows.append(row);group_lists.append(meta)
        alpha=torch.stack(alpha_rows)
        if return_diagnostics and alpha.requires_grad:alpha.retain_grad()
        max_groups=max(map(len,group_lists))
        all_parts=[];all_beta=[];all_part_valid=[];all_scores=[];all_group_maps=[];all_group_ids=[]
        all_centres=[];all_gripper_centres=[];all_cov=[]
        for bi,meta in enumerate(group_lists):
            z=[];v=[];centres=[];gripper_centres=[];covariances=[]
            for gid,oid,ii in meta:
                w=alpha[bi,ii]
                local_features=(w[:,None]*current_feats[bi,ii]).sum(0)
                p=xyz[bi,-1,ii];centre=(w[:,None]*p).sum(0)
                gripper_centre=(w[:,None]*relation[bi,-1,ii,:3]).sum(0)
                delta=p-centre
                covariance=(delta*w[:,None]).T@delta
                cv=torch.stack([covariance[0,0],covariance[1,1],covariance[2,2],
                                covariance[0,1],covariance[0,2],covariance[1,2]])
                quality=(w[:,None]*relation[bi,-1,ii,14:16]).sum(0)
                token=self.part_projection(torch.cat([local_features,centre,gripper_centre,cv,quality]))
                logit=self.group_score(torch.cat([token,query_frames[bi][oid]],-1)).squeeze(-1)
                z.append(token);v.append(logit);centres.append(centre);gripper_centres.append(gripper_centre);covariances.append(cv)
            count=len(meta)
            vv=torch.stack(v)
            bb=torch.ones_like(vv)/count if uniform else (vv/self.cfg.temperature).softmax(0)
            all_parts.append(F.pad(torch.stack(z),(0,0,0,max_groups-count)))
            all_beta.append(F.pad(bb,(0,max_groups-count)))
            pm=torch.arange(max_groups,device=semantic.device)<count
            all_part_valid.append(pm)
            gm=torch.full((n,),-1,device=semantic.device,dtype=torch.long)
            for ki,(_,_,ii) in enumerate(meta):gm=gm.index_fill(0,ii,ki)
            all_group_maps.append(gm)
            all_group_ids.append(torch.tensor([x[0] for x in meta]+[-1]*(max_groups-count),device=semantic.device))
            all_centres.append(F.pad(torch.stack(centres),(0,0,0,max_groups-count)))
            all_gripper_centres.append(F.pad(torch.stack(gripper_centres),(0,0,0,max_groups-count)))
            all_cov.append(F.pad(torch.stack(covariances),(0,0,0,max_groups-count)))
        parts=torch.stack(all_parts);beta=torch.stack(all_beta);part_valid=torch.stack(all_part_valid)
        group_map=torch.stack(all_group_maps)
        scores=alpha*beta.gather(1,group_map.clamp_min(0))
        scores=scores.masked_fill(~valid[:,-1],0)
        if return_diagnostics:
            if beta.requires_grad:beta.retain_grad()
            if parts.requires_grad:parts.retain_grad()
        return {
            'parts':parts,'part_tokens':parts,'beta':beta,'alpha':alpha,'scores':scores,
            'point_valid':valid[:,-1],'part_valid':part_valid,
            'group_map':group_map,'group_ids':torch.stack(all_group_ids),
            'centres_base':torch.stack(all_centres),'centres_gripper':torch.stack(all_gripper_centres),
            'covariance':torch.stack(all_cov),'text_features':text,'text_valid':tm,
            'current_state':states[:,-1],'point_features':current_feats,
            'valid_observation':valid[:,-1].any(-1),
            'history_only_through_score':True,
        }

    def decode(self,context,action_noise):
        if action_noise.ndim not in (3,4):raise ValueError('action_noise_rank')
        batch=context['parts'].shape[0]
        if action_noise.shape[0]!=batch or tuple(action_noise.shape[-2:])!=(self.cfg.action_horizon,self.cfg.action_dim):
            raise ValueError('action_noise_shape')
        if not bool(torch.isfinite(action_noise).all()):raise ValueError('action_noise_nonfinite')
        groups=action_noise.shape[1] if action_noise.ndim==4 else 1
        def expand(x):
            if groups==1:return x
            return x[:,None].expand(batch,groups,*x.shape[1:]).reshape(batch*groups,*x.shape[1:])
        noise=action_noise.reshape(batch*groups,self.cfg.action_horizon,self.cfg.action_dim)
        out=self.action_expert(expand(context['parts']),expand(context['beta']),
                               expand(context['part_valid']),expand(context['text_features']),
                               expand(context['text_valid']),expand(context['current_state']),noise)
        return out.reshape(batch,groups,self.cfg.action_horizon,self.cfg.action_dim) if action_noise.ndim==4 else out

    def forward(self,batch,action_noise,return_diagnostics=True,uniform=False):
        context=self.encode(batch,return_diagnostics=return_diagnostics,uniform=uniform)
        context['actions']=self.decode(context,action_noise)
        return context

    def schema(self):
        return {'config':asdict(self.cfg),'trainable_parameters':sum(p.numel() for p in self.parameters()),
                'parameters_by_group':{k:sum(p.numel() for p in m.parameters()) for k,m in self.parameter_groups().items()},
                'external_frozen_inputs':['semantic','text_features'],
                'frozen_backbone_included':False,
                'contact_classifier':False,'legacy_targets':False}
