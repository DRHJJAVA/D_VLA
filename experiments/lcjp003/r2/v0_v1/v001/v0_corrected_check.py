import os,json,time,hashlib,traceback,pathlib,math
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np,h5py,torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoTokenizer
from smolvlm_with_expert_official import SmolVLMWithExpertModel
R=pathlib.Path('/mnt/nvme1/drh/D_VLA'); P=pathlib.Path(__file__).parent
def save(n,d): (P/n).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def digest(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def norm_stats():
 rows=json.loads((R/'experiments/contact_joint_object/manifests/episodes.json').read_text())['episodes']
 train=[e for e in rows if e['split']=='train' and int(e['episode'].split('_')[-1])<8]
 assert len(train)==80
 aa=[]
 for e in train:
  with h5py.File(e['raw'],'r') as f:aa.append(f['data'][e['episode']]['actions'][:].astype(np.float32))
 a=np.concatenate(aa);mu=a[:,:6].mean(0);sd=a[:,:6].std(0).clip(1e-6)
 restored=((a[:,:6]-mu)/sd)*sd+mu
 out={'training_episodes':80,'training_tasks':len(set(e['task'] for e in train)),'action_frames':len(a),'mean':mu.tolist(),'std':sd.tolist(),'continuous_dimensions':[0,1,2,3,4,5],'gripper_dimension':6,'gripper_values':np.unique(a[:,6]).tolist(),'roundtrip_max':float(abs(restored-a[:,:6]).max()),'development_used':False}
 assert out['roundtrip_max']<=1e-6
 assert set(out['gripper_values']).issubset({-1.,1.})
 save('action_normalization.json',out)
 return train,mu,sd
def make_data(e,t,mu,sd):
 import sys
 sys.path.insert(0,str(R/'experiments/lcjp003/src'))
 from data import P0Dataset,backproject
 base=R/'experiments/contact_joint_object/cache'
 sf=P0Dataset().frame(e['id'],t,n_points=8192)
 with h5py.File(base/'rgbd'/f"{e['id']}.h5",'r') as rgb,h5py.File(base/'observations'/f"{e['id']}.h5",'r') as ob:
  meta=json.loads(rgb.attrs['robot_metadata']);names=meta['body_names']
  ee=names.index('gripper0_eef'); lf=names.index('gripper0_leftfinger');rf=names.index('gripper0_rightfinger');rb=names.index('robot0_base')
  bi=rgb['robot/body_position'];br=rgb['robot/body_rotation']
  pos=bi[t,ee];rot=br[t,ee];fingers=(bi[t,[lf,rf]]-pos)@rot
  world=sf['xyz_base']@br[t,rb].T+bi[t,rb]
  g=ob['frames'][str(t)];obj=[];parity=[];normal_maps=[]
  for c,cn in enumerate(['agentview','robot0_eye_in_hand']):
   cg=rgb[cn];yy,xx=np.mgrid[0:256,0:256];duv=np.stack([xx.ravel(),yy.ravel()],-1)
   cloud=backproject(cg['depth_m'][t],cg['K'][t],cg['camera_to_world'][t],duv).reshape(256,256,3)
   nn=np.cross(np.gradient(cloud,axis=1),np.gradient(cloud,axis=0));nn/=np.maximum(np.linalg.norm(nn,axis=-1,keepdims=True),1e-8);normal_maps.append(nn)
   ix=np.flatnonzero((sf['camera_id']==c)&sf['point_valid'])
   xy=sf['depth_uv'][ix].astype(np.int64)
   direct=cloud[xy[:,1],xy[:,0]];parity.append(float(abs(direct-world[ix]).max()))
  for name in g:
   if not name.startswith('object_'):continue
   o=g[name]
   for c in [0,1]:
    if f'mask_{c}' not in o:continue
    mask=np.unpackbits(o[f'mask_{c}'][:]).reshape(256,256).astype(bool)
    uv=np.rint(sf['uv']).astype(np.int64).clip(0,255)
    good=(sf['camera_id']==c)&sf['point_valid']&mask[uv[:,1],uv[:,0]]
    ix=np.flatnonzero(good)
    if len(ix)==0:continue
    # Deduplicate physical source pixels; no duplicated padding counts as support.
    _,unique=np.unique(sf['depth_uv'][ix],axis=0,return_index=True);ix=ix[np.sort(unique)]
    xyz=world[ix];N=len(ix);xy=(sf['uv'][ix]//8).astype(np.int64).clip(0,31)
    feat=g[f'features_{c}'][:][xy[:,1],xy[:,0]].astype(np.float32)
    duv=sf['depth_uv'][ix].astype(np.int64);normal=normal_maps[c][duv[:,1],duv[:,0]]@rot
    local=(xyz-pos)@rot
    centers=[];dist=np.full(N,np.inf);j=0
    for _ in range(min(32,N)):
     centers.append(j);dist=np.minimum(dist,((local-local[j])**2).sum(-1));dist[centers]=-1;j=int(dist.argmax())
    q=np.zeros((32,32,396),np.float32);valid=np.zeros((32,32),bool)
    for k,j in enumerate(centers):
     jj=np.argsort(((local-local[j])**2).sum(-1),kind='stable')[:min(32,N)]
     aa=np.concatenate([feat[jj],local[jj],normal[jj],local[jj]-fingers[0],local[jj]-fingers[1]],-1)
     q[k,:len(jj)]=aa;valid[k,:len(jj)]=True
    obj.append((int(name.split('_')[-1])*2+c,q,valid,local.mean(0)))
  hist_idx=np.maximum(np.arange(t-15,t+1),0);hist=[]
  for j in hist_idx:
   p=bi[j,ee];rr=br[j,ee]
   hist.append(np.r_[p,rr.reshape(-1),rgb['robot/gripper_qpos'][j],((bi[j,[lf,rf]]-p)@rr).reshape(-1)])
 with h5py.File(e['raw'],'r') as f:act=f['data'][e['episode']]['actions'][t:t+16].astype(np.float32)
 assert len(act)==16 and max(parity)<1e-6
 save('v0_geometry_parity.json',{'source':'P0Dataset corrected depth and visibility_v3','world_backprojection_max_error':parity,'candidates':'view-specific mask proposals; legacy cross-view object IDs are not merged','legacy_xyz_used':False,'background_fallback_added':False,'support_pixel_deduplicated':True})
 targets=(act[:,:6]-mu)/sd
 return obj,np.stack(hist).astype(np.float32),targets,(act[:,6]>0).astype(np.int64),hist_idx
class Front(nn.Module):
 def __init__(self):
  super().__init__()
  self.point=nn.Sequential(nn.Linear(396,128),nn.GELU(),nn.Linear(128,256))
  self.local=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,4,512,dropout=0,batch_first=True),2)
  self.history_in=nn.Linear(20,256)
  self.history=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,4,512,dropout=0,batch_first=True),2)
  self.word=nn.Linear(960,256)
  self.role=nn.ModuleList([nn.MultiheadAttention(256,4,dropout=0,batch_first=True) for _ in range(2)])
  self.role_out=nn.Linear(256,3)
  self.rel=nn.ModuleList([nn.MultiheadAttention(256,4,dropout=0,batch_first=True) for _ in range(2)])
  self.score=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))
  self.adapter=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,960))
  self.motion_in=nn.Linear(6,720);self.time=nn.Sequential(nn.Linear(2,720),nn.SiLU(),nn.Linear(720,720))
  self.motion_out=nn.Linear(720,6);self.grip=nn.Linear(720,2)
 def encode(self,x,valid,h,words):
  C,K,N,_=x.shape
  z=self.point(x.masked_fill(~valid[...,None],0)); z=z.reshape(C*K,N,256)
  mask=valid.reshape(C*K,N);safe=mask.clone();safe[~safe.any(-1),0]=True
  z=self.local(z,src_key_padding_mask=~safe)
  patch=(z*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
  patch=patch.reshape(C,K,256); pv=valid.any(-1)
  hist=self.history_in(h)[None]
  causal=torch.triu(torch.ones(16,16,device=x.device,dtype=torch.bool),1)
  hist=self.history(hist,mask=causal)
  word=self.word(words)[None].expand(C,-1,-1)
  obj=(patch*pv[...,None]).sum(1)/pv.sum(1).clamp_min(1)[:,None]
  role=obj[:,None]
  for a in self.role:role=role+a(role,word,word,need_weights=False)[0]
  role_logits=self.role_out(role[:,0])
  rel=patch
  for a in self.rel:rel=rel+a(rel,hist.expand(C,-1,-1),hist.expand(C,-1,-1),need_weights=False)[0]
  scores=torch.sigmoid(self.score(rel).squeeze(-1))*pv
  # Select <=8 observed patches; selected scores remain differentiable.
  index=scores.masked_fill(~pv,-1).topk(min(8,K),-1).indices
  # Four local sub-tokens retain local shape, rather than duplicate a pooled token.
  zz=z.reshape(C,K,4,8,256);vm=valid.reshape(C,K,4,8)
  sub=(zz*vm[...,None]).sum(3)/vm.sum(3).clamp_min(1)[...,None]
  tokens=sub.gather(1,index[:,:,None,None].expand(-1,-1,4,256))
  weights=scores.gather(1,index); tokenvalid=vm.any(-1).gather(1,index[:,:,None].expand(-1,-1,4))
  tokens=tokens*weights[:,:,None,None]
  prefix=self.adapter(tokens.reshape(C,-1,256))
  prefix=prefix.masked_fill(~tokenvalid.reshape(C,-1,1),0)
  return prefix,tokenvalid.reshape(C,-1),scores,role_logits
def main():
 torch.set_num_threads(4);torch.manual_seed(20261027);np.random.seed(20261027)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 train,mu,sd=norm_stats()
 e=next(e for e in train if 'bbq_sauce' in e['task'] and e['episode']=='demo_0'); t=20
 obj,h,target,grip,idx=make_data(e,t,mu,sd)
 # All actual candidates at this frame, separate branches: no object or action averaging.
 ids=[o[0] for o in obj];x=torch.tensor(np.stack([o[1] for o in obj]),device='cuda');valid=torch.tensor(np.stack([o[2] for o in obj]),device='cuda')
 h=torch.tensor(h,device='cuda');target=torch.tensor(target,device='cuda');grip=torch.tensor(grip,device='cuda')
 cfg=json.loads((R/'experiments/lcjp003/weights/smolvla_base/config.json').read_text())
 backbone=SmolVLMWithExpertModel(model_id=str(R/'experiments/lcjp003/weights/smolvlm2'),load_vlm_weights=False,train_expert_only=True,freeze_vision_encoder=True,attention_mode=cfg['attention_mode'],num_expert_layers=cfg['num_expert_layers'],num_vlm_layers=cfg['num_vlm_layers'],self_attn_every_n_layers=cfg['self_attn_every_n_layers'],expert_width_multiplier=cfg['expert_width_multiplier'],device='cpu')
 state=load_file(str(R/'experiments/lcjp003/weights/smolvla_base/model.safetensors'));key='model.vlm_with_expert.'
 sub={k[len(key):]:v for k,v in state.items() if k.startswith(key)}
 backbone.load_state_dict(sub,strict=True);del state,sub
 for p in backbone.parameters():p.requires_grad_(False)
 backbone.float().to('cuda');backbone.eval()
 tok=AutoTokenizer.from_pretrained(str(R/'experiments/lcjp003/weights/smolvlm2'),local_files_only=True)
 text='Pick up the bbq sauce and place it in the basket.'
 ti=tok(text,return_tensors='pt')['input_ids'].to('cuda')
 # Pure language embeddings only; no whole-image tokens.
 words=backbone.embed_language_tokens(ti)[0].detach()
 front=Front().cuda();front.eval()
 before=digest(front)
 save('v0_tensor_manifest.json',{'pretrained_backbone':json.loads((P/'v0_weight_loading.json').read_text()),'new_random_tensors':{k:list(v.shape) for k,v in front.state_dict().items()},'new_parameters':sum(v.numel() for v in front.parameters()),'formal_training':False})
 start=time.perf_counter()
 prefix,pv,scores,roles=front.encode(x,valid,h,words);scores.retain_grad();roles.retain_grad()
 C=len(obj);L=prefix.shape[1]+words.shape[0];T=16
 prefix=torch.cat([words[None].expand(C,-1,-1),prefix],1)
 prefix_valid=torch.cat([torch.ones(C,words.shape[0],device='cuda',dtype=torch.bool),pv],1)
 noise=torch.randn(C,T,6,device='cuda');tt=torch.full((C,1,1),.5,device='cuda')
 noisy=(1-tt)*noise+tt*target[None]
 suffix=front.motion_in(noisy)+front.time(torch.cat([tt[:,0],1-tt[:,0]],-1))[:,None]
 attn=torch.zeros(C,L+T,L+T,device='cuda',dtype=torch.bool)
 attn[:,:L,:L]=prefix_valid[:,None,:];attn[:,L:,:L]=prefix_valid[:,None,:];attn[:,L:,L:]=True
 positions=torch.arange(L+T,device='cuda')[None].expand(C,-1)
 out,_=backbone(attention_mask=attn,position_ids=positions,inputs_embeds=[prefix,suffix],use_cache=False,fill_kv_cache=False)
 motion=front.motion_out(out[1]); gg=front.grip(out[1])
 branches=(motion-(target[None]-noise)).square().mean((1,2))+nn.functional.cross_entropy(gg.reshape(-1,2),grip[None].expand(C,-1).reshape(-1),reduction='none').reshape(C,T).mean(1)
 # Distinct candidate conditional losses; role marginalisation weights losses, never coordinates/actions.
 gate=roles[:,0].softmax(0);loss=(gate*branches).sum();loss.backward()
 torch.cuda.synchronize();seconds=time.perf_counter()-start
 grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in front.named_parameters()}
 finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in front.parameters())
 assert finite and scores.grad is not None and float(scores.grad.norm())>0
 assert float(roles.grad.norm())>0
 with torch.no_grad():
  pp,mm,ss,rr=front.encode(x,valid,h,words)
  xx=x.clone();xx[~valid]=12345
  p2,m2,s2,r2=front.encode(xx,valid,h,words)
  padding=float((pp-p2).abs().max())
  # All invalid patch outputs are masked; null is represented without manufacturing object points.
  pn,mn,sn,rn=front.encode(torch.zeros_like(x[:1]),torch.zeros_like(valid[:1]),h,words)
  null_finite=bool(torch.isfinite(pn).all()) and not bool(mn.any()) and float(sn.abs().max())==0
 assert padding<=1e-6 and null_finite
 after=digest(front);assert before==after
 # Exact causal input reconstruction; changing future observations cannot affect enumerated indices.
 assert idx.max()==t and idx.min()==t-15 and len(idx)==16
 input_hash=hashlib.sha256(x.cpu().numpy().tobytes()+h.cpu().numpy().tobytes()).hexdigest()
 report={'status':'core_interface_smoke_passed','episode':e['id'],'frame':t,'candidates':C,'candidate_ids':ids,'point_tensor_shape':list(x.shape),'real_points_with_overlap':int(valid.sum()),'history_indices':idx.tolist(),'history_shape':list(h.shape),'text_tokens':words.shape[0],'prefix_tokens':L,'action_shape':list(motion.shape),'gripper_logits_shape':list(gg.shape),'flow_matching_loss':float(loss),'action_loss_score_gradient_norm':float(scores.grad.norm()),'action_loss_role_gradient_norm':float(roles.grad.norm()),'all_computed_gradients_finite':finite,'parameter_gradient_norms':grads,'padding_perturbation_max':padding,'null_case_finite_and_masked':null_finite,'action_normalization_roundtrip':float(json.loads((P/'action_normalization.json').read_text())['roundtrip_max']),'backbone_frozen':True,'new_parameter_hash_before':before,'new_parameter_hash_after':after,'optimizer_updates':0,'input_sha256':input_hash,'whole_image_into_action_detail':False,'pure_text_embeddings':True,'future_actions_only_in_loss':True,'candidate_branches_separate':True,'seconds_one_forward_backward':seconds,'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'new_parameters':sum(v.numel() for v in front.parameters()),'coverage_limit':'Core B-H interface smoke; shared explanatory teacher, value weighting, plan-continuity scheduler and formal V2 objectives are not implemented or validated by this check. No trained role/score predictions are claimed.'}
 save('v0_result.json',report);save('v0_complete.json',{'status':report['status'],'optimizer_updates':0,'scope':report['coverage_limit']})
try:main()
except Exception as e:save('v0_full_failure.json',{'type':type(e).__name__,'summary':str(e).splitlines()[0][:300],'frames':[{'file':pathlib.Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]})
