import torch
from torch.nn import functional as F
from drifting import drift_loss
from data import CFG
def weighted(x,w):
 return (x*w).sum()/w.sum().clamp_min(1e-8)
def mask_objective(e,y):
 logits=e['seg']['logits'];target=y['mask'];w=y['mask_weight']
 terms=[]
 for v in range(2):
  for c in range(2):
   valid=w[v,c];truth=target[v,c];pred=logits[v,c];quality=valid.max()
   pos=valid*truth;neg=valid*(1-truth)
   bce=F.binary_cross_entropy_with_logits(pred,truth,reduction='none')
   balanced=.5*weighted(bce,pos)+.5*weighted(bce,neg)
   prob=pred.sigmoid()
   dice=1-(2*(prob*truth*valid).sum()+1e-5)/((prob*valid).sum()+(truth*valid).sum()+1e-5)
   terms.append((balanced+dice)*quality)
 return torch.stack(terms).mean()
def crossview_loss(d,e):
 p=e['points'];xyz=p['xyz'].detach();terms=[]
 for c in range(2):
  cam=(xyz-d['T'][c,:3,3])@d['T'][c,:3,:3]
  z=cam[:,2];proj=cam@d['K'][c].T;uv=proj[:,:2]/z[:,None].clamp_min(1e-8)
  pix=(uv-uv.new_tensor([.375,.875])).round().long();inside=(z>0)&(pix[:,0]>=0)&(pix[:,0]<256)&(pix[:,1]>=0)&(pix[:,1]<256)&p['valid']
  pix=pix.clamp(0,255)
  observed=d['depth'][c,pix[:,1],pix[:,0]]
  visible=inside&((observed-z).abs()<.006)&~d['robot'][c,pix[:,1],pix[:,0]]
  a=e['seg']['logits'][:,0].sigmoid().flatten()[p['ids']]
  b=e['seg']['logits'][c,0].sigmoid()[pix[:,1],pix[:,0]]
  terms.append(weighted((a-b).square(),visible.float()))
 return torch.stack(terms).mean()
def objective(model,d,y,noise,step,previous_plan=None,grip_weights=None):
 teacher=y['mask'][:,0]*(y['mask_weight'][:,0]>0) if step<=CFG['mask_union_updates'] else None
 e=model.encode(d,teacher_mask=teacher,step=step)
 motion,grip,switch=model.generate(e,noise,previous_plan)
 drift=drift_loss(motion.flatten(1)[None],y['target'].flatten()[None,None]).mean()
 grip_target=y['grip'][None].expand(noise.shape[0],-1)
 ce=F.cross_entropy(grip.reshape(-1,2),grip_target.reshape(-1),weight=grip_weights)
 swt=y['switch'][None].expand(noise.shape[0],-1)
 switch_loss=F.binary_cross_entropy_with_logits(switch,swt,pos_weight=switch.new_tensor(CFG.get('switch_positive_weight',4.)))
 ids=e['points']['ids'];valid=e['points']['valid'].float()
 target=y['part'].flatten()[ids];pw=y['part_weight'].flatten()[ids]*valid
 part=weighted(F.binary_cross_entropy_with_logits(e['point_logit'],target,reduction='none'),pw)*pw.max()
 # Smooth only observed supported neighbours on a geometrically continuous surface.
 kn=e['points']['knn'];anc=e['points']['anchors'];score=e['point_score']
 local_w=e['points']['neighbor'].float()*torch.minimum(pw[anc,None],pw[kn])
 local=weighted((score[anc,None]-score[kn]).square(),local_w)
 existence=weighted(F.binary_cross_entropy_with_logits(e['seg']['exist'],y['existence'],reduction='none'),y['exist_weight'])
 continuity=motion.new_zeros(())
 if previous_plan is not None:continuity=(motion[:,:15]-previous_plan[:,:15,:6].detach()).square().mean()
 components={'drift':drift,'grip':ce,'mask':mask_objective(e,y),'existence':existence,
 'crossview':crossview_loss(d,e),'part':part,'local':local,'switch':switch_loss,'continuity':continuity}
 loss=sum(CFG['loss_weights'][k]*v for k,v in components.items())
 return loss,{k:float(v.detach()) for k,v in components.items()},e,(motion,grip,switch)
