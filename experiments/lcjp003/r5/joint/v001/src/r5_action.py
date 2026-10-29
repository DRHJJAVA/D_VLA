import torch
from torch import nn

class R5Action(nn.Module):
 def __init__(self, initialized_r4):
  super().__init__()
  # Reuse official expert initialization/adapters, never a trained R4 checkpoint.
  for name in ['expert','prefix_proj','noise_proj','motion','grip']:
   setattr(self,name,getattr(initialized_r4,name))
  self.action_pos=initialized_r4.action_pos
 def forward(self,encoding,noise):
  assert noise.ndim==3 and noise.shape[1:]==(16,6)
  prefix=self.prefix_proj(encoding['prefix'])
  if prefix.shape[0]==1:prefix=prefix.expand(noise.shape[0],-1,-1)
  action=self.noise_proj(noise)+self.action_pos
  tokens=torch.cat([prefix,action],1);b,length,_=tokens.shape;p=prefix.shape[1]
  mask=tokens.new_zeros(b,1,length,length)
  mask[:,:,:p,p:]=torch.finfo(tokens.dtype).min
  bias=encoding['prefix_bias'].expand(b,-1)
  mask[:,:,p:,:p]+=bias[:,None,None,:]
  valid=encoding['prefix_valid'].expand(b,-1)
  mask[:,:,:,:p]=mask[:,:,:,:p].masked_fill(~valid[:,None,None,:],torch.finfo(tokens.dtype).min)
  out=self.expert(inputs_embeds=tokens,attention_mask=mask,use_cache=False).last_hidden_state[:,-16:]
  return self.motion(out),self.grip(out)
