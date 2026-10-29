import torch
VERSION='r5_response_mean_v1'
def response_evidence(observed,free_prediction,follow_prediction,visibility,track_confidence,excitation,sigma):
 """Independent observed 3D trajectories; future tensors are labels only."""
 sigma=torch.as_tensor(sigma,device=observed.device,dtype=observed.dtype).clamp_min(1e-6)
 e0=(observed-free_prediction).norm(dim=-1)
 e1=(observed-follow_prediction).norm(dim=-1)
 response=torch.sigmoid((e0-e1)/sigma)
 confidence=visibility.float()*track_confidence.clamp(0,1)*(excitation.clamp_min(0)/(excitation.clamp_min(0)+sigma))
 finite=torch.isfinite(response)&torch.isfinite(confidence)
 return torch.where(finite,response,torch.zeros_like(response)),torch.where(finite,confidence,torch.zeros_like(confidence))
def aggregate_response(response,confidence,assignment,episode_id):
 """Normalize each independent demonstration before pooling; never take max."""
 assert assignment.shape[:-1]==response.shape
 numer=torch.zeros(assignment.shape[-1],device=assignment.device,dtype=assignment.dtype);denom=numer.clone()
 for e in torch.unique(episode_id):
  take=episode_id==e
  w=assignment[take]*confidence[take,None]
  mass=w.sum()
  if mass<=0:continue
  w=w/mass
  numer+=(w*response[take,None]).sum(0);denom+=w.sum(0)
 return numer/denom.clamp_min(1e-8),denom,denom>0
def restrict_future(current,indices,segment_ids,horizon=16):
 return (indices>current)&(indices<=current+horizon)&(segment_ids[indices]==segment_ids[current])
def require_supervision(record):
 assert record.get('teacher_version')==VERSION,'R5 requires independently rebuilt response supervision'
 for key in ['instance_masks','instance_valid','role_ids','response','response_confidence','response_assignment','predicted_target_history','causal_history_valid']:
  assert key in record,'missing R5 label contract: '+key
 assert record.get('future_in_inputs') is False
 assert record.get('history_source')=='causal_student_predictions'
