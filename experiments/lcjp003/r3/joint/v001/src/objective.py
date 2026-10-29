import json,functools
import numpy as np
import torch
from torch.nn import functional as F
from geometry import ROOT,RUN,frame_index
from supervision import supervision
from drifting import drift_loss
import h5py
from scipy.spatial.transform import Rotation

def targets(e,t,d,y,objects):
    s=supervision(e,t,objects,d)
    s={k:torch.as_tensor(v) for k,v in s.items()}
    s.update(target=y['target'],grip=y['grip'])
    with h5py.File(ROOT/'experiments/contact_joint_object/cache/rgbd'/(e['id']+'.h5'),'r') as h:
        ids=h['frame_indices'][:];current=frame_index(h,t)
        names=json.loads(h.attrs['robot_metadata'])['geom_names'];gi=names.index('gripper0_hand_collision')
        r0=h['robot/geom_rotation'][current,gi];p0=h['robot/ee_position'][current]
        out=[]
        for off in [1,4,16]:
            ix=frame_index(h,min(t+off,int(ids[-1])))
            pos=(h['robot/ee_position'][ix]-p0)@r0/.1
            rot=Rotation.from_matrix(r0.T@h['robot/geom_rotation'][ix,gi]).as_rotvec()
            width=np.abs(h['robot/gripper_qpos'][ix]).sum()/.08
            out.extend(np.r_[pos,rot,width].tolist())
        s['aux']=torch.tensor(out,dtype=torch.float32)
        end=h['robot/ee_position'][-1]
    # Placement supervision is a weak geometric endpoint-role distribution, separate from target.
    box=d['box'].numpy();delta=np.maximum(np.abs(box[:,:3]-end)-box[:,3:]/2,0)
    dist=np.linalg.norm(delta,axis=-1);dest=np.exp(-.5*(dist/.08)**2)*(1-s['role'][:-1].numpy())
    total=dest.sum()
    dq=np.r_[dest/(total+1e-9),0.] if total>1e-6 else np.r_[np.zeros(len(objects)),1.]
    s['destination']=torch.tensor(dq,dtype=torch.float32)
    s['destination_weight']=torch.tensor(float(np.exp(-.5*(dist.min()/.08)**2)))
    return s

def loss(model,d,y,noise,selection_generator=None):
    e=model.encode(d)
    role=-(y['role'].to(e['roles'])*e['roles'].log_softmax(-1)).sum()*y['role_weight']
    dest=-(y['destination']*e['destinations'].log_softmax(-1)).sum()*y['destination_weight']
    part=-(y['parts']*e['alpha'].clamp_min(1e-12).log()).sum(-1)
    weights=y['part_weight']
    part=(part*weights).sum()/(weights.sum()+1e-8)*y['role_weight']
    aux=(F.smooth_l1_loss(e['aux'],y['aux'].expand_as(e['aux']),reduction='none').mean(-1)*y['role'][:-1]).sum()*y['role_weight']
    # One categorical draw is an unbiased estimator of per-object action loss under q.
    # It never averages coordinates or features of different objects.
    chosen=int(torch.multinomial(y['role'].float(),1,generator=selection_generator))
    action,grip=model.generate(e,noise,object_index=chosen)
    drift=drift_loss(action.flatten(1)[None],y['target'].flatten()[None,None]).mean()
    grip_loss=F.cross_entropy(grip.flatten(0,1),y['grip'].expand(len(noise),-1).flatten())
    total=drift+grip_loss+.5*role+.1*dest+.5*part+.1*aux
    return total,{'drift':float(drift.detach()),'grip':float(grip_loss.detach()),'role':float(role.detach()),'part':float(part.detach()),'aux':float(aux.detach()),'lambda_mean':float(e['gate'].detach().mean()),'role_weight':float(y['role_weight']),'chosen_target':chosen,'null_probability':float(e['roles'].detach().softmax(-1)[-1])}
