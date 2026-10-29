"""LCJP003 P0 causal two-training-episode RGB-D adapter. Raw sources are read-only."""
from pathlib import Path
from collections import OrderedDict
import json
import numpy as np
import h5py

CAMERAS = ('agentview', 'robot0_eye_in_hand')
INPUT_PROTOCOL_VERSION = 'p0_actual_gl_depth_sample0_v2'
DEPTH_SAMPLE_OFFSET = np.array([0.375, 0.875], dtype=np.float32)
RGB_PIXEL_CENTER_OFFSET = np.array([0.5, 0.5], dtype=np.float32)
P0_EPISODES = ('pick_up_the_bbq_sauce_and_place_it_in_the_basket__demo_0',
               'pick_up_the_alphabet_soup_and_place_it_in_the_basket__demo_0')


def fps(xyz, count):
    if len(xyz) <= count:
        return np.arange(len(xyz), dtype=np.int64)
    ids = np.empty(count, np.int64)
    ids[0] = np.argmin(np.linalg.norm(xyz - xyz.mean(0), axis=1))
    dist = np.full(len(xyz), np.inf)
    for i in range(count):
        if i:
            ids[i] = np.argmax(dist)
        dist = np.minimum(dist, ((xyz - xyz[ids[i]]) ** 2).sum(1))
        dist[ids[:i+1]] = -1
    return ids


def backproject(depth, K, C, uv):
    z = depth[uv[:, 1], uv[:, 0]]
    ray_uv = uv + DEPTH_SAMPLE_OFFSET
    xyz = np.column_stack(((ray_uv[:, 0]-K[0, 2])*z/K[0, 0],
                           (ray_uv[:, 1]-K[1, 2])*z/K[1, 1], z))
    return (xyz @ C[:3, :3].T + C[:3, 3]).astype(np.float32)


class P0Dataset:
    def __init__(self, root='/mnt/nvme1/drh/D_VLA'):
        self.root = Path(root)
        self.input_protocol_version = INPUT_PROTOCOL_VERSION
        self.adapter_sha256 = __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()
        self.old = self.root/'experiments/contact_joint_object'
        rows = json.loads((self.old/'manifests/episodes.json').read_text())['episodes']
        self.rows = {r['id']: r for r in rows if r['id'] in P0_EPISODES}
        if len(self.rows) != 2 or any(r['split'] != 'train' or r['episode'] != 'demo_0' for r in self.rows.values()):
            raise ValueError('P0 requires exactly the two authorized training demo_0 episodes')
        self.actions, self.instructions = {}, {}
        for eid, row in self.rows.items():
            with h5py.File(self.root/row['raw'], 'r') as h:
                self.actions[eid] = h['data/demo_0/actions'][:].astype(np.float32)
                info = json.loads(h['data'].attrs['problem_info'])
                self.instructions[eid] = info['language_instruction']
        a = np.concatenate(list(self.actions.values()), axis=0)
        self.action_mean = a.mean(0).astype(np.float32)
        self.action_std = np.maximum(a.std(0), 1e-6).astype(np.float32)
        self.normalization = {'scope': 'P0 two training demo_0 episodes only', 'count': len(a),
                              'mean': self.action_mean.tolist(), 'std': self.action_std.tolist(),
                              'clip': False, 'heldout_or_dev_used': False}
        self.read_log = []
        self._cache = OrderedDict()

    def __len__(self):
        return sum(len(a) for a in self.actions.values())

    def frame(self, episode_id, frame_index, n_points=2048):
        if episode_id not in self.rows:
            raise ValueError('Episode not authorized for P0')
        if not 0 <= frame_index < len(self.actions[episode_id]):
            raise IndexError('Frame out of range')
        self.read_log.append({'episode': episode_id, 'frame': int(frame_index)})
        key = (episode_id, int(frame_index), int(n_points))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        with h5py.File(self.old/'cache/rgbd'/f'{episode_id}.h5', 'r') as h, h5py.File(self.old/'cache/observations'/f'{episode_id}.h5', 'r') as obs:
            if h['frame_indices'][frame_index] != frame_index:
                raise ValueError('RGB-D frame indexing mismatch')
            rgb = np.stack([h[c+'/rgb'][frame_index] for c in CAMERAS])
            depth = np.stack([h[c+'/depth_m'][frame_index] for c in CAMERAS])
            K = np.stack([h[c+'/K'][frame_index] for c in CAMERAS])
            C = np.stack([h[c+'/camera_to_world'][frame_index] for c in CAMERAS])
            robot_mask = np.stack([h[c+'/robot_mask'][frame_index] != 0 for c in CAMERAS])
            body_p = h['robot/body_position'][frame_index]
            body_r = h['robot/body_rotation'][frame_index]
            joint = h['robot/joint_qpos'][frame_index]
            grip_q = h['robot/gripper_qpos'][frame_index]
            names = json.loads(h.attrs['robot_metadata'])['body_names']
            idx = {x: names.index(x) for x in ('robot0_base','gripper0_eef','gripper0_leftfinger','gripper0_rightfinger')}
            base_p, base_r = body_p[idx['robot0_base']], body_r[idx['robot0_base']]
            grip_p, grip_r = body_p[idx['gripper0_eef']], body_r[idx['gripper0_eef']]
            eef_base = (grip_p-base_p) @ base_r
            eef_rot = base_r.T @ grip_r
            state = np.concatenate((joint, grip_q, eef_base, eef_rot[:, :2].T.reshape(-1))).astype(np.float32)
            fg = obs[f'frames/{frame_index}']
            candidates = []
            claimed = np.zeros_like(depth, dtype=bool)
            covered = np.zeros_like(depth, dtype=bool)
            keys = sorted((k for k in fg if k.startswith('object_')), key=lambda k: (-float(fg[k].attrs.get('sam_quality',0)), int(k.split('_')[1])))
            for name in keys:
                g = fg[name]
                for ci in range(2):
                    mk = f'mask_{ci}'
                    if mk in g:
                        covered[ci] |= np.unpackbits(g[mk][:])[:depth.shape[1]*depth.shape[2]].reshape(depth.shape[1:]).astype(bool)
                uv = g['uv'][:].astype(np.int64)
                camera = g['camera'][:].astype(np.int64)
                xyz = np.zeros((len(uv),3),np.float32)
                normal = np.zeros_like(xyz)
                # Cached xyz/normal used integer pixel rays; rebuild from current measured depth.
                for ci in range(2):
                    at = np.flatnonzero(camera==ci)
                    if not len(at): continue
                    pixel = uv[at]
                    xyz[at] = backproject(depth[ci],K[ci],C[ci],pixel)
                    right = pixel.copy(); right[:,0] = np.minimum(right[:,0]+1,depth.shape[2]-1)
                    down = pixel.copy(); down[:,1] = np.minimum(down[:,1]+1,depth.shape[1]-1)
                    normal[at] = np.cross(backproject(depth[ci],K[ci],C[ci],right)-xyz[at],
                                          backproject(depth[ci],K[ci],C[ci],down)-xyz[at])
                    lengths = np.linalg.norm(normal[at],axis=1,keepdims=True)
                    normal[at] = normal[at]/np.maximum(lengths,1e-12)
                keep = np.isfinite(xyz).all(1) & np.isfinite(normal).all(1)
                keep &= ~claimed[camera, uv[:,1], uv[:,0]]
                xyz, uv, camera, normal = xyz[keep],uv[keep],camera[keep],normal[keep]
                if len(xyz):
                    claimed[camera,uv[:,1],uv[:,0]] = True
                    candidates.append((int(name.split('_')[1]),xyz,uv,camera,normal))
            for ci in range(2):
                free = np.isfinite(depth[ci]) & (depth[ci]>0) & ~robot_mask[ci] & ~covered[ci]
                vu = np.argwhere(free)
                if len(vu):
                    vu = vu[np.linspace(0,len(vu)-1,min(len(vu),2048),dtype=np.int64)]
                    uv = vu[:,::-1].copy()
                    xyz = backproject(depth[ci],K[ci],C[ci],uv)
                    candidates.append((1000000+2*frame_index+ci,xyz,uv,np.full(len(uv),ci),np.zeros_like(xyz)))
            if not candidates:
                raise ValueError('No valid observed points')
            # Equal candidate budgets avoid dropping a small object; all points remain genuine depth observations.
            budgets = np.zeros(len(candidates),np.int64)
            remaining = n_points
            while remaining and any(budgets[i] < len(c[1]) for i,c in enumerate(candidates)):
                for i,c in enumerate(candidates):
                    if remaining and budgets[i] < len(c[1]):
                        budgets[i] += 1; remaining -= 1
            fields = {x: [] for x in ('world','uv','camera','normal','object_id','xyz_norm','scale')}
            for (oid, xyz, uv, camera, normal), count in zip(candidates,budgets):
                if not count: continue
                selected = fps(xyz,int(count))
                center = xyz.mean(0)
                scale = max(float(np.linalg.norm(xyz.max(0)-xyz.min(0))),1e-6)
                for k,v in [('world',xyz[selected]),('uv',uv[selected]),('camera',camera[selected]),('normal',normal[selected]),('object_id',np.full(count,oid)),('xyz_norm',(xyz[selected]-center)/scale),('scale',np.full(count,scale))]:
                    fields[k].append(v)
            d = {k:np.concatenate(v) for k,v in fields.items()}
            valid_count = len(d['world']); xyz = d['world']
            nrm = d['normal']; nn = np.linalg.norm(nrm,axis=1)
            nvalid = nn > 1e-6
            nrm = nrm/np.maximum(nn[:,None],1e-6)
            projected_uv = np.zeros((valid_count,2,2),np.float32)
            projected_depth_uv = np.zeros_like(projected_uv)
            view_visible = np.zeros((valid_count,2),bool)
            residual = np.full((valid_count,2),np.nan,np.float32)
            for ci in range(2):
                q = (xyz-C[ci,:3,3])@C[ci,:3,:3]
                z = q[:,2]
                uvp = q[:,:2]/np.maximum(z[:,None],1e-9)*[K[ci,0,0],K[ci,1,1]]+[K[ci,0,2],K[ci,1,2]]
                projected_uv[:,ci] = uvp - RGB_PIXEL_CENTER_OFFSET
                projected_depth_uv[:,ci] = uvp - DEPTH_SAMPLE_OFFSET
                pix = np.rint(projected_depth_uv[:,ci]).astype(np.int64)
                inside = (z>0)&(pix[:,0]>=0)&(pix[:,0]<depth.shape[2])&(pix[:,1]>=0)&(pix[:,1]<depth.shape[1])
                ids = np.flatnonzero(inside)
                measured = depth[ci,pix[ids,1],pix[ids,0]]
                residual[ids,ci] = abs(measured-z[ids])
                view_visible[ids,ci] = (np.isfinite(measured)&(measured>0)&~robot_mask[ci,pix[ids,1],pix[ids,0]]&(residual[ids,ci] <= np.maximum(.005,.02*measured)))
            loc = [(xyz-body_p[idx[name]])@body_r[idx[name]] for name in ('gripper0_eef','gripper0_leftfinger','gripper0_rightfinger')]
            opening = np.full((valid_count,1),float(abs(grip_q[0]-grip_q[1])),np.float32)
            relation = np.concatenate((*loc,nrm@grip_r,opening,d['scale'][:,None],nvalid[:,None],view_visible.all(1)[:,None]),axis=1).astype(np.float32)
            result = {'xyz_base':((xyz-base_p)@base_r).astype(np.float32),'xyz_norm':d['xyz_norm'].astype(np.float32),
                      'relation':relation,'object_id':d['object_id'].astype(np.int64),'point_valid':np.ones(valid_count,bool),
                      'rgb':rgb,'uv':(d['uv']+DEPTH_SAMPLE_OFFSET-RGB_PIXEL_CENTER_OFFSET).astype(np.float32),
                      'depth_uv':d['uv'].astype(np.float32),'camera_id':d['camera'].astype(np.int64),
                      'projected_uv':projected_uv,'projected_depth_uv':projected_depth_uv,
                      'view_visible':view_visible,'depth_residual':residual,
                      'state':state,'K':K,'camera_to_world':C,'frame_index':np.int64(frame_index)}
            for name in ('xyz_base','xyz_norm','relation','object_id','point_valid','uv','depth_uv','camera_id','projected_uv','projected_depth_uv','view_visible','depth_residual'):
                if valid_count<n_points:
                    value = -1 if name in ('object_id','camera_id') else 0
                    pad = [(0,n_points-valid_count)]+[(0,0)]*(result[name].ndim-1)
                    result[name] = np.pad(result[name],pad,constant_values=value)
            self._cache[key] = result
            while len(self._cache)>128:
                self._cache.popitem(last=False)
            return result

    def load_window(self, episode_id, current_frame, history=16, horizon=16, n_points=2048):
        if episode_id not in self.rows or not 0<=current_frame<len(self.actions[episode_id]):
            raise ValueError('Unauthorized episode or invalid current frame')
        indices = list(range(max(0,current_frame-history+1),current_frame+1))
        records = [self.frame(episode_id,t,n_points) for t in indices]
        missing = history-len(records)
        tensor_names = ('xyz_base','xyz_norm','relation','object_id','point_valid','rgb','uv','depth_uv','camera_id','projected_uv','projected_depth_uv','view_visible','state','K','camera_to_world')
        batch = {}
        for k in tensor_names:
            value = np.stack([r[k] for r in records])
            if missing:
                pad = [(missing,0)]+[(0,0)]*(value.ndim-1)
                value = np.pad(value,pad,constant_values=-1 if k in ('object_id','camera_id') else 0)
            batch[k] = value
        batch['history_valid'] = np.array([False]*missing+[True]*len(records))
        batch['frame_indices'] = np.array([-1]*missing+indices,np.int64)
        # Reindex candidate identities only using the observed past and present.
        ids = np.unique(batch['object_id'][batch['point_valid']])
        idmap = {int(oid):i for i,oid in enumerate(ids)}
        original = batch['object_id'].copy()
        for oid,newid in idmap.items(): batch['object_id'][original==oid]=newid
        batch['candidate_original_ids'] = ids.astype(np.int64)
        current_ids = np.unique(batch['object_id'][-1,batch['point_valid'][-1]])
        ngroups = max(16,len(current_ids))
        group = np.full(n_points,-1,np.int64); group_start=0
        for j,oid in enumerate(current_ids):
            at = np.flatnonzero((batch['object_id'][-1]==oid)&batch['point_valid'][-1])
            count = min(len(at),ngroups//len(current_ids)+(j<ngroups%len(current_ids)))
            xyz = batch['xyz_base'][-1,at]
            center = xyz[fps(xyz,count)]
            label = ((xyz[:,None]-center[None])**2).sum(-1).argmin(1)
            group[at]=group_start+label; group_start += count
        batch['group_id'] = group
        a = self.actions[episode_id][current_frame:current_frame+horizon]
        av = np.arange(horizon)<len(a)
        raw = np.zeros((horizon,self.actions[episode_id].shape[1]),np.float32); raw[:len(a)]=a
        target = np.zeros_like(raw); target[av]=(raw[av]-self.action_mean)/self.action_std
        batch.update({'actions':target,'actions_raw':raw,'action_valid':av,'episode_id':episode_id,
                      'current_frame':int(current_frame),'instruction':self.instructions[episode_id],
                      'input_protocol_version':self.input_protocol_version,'adapter_sha256':self.adapter_sha256})
        return batch

    def save_window(self, path, **kwargs):
        window=self.load_window(**kwargs)
        arrays={k:v for k,v in window.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(path,**arrays)
        return {k:v for k,v in window.items() if not isinstance(v,np.ndarray)}
