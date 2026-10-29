import json, pathlib, hashlib
import numpy as np
import h5py
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation
from visibility import visible_reference,fuse_visible_features
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA')
OLD=ROOT/'experiments/lcjp003/r2/v2/v001'
RUN=ROOT/'experiments/lcjp003/r3/joint/v001'
CAMS=['agentview','robot0_eye_in_hand']

def fps(x,n):
    n=min(n,len(x)); out=np.zeros(n,dtype=np.int64); ds=np.full(len(x),np.inf)
    p=int(np.argmax(((x-x.mean(0))**2).sum(1)))
    for j in range(n):
        out[j]=p; ds=np.minimum(ds,((x-x[p])**2).sum(1)); p=int(ds.argmax())
    return out

def frame_index(h,t):
    ids=h['frame_indices'][:]; found=np.flatnonzero(ids==t)
    if len(found)!=1: raise ValueError('frame_index_not_exact')
    return int(found[0])

def load_objects(e,t,limit=2048):
    op=OLD/'cache'/e['id']/('observations_'+str(t)+'.h5')
    if not op.exists():
        shared=OLD/'cache'/e['id']/'observations.h5'
        if shared.exists():
            with h5py.File(shared,'r') as check:
                if 'frames/'+str(t) in check:op=shared
        if not op.exists():op=RUN/'recovered_observations/cache'/e['id']/('observations_'+str(t)+'.h5')
    rp=ROOT/'experiments/contact_joint_object/cache/rgbd'/(e['id']+'.h5')
    objs=[]
    with h5py.File(op,'r') as obs,h5py.File(rp,'r') as rgb:
        ix=frame_index(rgb,t); f=obs['frames/'+str(t)]
        features=[f['features_'+str(v)][:].astype(np.float32) for v in range(2)]
        rm=[rgb[c+'/robot_mask'][ix].astype(bool) for c in CAMS]
        imgs=[rgb[c+'/rgb'][ix] for c in CAMS]
        cameras=[dict(K=rgb[c+'/K'][ix],camera_to_world=rgb[c+'/camera_to_world'][ix],depth=rgb[c+'/depth_m'][ix],robot_mask=rm[v]) for v,c in enumerate(CAMS)]
        for key in sorted([k for k in f if k.startswith('object_')],key=lambda x:int(x.split('_')[1])):
            o=f[key]; uv=o['uv'][:]; cam=o['camera'][:].astype(int); xyz=o['xyz'][:]; normal=o['normal'][:]
            keep=np.ones(len(xyz),bool)
            for v in range(2):
                sel=cam==v; keep[sel]=~rm[v][uv[sel,1],uv[sel,0]]
            removed=int((~keep).sum()); total=len(keep)
            if keep.sum()<3: continue
            xyz=xyz[keep]; normal=normal[keep]; uv=uv[keep]; cam=cam[keep]
            # Retain masks as exact observed pixels; all point sampling uses actual observed points.
            masks=[]
            for v in range(2):
                m=np.zeros((256,256),bool); sel=cam==v; m[uv[sel,1],uv[sel,0]]=True; masks.append(m)
            ids=fps(xyz,min(limit,len(xyz)))
            xyz=xyz[ids]; normal=normal[ids]; uv=uv[ids]; cam=cam[ids]
            feat=np.zeros((len(xyz),384),np.float32); col=np.zeros((len(xyz),3),np.float32)
            for v in range(2):
                s=cam==v
                feat[s]=features[v][np.minimum(uv[s,1]//8,31),np.minimum(uv[s,0]//8,31)]
                col[s]=imgs[v][uv[s,1],uv[s,0]]/255.
            reference=visible_reference(xyz,cameras)
            vf=np.zeros((len(xyz),2,384),np.float32);vc=np.zeros((len(xyz),2,3),np.float32)
            for v in range(2):
                present=reference['visible'][:,v];p_uv=reference['uv'][present,v]
                vf[present,v]=features[v][np.minimum(p_uv[:,1]//8,31),np.minimum(p_uv[:,0]//8,31)]
                vc[present,v]=imgs[v][p_uv[:,1],p_uv[:,0]]/255.
            observed=~reference['current_unknown']
            if observed.sum()<3:continue
            feat=fuse_visible_features(vf,reference)[observed]
            col=fuse_visible_features(vc,reference)[observed]
            xyz=xyz[observed];normal=normal[observed];uv=uv[observed];cam=cam[observed]
            objs.append(dict(ids=[int(key.split('_')[1])],xyz=xyz,normal=normal,features=feat,rgb=col,uv=uv,camera=cam,masks=masks,removed_robot=removed,total=total))
        # Conservative same-frame cross-camera association by metric overlap and reciprocal best.
        # These thresholds define observation correspondence, not contact labels.
        candidates=[]
        for a in range(len(objs)):
            for b in range(a+1,len(objs)):
                x,y=objs[a],objs[b]
                if set(x['camera'])==set(y['camera']): continue
                if np.linalg.norm(x['xyz'].mean(0)-y['xyz'].mean(0))>.3: continue
                da=cKDTree(x['xyz']).query(y['xyz'])[0]; db=cKDTree(y['xyz']).query(x['xyz'])[0]
                overlap=min(float((da<.006).mean()),float((db<.006).mean()))
                va=x['features'].mean(0); vb=y['features'].mean(0)
                cos=float(va@vb/(np.linalg.norm(va)*np.linalg.norm(vb)+1e-9))
                if overlap>=.20: candidates.append((overlap*(.75+.25*max(0.,cos)),a,b))
        best={}
        for s,a,b in candidates:
            for j in [a,b]:
                if j not in best or s>best[j][0]:best[j]=(s,a,b)
        used=set(); merged=[]; pairs=[]
        for s,a,b in sorted(candidates,reverse=True):
            if a in used or b in used: continue
            if best[a][1:]!=(a,b) or best[b][1:]!=(a,b): continue
            x,y=objs[a],objs[b]; z={}
            for key in ['xyz','normal','features','rgb','uv','camera']:z[key]=np.concatenate([x[key],y[key]])
            for key in ['ids']:z[key]=x[key]+y[key]
            z['masks']=[x['masks'][v]|y['masks'][v] for v in range(2)]
            z['removed_robot']=x['removed_robot']+y['removed_robot'];z['total']=x['total']+y['total']
            ids=fps(z['xyz'],min(limit,len(z['xyz'])))
            for key in ['xyz','normal','features','rgb','uv','camera']:z[key]=z[key][ids]
            merged.append(z);used.update([a,b]);pairs.append({'candidate_ids':z['ids'],'correspondence_score':s})
        merged.extend([o for j,o in enumerate(objs) if j not in used])
        refined=[]
        for obj in merged:
            points=obj['xyz'];edge=cKDTree(points).query_pairs(.008,output_type='ndarray')
            graph=coo_matrix((np.ones(len(edge)),(edge[:,0],edge[:,1])),shape=(len(points),len(points)))
            count,labels=connected_components(graph,directed=False)
            groups=[np.flatnonzero(labels==j) for j in range(count)]
            large=[g for g in groups if len(g)>=max(3,.05*len(points))]
            separate=False
            for a in range(len(large)):
                for b in range(a+1,len(large)):
                    pa,pb=points[large[a]],points[large[b]]
                    if np.linalg.norm(pa.mean(0)-pb.mean(0))>.5*(np.linalg.norm(np.ptp(pa,axis=0))+np.linalg.norm(np.ptp(pb,axis=0)))+.01:separate=True
            if separate:
                distances=np.stack([cKDTree(points[g]).query(points)[0] for g in large],1)
                assignment=distances.argmin(1)
                for j,g in enumerate(large):
                    take=np.flatnonzero(assignment==j)
                    z={k:(v[take] if k in ['xyz','normal','features','rgb','uv','camera'] else v) for k,v in obj.items()}
                    z['component']=j
                    masks=[]
                    for v in range(2):
                        mask=np.zeros((256,256),bool);seen=z['camera']==v;p_uv=z['uv'][seen]
                        mask[p_uv[:,1],p_uv[:,0]]=True
                        masks.append(binary_dilation(mask,iterations=1)&obj['masks'][v])
                    z['masks']=masks
                    refined.append(z)
            else:
                obj['component']=0;refined.append(obj)
        # Collapse reciprocal surface duplicates after separating disconnected objects.
        accepted=[]
        for obj in refined:
            match=None
            for j,other in enumerate(accepted):
                da=cKDTree(obj['xyz']).query(other['xyz'])[0]
                db=cKDTree(other['xyz']).query(obj['xyz'])[0]
                low=min(float((da<.006).mean()),float((db<.006).mean()))
                high=max(float((da<.006).mean()),float((db<.006).mean()))
                if low>=.5 and high>=.85:match=j;break
            if match is None:accepted.append(obj);continue
            other=accepted[match]
            for key in ['xyz','normal','features','rgb','uv','camera']:
                other[key]=np.concatenate([other[key],obj[key]])
            keep=fps(other['xyz'],min(limit,len(other['xyz'])))
            for key in ['xyz','normal','features','rgb','uv','camera']:other[key]=other[key][keep]
            other['ids']=sorted(set(other['ids']+obj['ids']))
            other['masks']=[other['masks'][v]|obj['masks'][v] for v in range(2)]
        merged=accepted
        metadata=json.loads(rgb.attrs['robot_metadata']); names=metadata['geom_names']
        finger_xyz=[];finger_norm=[]
        for name in ['gripper0_finger1_collision','gripper0_finger2_collision']:
            gi=names.index(name); verts=rgb['finger_surface/'+name+'/vertices'][:]; faces=rgb['finger_surface/'+name+'/faces'][:]
            tri=verts[faces]; n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n=n/(np.linalg.norm(n,axis=1,keepdims=True)+1e-9)
            rot=rgb['robot/geom_rotation'][ix,gi];pos=rgb['robot/geom_position'][ix,gi]
            finger_xyz.append(tri.mean(1)@rot.T+pos);finger_norm.append(n@rot.T)
        ee=rgb['robot/ee_position'][ix]
        gi=names.index('gripper0_hand_collision'); rotation=rgb['robot/geom_rotation'][ix,gi]
        fingers=np.concatenate([np.concatenate(finger_xyz)-ee,np.concatenate(finger_norm)],-1).astype(np.float32)
        fingers[:,:3]=fingers[:,:3]@rotation; fingers[:,3:]=fingers[:,3:]@rotation
        return merged,dict(frame=t,index=ix,ee=ee,rotation=rotation,fingers=fingers,crossview_pairs=pairs)

def policy_input(e,t):
    objects,meta=load_objects(e,t)
    if not objects:raise ValueError('no_valid_observed_objects')
    prev=torch.load(OLD/'cache'/e['id']/(str(t)+'.pt'),map_location='cpu',weights_only=False)
    fields={k:[] for k in ['x','xyz','normal','valid','box','anchors','knn','interp','interp_weight','dual_visible']}
    for o in objects:
        n=len(o['xyz']);valid=np.arange(2048)<n
        ids=np.minimum(np.arange(2048),n-1)
        xyz=o['xyz'][ids]; normals=o['normal'][ids];center=o['xyz'].mean(0);extent=np.ptp(o['xyz'],axis=0)
        rel=(xyz-meta['ee'])@meta['rotation']; unit=(xyz-center)/max(float(np.linalg.norm(extent)),1e-4)
        x=np.concatenate([o['features'][ids],unit,rel,normals,o['rgb'][ids]],-1)
        anc=fps(o['xyz'],64);anc=np.pad(anc,(0,64-len(anc)),mode='edge')
        tree=cKDTree(o['xyz']);_,kn=tree.query(xyz[anc],k=min(32,n))
        if kn.ndim==1:kn=kn[:,None]
        kn=np.pad(kn,((0,0),(0,32-kn.shape[1])),mode='edge')
        di,inter=cKDTree(xyz[anc]).query(xyz,k=3);iw=1/np.maximum(di,1e-6);iw/=iw.sum(1,keepdims=True)
        vals=dict(x=x,xyz=xyz,normal=normals,valid=valid,box=np.r_[center,extent],anchors=anc,knn=kn,interp=inter,interp_weight=iw,dual_visible=float(len(set(o['camera']))==2))
        for k,v in vals.items():fields[k].append(v)
    out={k:torch.from_numpy(np.array(v)).to(torch.bool if k=='valid' else torch.long if k in ['anchors','knn','interp'] else torch.float32) for k,v in fields.items()}
    out.update({k:prev[k] for k in ['h','words']})
    out['fingers']=torch.from_numpy(meta['fingers'])
    # Supervision returned separately; there is no future tensor in the policy interface.
    target={k:prev[k] for k in ['target','grip']}
    target['episode']=e['id'];target['frame']=t
    return out,target,objects,meta
