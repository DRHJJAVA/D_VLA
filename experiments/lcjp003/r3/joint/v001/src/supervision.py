import functools,json,pathlib
import numpy as np
import h5py
from scipy.spatial import cKDTree
from geometry import ROOT,RUN,CAMS

@functools.lru_cache(maxsize=80)
def episode_tracks(eid):
    op=ROOT/'experiments/contact_joint_object/cache/observations'/(eid+'.h5')
    rp=ROOT/'experiments/contact_joint_object/cache/rgbd'/(eid+'.h5')
    with h5py.File(op,'r') as o,h5py.File(rp,'r') as r:
        fi=r['frame_indices'][:];T=len(fi);md=json.loads(r.attrs['robot_metadata'])
        names=md['geom_names'];fingers=[]
        for name in ['gripper0_finger1_collision','gripper0_finger2_collision']:
            gi=names.index(name);verts=r['finger_surface/'+name+'/vertices'][:];faces=r['finger_surface/'+name+'/faces'][:]
            local=verts[faces].mean(1)
            rot=r['robot/geom_rotation'][:,gi];pos=r['robot/geom_position'][:,gi]
            fingers.append(np.einsum('nj,tij->tni',local,rot)+pos[:,None])
        fingers=np.concatenate(fingers,1);ee=r['robot/ee_position'][:]
        all_world=[];all_uv=[];all_good=[];view=[]
        for v,c in enumerate(CAMS):
            g=o['tracks/'+str(v)];world=g['world'][:];uv=g['uv'][:]
            good=g['valid'][:]&g['visible'][:]&np.isfinite(world).all(-1)&np.isfinite(uv).all(-1)
            pix=np.rint(np.nan_to_num(uv)).astype(int)
            inside=(pix[...,0]>=0)&(pix[...,0]<256)&(pix[...,1]>=0)&(pix[...,1]<256)
            good&=inside
            for t in range(T):
                ids=np.flatnonzero(good[t]);pp=pix[t,ids]
                good[t,ids]&=~r[c+'/robot_mask'][t][pp[:,1],pp[:,0]].astype(bool)
            all_world.append(world);all_uv.append(pix);all_good.append(good);view.extend([v]*world.shape[1])
        world=np.concatenate(all_world,1);uv=np.concatenate(all_uv,1);good=np.concatenate(all_good,1);view=np.array(view)
        evidence=np.zeros(good.shape,np.float32)
        for t in range(1,T):
            ids=np.flatnonzero(good[t]&good[t-1])
            if not len(ids):continue
            dt=max(int(fi[t]-fi[t-1]),1)
            movement=(world[t,ids]-world[t-1,ids])/dt;gripmotion=(ee[t]-ee[t-1])/dt
            speed=np.linalg.norm(movement,axis=-1)
            moving=speed/(speed+.001)
            agreement=np.exp(-.5*(np.linalg.norm(movement-gripmotion,axis=-1)/.01)**2)
            distance=cKDTree(fingers[t]).query(world[t,ids])[0]
            near=np.exp(-.5*(distance/.03)**2)
            evidence[t,ids]=moving*agreement*near
    return fi,world,uv,good,view,evidence

def supervision(e,t,objects,d):
    fi,world,uv,good,view,evidence=episode_tracks(e['id'])
    indices=np.flatnonzero(fi==t)
    if len(indices)!=1:raise ValueError('teacher_frame_missing')
    i=int(indices[0]);future=np.flatnonzero(fi>=t)
    time_weight=np.exp(-(fi[future]-t)/48.)[:,None]
    # Future is used ONLY in supervision. No seed-object ID or student prediction is read.
    w=(evidence[future]*time_weight).sum(0)/(good[future]*time_weight).sum(0).clip(1e-8)
    currently=good[i];global_mass=float(w[currently].sum())
    role=np.zeros(len(objects)+1,np.float32);parts=np.zeros((len(objects),2048),np.float32);quality=np.zeros(len(objects),np.float32)
    for j,o in enumerate(objects):
        member=np.zeros(len(view),bool)
        for v in [0,1]:
            ids=np.flatnonzero(currently&(view==v));p=uv[i,ids]
            member[ids]=o['masks'][v][p[:,1],p[:,0]]
        # Membership also uses metric surface correspondence; a union mask is not an object identity.
        current_ids=np.flatnonzero(currently)
        distance_to_surface=cKDTree(o['xyz']).query(world[i,current_ids])[0]
        metric=np.zeros(len(view),bool);metric[current_ids]=distance_to_surface<=.012
        member=metric
        ids=np.flatnonzero(member)
        if not len(ids):continue
        weights=w[ids];mass=float(weights.sum());mean=float(weights.mean())
        role[j]=mean*mass/max(global_mass,1e-8)
        quality[j]=mean
        xyz=d['xyz'][j].numpy()
        dist=np.linalg.norm(xyz[:,None,:]-world[i,ids][None],axis=-1)
        support=(np.exp(-.5*(dist/.025)**2)*weights[None]).sum(1)
        support*=d['valid'][j].numpy()
        if support.sum()>0:parts[j]=support/support.sum()
    total=float(role.sum())
    if total>1e-12:role/=total
    else:role[-1]=1.
    return {'role':role,'role_weight':float(quality.max()) if len(quality) else 0.,'parts':parts,'part_weight':quality,'valid_target_evidence':total>1e-12}

def prepare_episode(e):
    fi,world,uv,good,view,evidence=episode_tracks(e['id'])
    return {'episode':e['id'],'frames':len(fi),'tracked_points':world.shape[1],'max_interaction':float(evidence.max()),'seed_object_ids_used':False,'student_predictions_used':False}
