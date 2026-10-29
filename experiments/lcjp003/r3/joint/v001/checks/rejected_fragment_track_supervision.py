import pathlib,json,math,time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from geometry import load_objects,OLD,RUN,ROOT,fps

def rigid(a,b):
    rot=np.eye(3);shift=b.mean(0)-a.mean(0)
    for _ in range(5):
        moved=a@rot.T+shift
        ds,ids=cKDTree(b).query(moved)
        w=np.exp(-.5*(ds/.015)**2);ws=w.sum()+1e-9
        ca=(a*w[:,None]).sum(0)/ws; cb=(b[ids]*w[:,None]).sum(0)/ws
        u,s,v=np.linalg.svd((a-ca).T@((b[ids]-cb)*w[:,None]))
        rr=v.T@u.T
        if np.linalg.det(rr)<0:v[-1]*=-1;rr=v.T@u.T
        rot=rr;shift=cb-ca@rot.T
    return rot,shift,float(w.mean())

def prepare_episode(e):
    out=RUN/'cache'/(e['id']+'_tracks.npz')
    if out.exists():return {'episode':e['id'],'cached':True}
    frames=sorted(int(p.stem.split('_')[1]) for p in (OLD/'cache'/e['id']).glob('observations_*.h5'))
    tracks={};obs=[];nextid=0;previous=[]
    for frame in frames:
        objects,meta=load_objects(e,frame,2048)
        for o in objects:
            take=fps(o['xyz'],128)
            for field in ['xyz','features']:o[field]=o[field][take]
        cur=[]
        for o in objects:
            cur.append(dict(ids=o['ids'],xyz=o['xyz'],desc=o['features'].mean(0),center=o['xyz'].mean(0),size=np.ptp(o['xyz'],axis=0),ee=meta['ee'],fingers=meta['fingers'][:,:3]@meta['rotation'].T+meta['ee'],frame=frame))
        assignments={}
        if previous and cur:
            cost=np.zeros((len(previous),len(cur)))
            for a,p in enumerate(previous):
                for b,c in enumerate(cur):
                    cos=float(p['desc']@c['desc']/(np.linalg.norm(p['desc'])*np.linalg.norm(c['desc'])+1e-9))
                    distance=np.linalg.norm(p['center']-c['center'])
                    size=np.linalg.norm(p['size']-c['size'])
                    cost[a,b]=1-cos+distance/.1+size/.2
            rows,cols=linear_sum_assignment(cost)
            for a,b in zip(rows,cols):
                if cost[a,b]<2.0:assignments[b]=(previous[a]['track'],float(np.exp(-cost[a,b])))
        for b,c in enumerate(cur):
            if b in assignments:
                c['track'],quality=assignments[b];p=tracks[c['track']][-1]
                rot,shift,fit=rigid(p['xyz'],c['xyz'])
                c['R']=rot@p['R'];c['T']=p['T']@rot.T+shift
                c['quality']=quality*fit
            else:
                c['track']=nextid;nextid+=1;c['R']=np.eye(3);c['T']=np.zeros(3);c['quality']=0.
            tracks.setdefault(c['track'],[]).append(c)
        previous=cur;obs.extend(cur)
    records=[]
    for tid,seq in tracks.items():
        centers=np.array([v['center'] for v in seq]);ee=np.array([v['ee'] for v in seq])
        for j,c in enumerate(seq):
            if j:
                dt=max(c['frame']-seq[j-1]['frame'],1)
                do=(centers[j]-centers[j-1])/dt;dg=(ee[j]-ee[j-1])/dt
                moving=np.linalg.norm(do)/(np.linalg.norm(do)+.001)
                agreement=np.exp(-np.sum((do-dg)**2)/(2*.01**2))
                near=np.exp(-.5*(cKDTree(c['xyz']).query(c['fingers'])[0].min()/.03)**2)
                c['interaction']=float(c['quality']*moving*agreement*near)
            else:c['interaction']=0.
            records.append(dict(frame=c['frame'],ids=c['ids'],track=tid,R=c['R'].tolist(),T=c['T'].tolist(),quality=c['quality'],interaction=c['interaction'],center=c['center'].tolist(),fingers=c['fingers'].tolist()))
    payload={'episode':e['id'],'records':records,'protocol':'independent_observed_motion_v1','contact_truth_used':False,'student_predictions_used':False}
    tmp=out.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(payload,f)
    tmp.replace(out.with_suffix('.json'))
    # Sentinel stores no model-dependent target.
    np.savez(out,frames=np.array(frames),track_count=nextid)
    return {'episode':e['id'],'frames':len(frames),'tracks':nextid,'records':len(records),'max_interaction':max([v['interaction'] for v in records] or [0])}

def supervision(e,t,objects,d):
    p=RUN/'cache'/(e['id']+'_tracks.json')
    if not p.exists():prepare_episode(e)
    rec=json.loads(p.read_text())['records'];current=[v for v in rec if v['frame']==t]
    k=len(objects);role=np.zeros(k+1,np.float32);parts=np.zeros((k,2048),np.float32);quality=np.zeros(k,np.float32)
    for j,o in enumerate(objects):
        match=[c for c in current if set(c['ids'])==set(o['ids'])]
        if len(match)!=1:continue
        c=match[0];R0=np.array(c['R']);T0=np.array(c['T'])
        future=[v for v in rec if v['track']==c['track'] and v['frame']>=t]
        scores=[];weights=[]
        xyz=d['xyz'][j].numpy()
        for f in future:
            w=f['interaction']*np.exp(-(f['frame']-t)/48.)
            if w<=1e-8:continue
            rf=np.array(f['R']);tf=np.array(f['T'])
            finger=(np.array(f['fingers'])-tf)@rf
            finger=finger@R0.T+T0
            distance=cKDTree(finger).query(xyz)[0]
            support=np.exp(-.5*(distance/.025)**2)
            # A neighborhood around the two finger surfaces supports the whole local part.
            scores.append(support*w);weights.append(w)
        if weights:
            parts[j]=np.sum(scores,axis=0)/(sum(weights)+1e-9)
            parts[j]*=d['valid'][j].numpy()
            quality[j]=max(weights)
            role[j]=sum(weights)/len(weights)
    mass=float(role[:-1].sum())
    if mass>0:
        role[:-1]/=mass
        # Evidence strength controls supervision weight, not fabricated null labels.
    else:role[-1]=1.
    den=parts.sum(-1,keepdims=True)
    parts=np.divide(parts,den,out=np.zeros_like(parts),where=den>0)
    role_weight=float(quality.max()) if len(quality) else 0.
    return {'role':role,'role_weight':role_weight,'parts':parts,'part_weight':quality,'valid_target_evidence':mass>0}
