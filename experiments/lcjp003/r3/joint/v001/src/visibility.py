import numpy as np

def visible_reference(xyz,cameras,depth_tolerance=.006):
    """Current-observation geometry only. No task labels, teacher scores or future data."""
    count=len(xyz); visible=np.zeros((count,2),bool); weights=np.zeros((count,2),np.float32)
    uv_all=np.full((count,2,2),-1,np.int64)
    for v,c in enumerate(cameras):
        ctw=c['camera_to_world']; pc=(xyz-ctw[:3,3])@ctw[:3,:3]
        pr=pc@c['K'].T
        finite=np.isfinite(pr).all(1)&(pc[:,2]>1e-6)
        uv=np.rint(pr[:,:2]/np.maximum(pr[:,2:3],1e-6)).astype(np.int64)
        h,w=c['depth'].shape
        inside=finite&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        ids=np.flatnonzero(inside)
        if not len(ids):continue
        px=uv[ids,0];py=uv[ids,1];depth=c['depth'][py,px]
        delta=np.abs(depth-pc[ids,2])
        good=np.isfinite(depth)&(depth>0)&(delta<=depth_tolerance)&(~c['robot_mask'][py,px].astype(bool))
        ids=ids[good];uv_all[ids,v]=uv[ids];visible[ids,v]=True
        # Measurement agreement weights, normalized only among currently visible views.
        weights[ids,v]=np.exp(-.5*(delta[good]/(depth_tolerance/2))**2)
    total=weights.sum(1,keepdims=True)
    weights=np.divide(weights,total,out=np.zeros_like(weights),where=total>0)
    return {'visible':visible,'weights':weights,'uv':uv_all,'current_unknown':~visible.any(1)}

def fuse_visible_features(features,visibility):
    assert features.shape[:2]==visibility['weights'].shape
    safe=np.where(visibility['visible'][...,None],features,0)
    return (safe*visibility['weights'][...,None]).sum(1)
