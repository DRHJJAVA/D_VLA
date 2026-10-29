"""LCJP003 v3: geometric observation validity only, never contact labels."""
import json
from pathlib import Path
import numpy as np
CONFIG=json.loads((Path(__file__).resolve().parents[1]/'p0/visibility_v3_config.json').read_text())
EPS=CONFIG['single_view_depth_error_m']
PAIR_EPS=CONFIG['pair_depth_error_m']
GRID_X,GRID_Y=np.meshgrid(np.arange(4,dtype=np.float64)-1.5,np.arange(4,dtype=np.float64)-1.5)
GX=GRID_X.ravel(); GY=GRID_Y.ravel()
def surface_query(depth,uv,z,excluded=None):
    """uv continuous depth-raster coordinates. Returns state and diagnostics."""
    depth=np.asarray(depth,dtype=np.float64);uv=np.asarray(uv,dtype=np.float64);z=np.asarray(z,dtype=np.float64)
    H,W=depth.shape;N=len(z)
    finite=np.isfinite(uv).all(1)&np.isfinite(z)&(z>0)
    clean=np.where(np.isfinite(uv),uv,0)
    x=np.floor(clean[:,0]).astype(np.int64);y=np.floor(clean[:,1]).astype(np.int64)
    inside=finite&(clean[:,0]>=0)&(clean[:,0]<=W-1)&(clean[:,1]>=0)&(clean[:,1]<=H-1)
    support=inside&(x>=1)&(x+2<W)&(y>=1)&(y+2<H)
    dx=np.tile(np.arange(-1,3),4);dy=np.repeat(np.arange(-1,3),4)
    xx=np.clip(x[:,None]+dx,0,W-1); yy=np.clip(y[:,None]+dy,0,H-1)
    values=depth[yy,xx]
    valid=np.isfinite(values).all(1)&(values>0).all(1)
    if excluded is not None:valid&=~np.asarray(excluded,dtype=bool)[yy,xx].any(1)
    support&=valid
    inv=1/np.where(np.isfinite(values)&(values>0),values,1.)
    a=inv.mean(1);b=(inv*GX).sum(1)/(GX@GX);c=(inv*GY).sum(1)/(GY@GY)
    prediction_inv=a[:,None]+b[:,None]*GX+c[:,None]*GY
    pred_support=1/np.maximum(prediction_inv,1e-12)
    fit=np.max(np.abs(pred_support-values),axis=1)
    uq=clean[:,0]-x-.5;vq=clean[:,1]-y-.5
    query_inv=a+b*uq+c*vq
    estimated=1/np.maximum(query_inv,1e-12)
    smooth=support&(fit<=EPS)&(query_inv>0)
    gap=z-estimated
    state=np.zeros(N,dtype=np.uint8);state[inside]=4
    state[smooth&(gap>PAIR_EPS)]=3
    state[smooth&(np.abs(gap)<=PAIR_EPS)]=2
    # Discontinuous patch is never interpolated into a pretend surface.
    # All measured local surfaces in front provides conservative occlusion evidence.
    state[support&~smooth&(np.max(values,axis=1)+PAIR_EPS<z)]=3
    return {'state':state,'surface_depth':np.where(smooth,estimated,0).astype(np.float32),
            'depth_gap':np.where(smooth,gap,0).astype(np.float32),
            'support_error':np.where(support,fit,0).astype(np.float32),
            'surface_supported':smooth}
def apply_visibility(result,depth,robot_mask,xyz_world):
    old=result['view_visible'].copy()
    N=len(xyz_world);states=np.zeros((N,2),np.uint8)
    surface=np.zeros((N,2),np.float32);gap=np.zeros((N,2),np.float32);fit=np.zeros((N,2),np.float32)
    for c in range(2):
        C=result['camera_to_world'][c]
        camera=(np.asarray(xyz_world,dtype=np.float64)-C[:3,3])@C[:3,:3]
        d=surface_query(depth[c],result['projected_depth_uv'][:,c],camera[:,2],robot_mask[c])
        states[:,c]=d['state'];surface[:,c]=d['surface_depth'];gap[:,c]=d['depth_gap'];fit[:,c]=d['support_error']
    valid=result['point_valid']
    for c in range(2):states[valid&(result['camera_id']==c),c]=1
    states[~valid]=0
    result['view_visible_legacy']=old
    result['view_state']=states
    result['view_surface_depth']=surface
    result['view_depth_gap']=gap
    result['view_support_error']=fit
    result['view_visible']=(states==1)|(states==2)
    result['relation'][:,-1]=result['view_visible'].all(1).astype(np.float32)
    return result
