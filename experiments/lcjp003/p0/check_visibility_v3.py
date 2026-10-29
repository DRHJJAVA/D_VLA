import sys,json,hashlib
from pathlib import Path
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003');sys.path.insert(0,str(R/'src'))
from visibility import surface_query,CONFIG
H=W=128;fx=180.;fy=180.;cx=64.;cy=64.;off=np.array([.375,.875])
def rays(uv):return np.column_stack(((uv[:,0]+off[0]-cx)/fx,(uv[:,1]+off[1]-cy)/fy,np.ones(len(uv))))
def intersect(origin,direction,scene):
    n=len(direction);t=np.full(n,1.2);ids=np.zeros(n,np.int32)
    if scene=='tilted_plane':
        t=(1.+.15*origin[0]-.12*origin[1])/(direction[:,2]-.15*direction[:,0]+.12*direction[:,1]);ids[:]=1
    elif scene in ['step','thin_plate']:
        tf=.65*np.ones(n);x=origin[0]+direction[:,0]*tf
        keep=x<0 if scene=='step' else (x>-.008)&(x<.008)
        t[keep]=tf[keep];ids[keep]=1
    elif scene=='sphere':
        c=np.array([.07,0.,.75]);oc=origin-c;a=np.einsum('ij,ij->i',direction,direction);b=2*(direction@oc);q=oc@oc-.14**2
        disc=b*b-4*a*q;tt=(-b-np.sqrt(np.maximum(disc,0)))/(2*a)
        hit=(disc>=0)&(tt>0)&(tt<t);t[hit]=tt[hit];ids[hit]=1
    return t,ids
def render(origin,scene):
    yy,xx=np.mgrid[:H,:W];uv=np.stack([xx.ravel(),yy.ravel()],1)
    z,ids=intersect(origin,rays(uv),scene)
    return z.reshape(H,W).astype(np.float32),ids.reshape(H,W)
rows=[];all_counts=dict(accepted=0,false_accepted=0,visible_interior=0,accepted_visible_interior=0)
for scene in ['tilted_plane','step','thin_plate','sphere']:
    cams=[np.array([-.06,0.,0.]),np.array([.06,0.,0.])]
    maps=[render(c,scene) for c in cams]
    for source in [0,1]:
        target=1-source
        yy,xx=np.mgrid[2:H-2:2,2:W-2:2];uv=np.stack([xx.ravel(),yy.ravel()],1)
        z=maps[source][0][uv[:,1],uv[:,0]]
        world=cams[source]+rays(uv)*z[:,None];p=world-cams[target]
        dest=np.column_stack((fx*p[:,0]/p[:,2]+cx,fy*p[:,1]/p[:,2]+cy))-off
        actual,ids=intersect(cams[target],rays(dest),scene)
        in_image=(dest[:,0]>=0)&(dest[:,0]<=W-1)&(dest[:,1]>=0)&(dest[:,1]<=H-1)
        true_visible=in_image&(np.abs(actual-p[:,2])<2e-6)
        out=surface_query(maps[target][0],dest,p[:,2])
        accept=out['state']==2
        x=np.floor(dest[:,0]).astype(int);y=np.floor(dest[:,1]).astype(int)
        interior=true_visible&(x>=1)&(x+2<W)&(y>=1)&(y+2<H)
        for dx in range(-1,3):
            for dy in range(-1,3):
                interior&=maps[target][1][np.clip(y+dy,0,H-1),np.clip(x+dx,0,W-1)]==ids
        c={'accepted':int(accept.sum()),'false_accepted':int((accept&~true_visible).sum()),'visible_interior':int(interior.sum()),'accepted_visible_interior':int((accept&interior).sum())}
        foreground_interior=interior&(ids==1)
        foreground_counts={'foreground_visible_interior':int(foreground_interior.sum()),'foreground_accepted_interior':int((accept&foreground_interior).sum())}
        for k in c:all_counts[k]+=c[k]
        rows.append({'scene':scene,'source':source,'points':len(z),**c,'truth_visible':int(true_visible.sum()),**foreground_counts,'state_counts':{str(i):int((out['state']==i).sum()) for i in [0,2,3,4]},'false_accept_max_depth_gap':float(np.abs(actual-p[:,2])[accept&~true_visible].max(initial=0))})
fa=all_counts['false_accepted']/max(1,all_counts['accepted']);recall=all_counts['accepted_visible_interior']/max(1,all_counts['visible_interior'])
out={'status':'passed' if fa<=.001 and recall>=.3 else 'failed','reference':'independent analytic ray/surface intersections, not the production local plane or cycle threshold','config':CONFIG,'counts':all_counts,'false_accept_rate':fa,'visible_interior_recall':recall,'rows':rows,'visibility_source_sha256':hashlib.sha256((R/'src/visibility.py').read_bytes()).hexdigest()}
(R/'p0/visibility_v3_synthetic.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
