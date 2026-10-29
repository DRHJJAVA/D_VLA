import numpy as np
from scipy.spatial import cKDTree
from sampling import fps
def box_tri(size):
 v=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]],dtype=float)*size
 f=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
 return v[f]
def orient(tri):
 n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
 flip=(n*(tri.mean(1)-tri.mean((0,1)))).sum(1)<0
 tri=tri.copy();tri[flip]=tri[flip][:,[0,2,1]]
 return tri
def closest(x,tri):
 tri=orient(tri);a,b,c=tri[:,0],tri[:,1],tri[:,2]
 n=np.cross(b-a,c-a);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
 d=np.einsum('nti,ti->nt',x[:,None]-a[None],n)
 proj=x[:,None]-d[:,:,None]*n
 e0=b-a;e1=c-a;v=proj-a
 d00=(e0*e0).sum(1);d01=(e0*e1).sum(1);d11=(e1*e1).sum(1)
 d20=(v*e0).sum(-1);d21=(v*e1).sum(-1);den=np.maximum(d00*d11-d01*d01,1e-20)
 u=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;inside=(u>=0)&(w>=0)&(u+w<=1)
 candidates=[proj];cost=[np.where(inside,d*d,np.inf)]
 for v0,v1 in [(a,b),(b,c),(c,a)]:
  e=v1-v0;alpha=np.clip(((x[:,None]-v0)*e).sum(-1)/np.maximum((e*e).sum(-1),1e-20),0,1)
  y=v0+alpha[:,:,None]*e;candidates.append(y);cost.append(((y-x[:,None])**2).sum(-1))
 cost=np.stack(cost,-1);mi=cost.reshape(len(x),-1).argmin(1);face=mi//4;which=mi%4
 cand=np.stack(candidates,2);point=cand[np.arange(len(x)),face,which]
 distance=np.sqrt(cost.reshape(len(x),-1).min(1));within=(d<=1e-7).all(1)
 return distance*np.where(within,-1,1),point,n[face]
def finger_geometry(h,t,xyz):
 groups=h['finger_surface'];gp=h['robot/geom_position'];gr=h['robot/geom_rotation'];allnodes=[];allnorm=[];sdf=[];cp=[];norm=[];disp=[]
 for side in [1,2]:
  parts=[];partsnext=[]
  for name,g in groups.items():
   if 'finger'+str(side) not in name:continue
   if 'vertices' in g:local=g['vertices'][:][g['faces'][:]]
   else:local=box_tri(g['size'][:])
   local=orient(local);idx=int(g.attrs['robot_geom_index'])
   current=local@gr[t,idx].T+gp[t,idx];nxt=local@gr[min(t+1,len(gp)-1),idx].T+gp[min(t+1,len(gp)-1),idx]
   parts.append((current,idx,local));partsnext.append(nxt)
  ds=[];cs=[];ns=[];vs=[]
  for (tri,idx,local),nxt in zip(parts,partsnext):
   d,c,n=closest(xyz,tri);oldR=gr[t,idx];newR=gr[min(t+1,len(gp)-1),idx]
   cl=(c-gp[t,idx])@oldR;cnext=cl@newR.T+gp[min(t+1,len(gp)-1),idx]
   ds.append(d);cs.append(c);ns.append(n);vs.append(cnext-c)
  ds=np.stack(ds,1);choice=ds.argmin(1);ii=np.arange(len(xyz))
  sdf.append(ds[ii,choice]);cp.append(np.stack(cs,1)[ii,choice]);norm.append(np.stack(ns,1)[ii,choice]);disp.append(np.stack(vs,1)[ii,choice])
  # True robot mesh samples, deterministic area-proportional surface nodes.
  triangles=np.concatenate([p[0] for p in parts]);triangles=orient(triangles)
  n=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);area=np.linalg.norm(n,axis=1);n/=np.maximum(area[:,None],1e-12)
  cdf=np.cumsum(area)/area.sum();ids=np.searchsorted(cdf,(np.arange(1024)+.5)/1024)
  u=np.sqrt((np.arange(1024)*.61803398875)%1);v=(np.arange(1024)*.41421356237)%1
  pts=(1-u[:,None])*triangles[ids,0]+(u*(1-v))[:,None]*triangles[ids,1]+(u*v)[:,None]*triangles[ids,2]
  allowed=np.ones(len(pts),bool)
  for tri,_,_ in parts:allowed&=closest(pts,tri)[0]>-1e-6
  pts=pts[allowed];nn=n[ids][allowed];assert len(pts)>=64
  sel=fps(pts,64);allnodes.append(pts[sel]);allnorm.append(nn[sel])
 return {'finger_nodes':np.concatenate(allnodes),'finger_normals':np.concatenate(allnorm),'finger_side':np.repeat(np.eye(2),64,axis=0),'sdf':np.stack(sdf,1),'closest':np.stack(cp,1),'surface_normal':np.stack(norm,1),'finger_displacement':np.stack(disp,1)}
def graph(x):
 result={};levels=[x]
 for i,n in enumerate([512,128,32],1):
  old=levels[-1];sel=fps(old.astype(float),min(n,len(old)));new=old[sel]
  d,ids=cKDTree(old).query(new,k=min(32,len(old)));ids=np.atleast_2d(ids)
  result['s'+str(i)]=sel;result['n'+str(i)]=ids;levels.append(new)
 for hi,lo,name in [(3,2,'32'),(2,1,'21'),(1,0,'10')]:
  d,idx=cKDTree(levels[hi]).query(levels[lo],k=min(3,len(levels[hi])))
  if d.ndim==1:d=d[:,None];idx=idx[:,None]
  w=1/np.maximum(d,1e-8);w/=w.sum(1,keepdims=True);result['i'+name]=idx;result['w'+name]=w
 return result
def rigid(A,B):
 if len(A)<3:return np.eye(3),np.zeros(3),np.inf
 a=A.mean(0);b=B.mean(0);u,s,v=np.linalg.svd((A-a).T@(B-b))
 if s[1]<1e-10:return np.eye(3),np.zeros(3),np.inf
 R=v.T@u.T
 if np.linalg.det(R)<0:v[-1]*=-1;R=v.T@u.T
 tr=b-a@R.T;res=np.linalg.norm(A@R.T+tr-B,axis=1)
 return R,tr,float(np.median(res))

def separation_lower_bound(h,t,xyz):
 distances=[]
 for name,g in h['finger_surface'].items():
  idx=int(g.attrs['robot_geom_index'])
  if 'vertices' in g:v=g['vertices'][:]
  else:v=box_tri(g['size'][:]).reshape(-1,3)
  v=v@h['robot/geom_rotation'][t,idx].T+h['robot/geom_position'][t,idx]
  low=v.min(0);high=v.max(0);delta=np.maximum(np.maximum(low-xyz,xyz-high),0)
  distances.append(np.linalg.norm(delta,axis=1))
 return np.min(distances,axis=0)
