import numpy as np,cv2
def rigid(a,b):
 ac=a.mean(0);bc=b.mean(0);u,s,v=np.linalg.svd((a-ac).T@(b-bc));rot=v.T@u.T
 if np.linalg.det(rot)<0:v[-1]*=-1;rot=v.T@u.T
 T=np.eye(4);T[:3,:3]=rot;T[:3,3]=bc-rot@ac;return T

def apply(p,T):return p@T[:3,:3].T+T[:3,3]

def estimate(a,b):
 if len(a)<8:return None,np.zeros(len(a),bool),{'valid':False,'reason':'fewer_than_8_endpoint_matches','matches':len(a)}
 rng=np.random.default_rng(2026);best=np.zeros(len(a),bool)
 for _ in range(500):
  ix=rng.choice(len(a),3,replace=False)
  if np.linalg.norm(np.cross(a[ix[1]]-a[ix[0]],a[ix[2]]-a[ix[0]]))<1e-6:continue
  T=rigid(a[ix],b[ix]);inlier=np.linalg.norm(apply(a,T)-b,axis=1)<=.003
  if inlier.sum()>best.sum():best=inlier
 if best.sum()<8:return None,best,{'valid':False,'reason':'fewer_than_8_rigid_inliers','matches':len(a),'inliers':int(best.sum())}
 T=rigid(a[best],b[best]);err=np.linalg.norm(apply(a,T)-b,axis=1);inlier=err<=.003
 if inlier.sum()>=8:T=rigid(a[inlier],b[inlier]);err=np.linalg.norm(apply(a,T)-b,axis=1)
 scale=np.linalg.svd(a[inlier]-a[inlier].mean(0),compute_uv=False)/np.sqrt(max(1,inlier.sum()))
 valid=bool(inlier.sum()>=8 and inlier.mean()>=.6 and len(scale)>1 and scale[1]>=.003 and np.median(err[inlier])<=.002)
 return T,inlier,dict(valid=valid,matches=len(a),inliers=int(inlier.sum()),ratio=float(inlier.mean()),second_axis_scale_mm=float(scale[1]*1000),median_inlier_mm=float(np.median(err[inlier])*1000))

def patchcorr(g0,g1,p0,p1):
 a=cv2.getRectSubPix(g0,(7,7),tuple(map(float,p0))).astype(float);b=cv2.getRectSubPix(g1,(7,7),tuple(map(float,p1))).astype(float)
 if min(a.std(),b.std())<3:return -1.
 a-=a.mean();b-=b.mean();return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
