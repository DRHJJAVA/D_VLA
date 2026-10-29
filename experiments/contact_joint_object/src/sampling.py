import numpy as np
from numba import njit
@njit(cache=True)
def fps(x,n):
 N=len(x)
 if N<=n:return np.arange(N)
 d=np.full(N,np.inf);idx=np.empty(n,np.int64);mean=np.zeros(3)
 for j in range(N):
  for a in range(3):mean[a]+=x[j,a]/N
 far=-1.;current=0
 for j in range(N):
  a=x[j,0]-mean[0];b=x[j,1]-mean[1];c=x[j,2]-mean[2];v=a*a+b*b+c*c
  if v>far:far=v;current=j
 for k in range(n):
  idx[k]=current;far=-1.;nextid=0
  for j in range(N):
   a=x[j,0]-x[current,0];b=x[j,1]-x[current,1];c=x[j,2]-x[current,2];v=a*a+b*b+c*c
   if v<d[j]:d[j]=v
   if d[j]>far:far=d[j];nextid=j
  current=nextid
 return idx
