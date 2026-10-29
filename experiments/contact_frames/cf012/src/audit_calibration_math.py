from common12 import *
import numpy as np
from scipy.optimize import minimize_scalar
rng=np.random.default_rng(20261024)
sizes=[100,1000,10000];scales=[.5,2,5];events=[]
for n,k in zip(sizes,scales):
 s=rng.uniform(.2,2,n)
 e=rng.normal(size=(n,3))*s[:,None]*k
 events.append((e,s))
q=[np.mean(np.sum(e**2,axis=-1)/(3*s**2)) for e,s in events]
alpha=np.sqrt(np.mean(q))
def objective(logalpha):
 a=np.exp(logalpha)
 return np.mean([np.mean(.5*np.sum((e/(a*s[:,None]))**2,axis=-1)+3*np.log(a*s)) for e,s in events])
numeric=minimize_scalar(objective,bounds=(-5,5),method='bounded',options={'xatol':1e-12})
assert abs(np.exp(numeric.x)-alpha)<1e-6
assert objective(np.log(alpha))<objective(0)
pooled=np.sqrt(sum(n*x for n,x in zip(sizes,q))/sum(sizes))
assert abs(pooled-alpha)>.5
save(BASE/'audits/calibration_math.json',dict(time=now(),event_equal_alpha=float(alpha),numeric_nll_optimum=float(np.exp(numeric.x)),pooled_alpha=float(pooled),unequal_event_sizes=sizes,nll_raw=float(objective(0)),nll_calibrated=float(objective(np.log(alpha))),result='passed'))
print('calibration_math_passed')
