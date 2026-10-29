
import pathlib,sys,json,torch,math
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003')
sys.path.insert(0,str(ROOT/'src'))
from drifting import conditional_drifting_loss
torch.manual_seed(23)
x=torch.randn(2,8,16,7,requires_grad=True)
y=torch.randn(2,16,7,requires_grad=True)
mask=torch.ones(2,16,dtype=torch.bool);mask[:,14:]=False
loss,diags=conditional_drifting_loss(x,y,mask)
gx,gy=torch.autograd.grad(loss,(x,y),allow_unused=True)
assert torch.isfinite(loss) and torch.isfinite(gx).all()
assert gy is None
assert gx[:,:,14:].abs().max()==0
xp=x.detach().clone();xp[:,:,14:]=float('nan');xp.requires_grad_()
yp=y.detach().clone();yp[:,14:]=float('nan')
lp,_=conditional_drifting_loss(xp,yp,mask)
assert torch.allclose(loss,lp,atol=1e-6)
x2=x.detach().clone().requires_grad_();y2=y.detach().clone();y2[1]+=100
l2,_=conditional_drifting_loss(x2,y2,mask)
g2=torch.autograd.grad(l2,x2)[0]
assert torch.equal(gx[0],g2[0])
# For fixed drift targets, dL/dx = -2 V / (S * total_valid_per_condition * B).
# Independently compute one temperature field using explicit pair loops.
p=x.detach()[0,:,:14].reshape(8,-1);t=y.detach()[0,:14].reshape(1,-1);D=p.shape[-1]
cand=torch.cat([t,p]);raw=torch.stack([torch.stack([torch.linalg.vector_norm(pi-cj) for cj in cand]) for pi in p])
S=(raw.mean()/math.sqrt(D)).clamp_min(1e-6)
pn=p/S;tn=t/S;cn=torch.cat([tn,pn])
distance=torch.stack([torch.stack([torch.linalg.vector_norm(pi-cj) for cj in cn]) for pi in pn])
for i in range(8): distance[i,i+1]+=1e6
V=torch.zeros_like(p)
for tau in (.02,.05,.2):
 z=-distance/(tau*math.sqrt(D))
 a=(torch.softmax(z,1)*torch.softmax(z,0)).sqrt()
 v=torch.stack([sum(a[i,0]*a[i,j+1]*(tn[0]-pn[j]) for j in range(8)) for i in range(8)])
 V+=v/v.square().mean().sqrt().clamp_min(1e-6)
expected=-2*V/(S*8*D*2)
observed=gx[0,:,:14].reshape(8,-1)
maxdiff=float((expected-observed).abs().max())
assert torch.allclose(expected,observed,atol=1e-5,rtol=1e-4)
out={'status':'passed','scope':'synthetic_mathematical_unit_only_not_real_P0','finite_loss_gradient':True,'positive_target_stop_gradient':True,'invalid_action_padding_invariant':True,'cross_condition_isolated':True,'explicit_pair_reference_max_gradient_difference':maxdiff,'loss':float(loss),'gradient_norm':float(gx.norm()),'temperatures':[.02,.05,.2],'G':8,'paper':'https://arxiv.org/html/2602.04770v2'}
(ROOT/'p0/drifting_unit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
