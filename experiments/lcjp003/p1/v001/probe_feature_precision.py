from common import *
from features import FrozenSmolFeatures
cpu_setup()
ds=Dataset();e=CFG['train'][0];t=67
f=ds.frame(e,t)
enc=FrozenSmolFeatures(str(R/'weights/smolvlm2'),'cuda')
def val():
 return enc.point_features(f['rgb'][None],f['projected_uv'][None],f['view_visible'][None])[0]
x=val()
with np.load(P/'frames'/e/'67.npz') as z:cached=z['semantic']
with np.load(R/'p0/features'/(e+'_frame0067.npz')) as z:old=z['semantic'][-1]
torch.backends.cudnn.allow_tf32=True
y=val()
out={'cached_vs_same_frame_max':float(np.max(np.abs(x-cached))),'p0_vs_ieee_max':float(np.max(np.abs(x-old))),'p0_vs_cudnn_tf32_max':float(np.max(np.abs(y-old))),'valid_points':int(f['point_valid'].sum())}
write('frozen_precision_probe.json',out);print(json.dumps(out))
