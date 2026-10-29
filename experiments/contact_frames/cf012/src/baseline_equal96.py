from common12 import *
import numpy as np,cv2,h5py
from scipy.ndimage import maximum_filter,minimum_filter
cv2.setNumThreads(2)
def main():
 rows=[]
 for ep in CFG['episodes']:
  with np.load(PREV/'data'/f'{ep}.npz') as z:
   frames=z['frames'];depth=z['depth'][:,0];center=z['center'];grid=z['grid'];eligible=z['eligible'];target=z['target'];mask=z['mask']
  oldpath=ROOT/'experiments/contact_frames/cf005/data'/ep
  c=json.loads((oldpath/'config.json').read_text())
  with h5py.File(ROOT/c['rgbd'],'r') as h:original_width=h['agentview/depth_m'].shape[-1]
  T,N=grid.shape[:2];velocity=np.zeros((T,N,3),np.float32);valid=np.zeros((T,N),bool)
  a=CFG['flow_parameters'];args=(None,a['pyr_scale'],a['levels'],a['winsize'],a['iterations'],a['poly_n'],a['poly_sigma'],a['flags'])
  gray=np.stack([cv2.cvtColor(np.uint8(np.clip(x[:3].transpose(1,2,0),0,1)*255),cv2.COLOR_RGB2GRAY) for x in frames])
  world=frames[:,3:6].transpose(0,2,3,1)*.1+center
  ok=frames[:,6]>.5
  for t in range(T):
   ok[t]&=(maximum_filter(depth[t],3)-minimum_filter(depth[t],3))<=.01
  for t in eligible:
   # Both complete dense fields precede evaluation-point sampling.
   b=cv2.calcOpticalFlowFarneback(gray[t],gray[t-1],*args);f=cv2.calcOpticalFlowFarneback(gray[t-1],gray[t],*args)
   uv=(grid[t]+1)*48-.5
   flow=cv2.remap(b,uv[:,0][None],uv[:,1][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[0]
   past=uv+flow
   rev=cv2.remap(f,past[:,0][None],past[:,1][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[0]
   err=np.linalg.norm(flow+rev,axis=-1)*original_width/96
   ix=np.rint(uv).astype(np.int64);ip=np.rint(past).astype(np.int64)
   inside=(uv>=0).all(-1)&(uv<=95).all(-1)&(past>=0).all(-1)&(past<=95).all(-1)
   ix=np.clip(ix,0,95);ip=np.clip(ip,0,95)
   velocity[t]=world[t,ix[:,1],ix[:,0]]-world[t-1,ip[:,1],ip[:,0]]
   valid[t]=inside&ok[t,ix[:,1],ix[:,0]]&ok[t-1,ip[:,1],ip[:,0]]&(err<=1.5)
  out=BASE/'equal96_baseline'/f'{ep}.npz';out.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(out,velocity=velocity,valid=valid,eligible=eligible)
  for seed in CFG['seeds']:
   for arm in CFG['arms']:
    with np.load(PREV/'runs'/ep/f'seed_{seed}'/arm/'predictions.npz') as z:
     ts=z['frames'];pred=z['mean'];truth=z['target'];vm=z['mask']
    for j,h in enumerate(CFG['horizons']):
     good=vm[:,j]&valid[ts];n=int(good.sum());total=int(vm[:,j].sum())
     rows.append(dict(episode=ep,seed=seed,arm=arm,horizon=h,count=n,total=total,valid_fraction=n/total,network_mm=float(np.linalg.norm(pred[:,j]-truth[:,j],axis=-1)[good].mean()*1000) if n else None,velocity_mm=float(np.linalg.norm(h*velocity[ts]-truth[:,j],axis=-1)[good].mean()*1000) if n else None,zero_mm=float(np.linalg.norm(truth[:,j],axis=-1)[good].mean()*1000) if n else None))
 save(BASE/'equal96_comparison.json',dict(time=now(),rows=rows,scope='posthoc matched-input baseline, no retraining or labels'))
 verify(json.loads((PREV/'completed_locked.json').read_text()));verify(json.loads((BASE/'completed_locked.json').read_text()))
 lines=['\n\n### CF012 同分辨率基线结果（'+now()+'）\n','|示范|跨度|保留比例|96恒速/mm|视觉历史/mm|机器人条件/mm|\n|---|---:|---:|---:|---:|---:|']
 for ep in CFG['episodes']:
  for h in CFG['horizons']:
   rr=[x for x in rows if x['episode']==ep and x['horizon']==h]
   f=lambda k:float(np.mean([x[k] for x in rr]))
   e0=float(np.mean([x['network_mm'] for x in rr if x['arm']=='object_history']))
   ea=float(np.mean([x['network_mm'] for x in rr if x['arm']=='robot_conditioned']))
   lines.append('|'+ep+'|'+str(h)+'|'+format(100*f('valid_fraction'),'.1f')+'%|'+format(f('velocity_mm'),'.3f')+'|'+format(e0,'.3f')+'|'+format(ea,'.3f')+'|')
 lines.append('\n各行仅比较同一交集点；96输入的有效率不同于原分辨率，不跨表直接比较均值。这是事后固定追加基线，不能宣称最初已预注册；未用于修改本轮模型或校准参数。\n')
 with (ROOT/'docs/小型验证方案_夹持运动证据与区域表示_v1.md').open('a') as f:f.write('\n'.join(lines))
 save(BASE/'equal96_locked.json',lock([BASE/'equal96_design.json',BASE/'src/baseline_equal96.py',BASE/'equal96_comparison.json',*(BASE/'equal96_baseline').glob('*.npz')]))
 print(json.dumps({'complete':True,'rows':len(rows)}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  (BASE/'equal96_failure_traceback.log').write_text(traceback.format_exc())
  save(BASE/'equal96_failure.json',dict(time=now(),type=type(e).__name__,frames=[dict(file=x.filename,line=x.lineno) for x in traceback.extract_tb(e.__traceback__)]))
  raise SystemExit(1)
