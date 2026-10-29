from pathlib import Path
import json,datetime,hashlib,sys
import numpy as np,h5py
from PIL import Image
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/contact_frames/reviews/cf013';C=R/'experiments/contact_frames/cf013';O=R/'experiments/contact_frames/cf011'
CFG=json.loads((B/'config.json').read_text());OUT=R/'docs/图示/cf013_motion';OUT.mkdir(parents=True,exist_ok=False)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,separators=(',',':'),allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for s in iter(lambda:f.read(8*1024*1024),b''):h.update(s)
 return h.hexdigest()
def project(x,K,C):
 q=np.linalg.solve(C[:3,:3],(x-C[:3,3]).T).T
 uv=q@K.T;uv=uv[:,:2]/uv[:,2:]
 valid=np.isfinite(uv).all(1)&(q[:,2]>0)
 return [[round(float(a),3),round(float(b),3)] if v else None for (a,b),v in zip(uv,valid)]
def main():
 inputs=[C/'completed_locked.json',C/'comparison.json',B/'config.json']
 for ep in CFG['episodes']:
  cp=R/'experiments/contact_frames/cf005/data'/ep/'config.json';cc=json.loads(cp.read_text())
  inputs += [cp,R/cc['rgbd'],O/'data'/f'{ep}.npz']
  for seed in CFG['seeds']:
   inputs += [O/'runs'/ep/f'seed_{seed}/robot_conditioned/predictions.npz']
   inputs += [C/'runs'/ep/f'seed_{seed}'/a/'predictions.npz' for a in ['motion_absolute','motion_residual']]
 lock={str(p):sha(p) for p in inputs};save(B/'input_locked.json',lock)
 index={'created':now(),'episodes':[],'aggregate':json.loads((C/'comparison.json').read_text())['aggregate'],'means':json.loads((C/'comparison.json').read_text())['means']}
 audit=[];total_images=0;max_projection=0.
 for ep in CFG['episodes']:
  target=OUT/ep;target.mkdir()
  cc=json.loads((R/'experiments/contact_frames/cf005/data'/ep/'config.json').read_text());base=np.load(O/'data'/f'{ep}.npz')
  with h5py.File(R/cc['rgbd'],'r') as f:
   g=f['agentview'];T,H,W=g['rgb'].shape[:3]
   K=g['K'][:];cam=g['camera_to_world'][:]
   for t in range(T):Image.fromarray(g['rgb'][t]).save(target/f'{t:04d}.jpg',quality=90)
   total_images+=T
  item=dict(id=ep,frames=T,width=W,height=H,seeds=CFG['seeds'],horizons=CFG['horizons'],views={})
  for seed in CFG['seeds']:
   a=np.load(C/'runs'/ep/f'seed_{seed}/motion_absolute/predictions.npz')
   z=np.load(C/'runs'/ep/f'seed_{seed}/motion_residual/predictions.npz')
   old=np.load(O/'runs'/ep/f'seed_{seed}/robot_conditioned/predictions.npz')
   assert np.array_equal(z['frames'],old['frames']) and np.array_equal(z['mask'],old['mask'])
   assert np.allclose(z['target'],old['target'],atol=1e-7,rtol=0)
   for hi,h in enumerate(CFG['horizons']):
    frames={}
    for i,t0 in enumerate(z['frames']):
     t=int(t0);mask=z['mask'][i,hi]&z['motion_valid'][i];ids=np.flatnonzero(mask)
     means={'old':old['mean'][i,hi],'velocity':z['velocity'][i]*h,'residual':z['mean'][i,hi],'absolute':a['mean'][i,hi]}
     truth=z['target'][i,hi];orig=base['xyz'][t]
     uv=project((orig+truth)[ids],K[t+h],cam[t+h])
     actual=base['pixels'][t+h,ids]
     if len(ids):
      pd=np.array([p for p in uv]);delta=float(np.max(np.abs(pd-actual)));max_projection=max(max_projection,delta)
      assert delta<.02,('projection_not_actual_pixel',ep,seed,t,h,delta)
     metrics={};projections={}
     for name,pred in means.items():
      error=np.linalg.norm(pred-truth,axis=1)
      metrics[name]={'error_mm':float(error[mask].mean()*1000) if len(ids) else None,'median_mm':float(np.median(error[mask])*1000) if len(ids) else None}
      projections[name]=project((orig+pred)[ids],K[t+h],cam[t+h])
     frames[str(t)]={'future':t+h,'count':len(ids),'total_target_count':int(z['mask'][i,hi].sum()),'indices':ids.tolist(),'current_uv':base['pixels'][t,ids].tolist(),'target_uv':uv,'predicted_uv':projections,'metrics':metrics}
    item['views'][str(seed)+'_'+str(h)]=frames
  save(target/'data.json',item)
  index['episodes'].append({k:item[k] for k in ['id','frames','width','height']})
  audit.append({'episode':ep,'images':T,'seeds':3,'horizons':3,'evaluated_frames':len(z['frames']),'actual_saved_prediction_runs':9})
  print(json.dumps(audit[-1]),flush=True)
 save(OUT/'index.json',index)
 page=R/'docs'/CFG['page'];page.write_text((B/'src/page.html').read_text(),encoding='utf8')
 for p,h in lock.items():assert sha(Path(p))==h,('input_changed',p)
 save(B/'generation_audit.json',dict(time=now(),episodes=audit,images=total_images,max_projected_target_pixel_difference=max_projection,source='saved frozen actual predictions',retraining=False,new_inference=False,manual_ranges_used=False,common_point_masks=True))
 outputs=[page,*[p for p in OUT.rglob('*') if p.is_file()]]
 save(B/'display_locked.json',{str(p):sha(p) for p in outputs})
 print(json.dumps({'page':str(page),'assets':len(outputs),'time':now()}),flush=True)
if __name__=='__main__':main()
