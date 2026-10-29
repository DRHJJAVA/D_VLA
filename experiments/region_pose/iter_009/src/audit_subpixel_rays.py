from pathlib import Path
import json,hashlib,csv
import numpy as np,h5py
R=Path('/mnt/nvme1/drh/D_VLA');I=Path(__file__).resolve().parents[1];OLD=I.parent/'iter_003';SRC=I.parent/'iter_008/train'
def apply(p,T):return p@T[:3,:3].T+T[:3,3]
def back(uv,z,K,T):
 c=np.stack([(uv[:,0]-K[0,2])*z/K[0,0],(uv[:,1]-K[1,2])*z/K[1,1],z],1)
 return apply(c,T)
def stats(v):
 a=np.asarray(v,float)
 return {'n':len(a),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'max':float(a.max()),'mean':float(a.mean())} if len(a) else {'n':0}
def main():
 refs={x['id']:x for x in json.loads((OLD/'evidence/base/protocol.json').read_text())['episodes']}
 summary=json.loads((SRC/'summary.json').read_text());rows=[];events=[];inputs=[SRC/'summary.json',OLD/'evidence/base/protocol.json']
 for case in summary['cases']:
  ident=case['id'];q=case['query'];name=case['event']+'_query'+str(q);path=SRC/'correspondences'/(name+'.npz')
  if not path.exists():
   events.append({'id':ident,'event':case['event'],'available':False,'reason':case['reason']});continue
  inputs.append(path);mfile=OLD/'evidence/motion'/(ident+'.npz');inputs.append(mfile)
  with np.load(mfile) as m:H=m['hand_to_world']
  eventrows=[]
  with np.load(path) as z,h5py.File(refs[ident]['rgbd'],'r') as f:
   g=f['agentview']
   for fit in case['bridge_results']:
    t=fit['frame'];samples=[]
    for prefix,frame in [('query',q),('bridge',t)]:
     uv=z[prefix+'_uv_'+str(t)].astype(float);pix=np.rint(uv).astype(int)
     dep=g['depth_m'][frame];K=g['K'][frame];T=g['camera_to_world'][frame];d=dep[pix[:,1],pix[:,0]].astype(float)
     original=back(pix,d,K,T);floating=back(uv,d,K,T);saved=z[prefix+'_xyz_'+str(t)]
     err=np.linalg.norm(original-saved,axis=1)
     assert np.max(err,initial=0)<1e-6,(ident,frame,err.max())
     shift=np.linalg.norm(floating-original,axis=1)*1000
     span=np.array([np.ptp(dep[y-1:y+2,x-1:x+2])*1000 for x,y in pix])
     samples.append((original,floating,shift,span,err))
    frozen=fit.get('query_world_to_anchor_hand');rr0=rr1=None
    if frozen is not None:
     W=H[t]@np.array(frozen)
     rr0=np.linalg.norm(apply(samples[0][0],W)-samples[1][0],axis=1)*1000
     rr1=np.linalg.norm(apply(samples[0][1],W)-samples[1][1],axis=1)*1000
    for k in range(len(samples[0][0])):
     span=float(max(samples[0][3][k],samples[1][3][k]))
     row={'id':ident,'event':case['event'],'query':q,'held':t,'match_index':k,'original_inlier':bool(z['inlier_'+str(t)][k]),'query_ray_shift_mm':float(samples[0][2][k]),'held_ray_shift_mm':float(samples[1][2][k]),'pair_shift_bound_mm':float(samples[0][2][k]+samples[1][2][k]),'max_depth_span_mm':span,'depth_group':'smooth_le5mm' if span<=5 else ('middle' if span<=10 else 'jump_gt10mm'),'xyz_reconstruction_error_mm':float(max(samples[0][4][k],samples[1][4][k])*1000),'frozen_original_residual_mm':None if rr0 is None else float(rr0[k]),'frozen_continuous_ray_residual_mm':None if rr1 is None else float(rr1[k])}
     if rr0 is not None:assert abs(rr1[k]-rr0[k])<=row['pair_shift_bound_mm']+1e-6
     rows.append(row);eventrows.append(row)
  events.append({'id':ident,'event':case['event'],'available':True,'pair_observations':len(eventrows),'pair_bound_mm':stats([x['pair_shift_bound_mm'] for x in eventrows]),'ray_shift_mm':stats([x[k] for x in eventrows for k in ['query_ray_shift_mm','held_ray_shift_mm']]),'frozen_residual_before_mm':stats([x['frozen_original_residual_mm'] for x in eventrows if x['frozen_original_residual_mm'] is not None]),'frozen_residual_after_mm':stats([x['frozen_continuous_ray_residual_mm'] for x in eventrows if x['frozen_original_residual_mm'] is not None])})
 with (I/'measurements.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 groups={}
 for group in ['all','smooth_le5mm','middle','jump_gt10mm']:
  a=rows if group=='all' else [x for x in rows if x['depth_group']==group]
  groups[group]={'pair_bound_mm':stats([x['pair_shift_bound_mm'] for x in a]),'single_ray_shift_mm':stats([x[k] for x in a for k in ['query_ray_shift_mm','held_ray_shift_mm']]),'fraction_pair_bound_gt2mm':float(np.mean([x['pair_shift_bound_mm']>2 for x in a])),'fraction_pair_bound_gt3mm':float(np.mean([x['pair_shift_bound_mm']>3 for x in a])),'frozen_residual_before_mm':stats([x['frozen_original_residual_mm'] for x in a if x['frozen_original_residual_mm'] is not None]),'frozen_residual_after_mm':stats([x['frozen_continuous_ray_residual_mm'] for x in a if x['frozen_original_residual_mm'] is not None])}
 result={'protocol':'009_train_only_measurement','cases':events,'groups':groups,'max_reconstruction_error_mm':max(x['xyz_reconstruction_error_mm'] for x in rows),'event_equal_mean_ray_shift_mm':float(np.mean([x['ray_shift_mm']['mean'] for x in events if x['available'] and x['ray_shift_mm']['n']])),'new_targets':False,'training':False,'refitting':False,'validation_read':False}
 (I/'summary.json').write_text(json.dumps(result,indent=2))
 (I/'input_locked.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(inputs)},indent=2))
 # Large H5 provenance is recorded without copying or hashing unrelated frames.
 (I/'rgbd_provenance.json').write_text(json.dumps({x['id']:{'path':refs[x['id']]['rgbd'],'size':Path(refs[x['id']]['rgbd']).stat().st_size,'mtime_ns':Path(refs[x['id']]['rgbd']).stat().st_mtime_ns} for x in events if x['available']},indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ['cases','groups']}),flush=True)
 print(json.dumps(groups['all']),flush=True)
if __name__=='__main__':main()
