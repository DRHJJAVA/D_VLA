from pathlib import Path
import sys,os,json,time,copy,hashlib,random,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,torch,h5py
from data import Dataset,to_device,B,P
from network import JointModel,joint_loss
from online_targets import refresh_episode,target_for
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def state(stage,**kw):save(B/'training_state.json',dict(stage=stage,time=now(),**kw))
def calibrate(ds):
 vals=[];sources=[];missing=[]
 for r in ds.rows:
  errors=[]
  for t in np.linspace(1,r['frames']-2,4).astype(int):
   for oid in ds.frame_objects(r,int(t)):
    b=ds.get(r,int(t),oid,point_limit=128)
    if np.isfinite(b['history_fit_error']):errors.append(b['history_fit_error'])
  if errors:vals.extend(errors);sources.append({'id':r['id'],'count':len(errors),'median':float(np.median(errors))})
  else:missing.append(r['id'])
 if not vals:raise RuntimeError('No measured train motion residuals available for observation scale')
 sigma=max(float(np.median(vals)/.67448975),.0005)
 if sigma>.02:raise RuntimeError('Training observation scale exceeds 20 mm; inspect observations before training')
 save(B/'manifests/train_measurement_scale.json',{'time':now(),'sigma_m':sigma,'estimator':'median training rigid-fit residual / 0.67448975; 0.5 mm numerical measurement floor','train_only':True,'sources':sources,'missing':missing,'contact_labels_used':False})
 ds.sigma=sigma
 return sigma
def evaluate(m,ds,epoch):
 m.eval();rows=[]
 for r in ds.rows:
  values=[];objects=0;with_motion=0
  # All video frames, current observations only, sampled real surfaces.
  for t in range(r['frames']):
   for oid in ds.frame_objects(r,t):
    torch.manual_seed(2026+t)
    raw=ds.get(r,t,oid,ci=t%2,point_limit=128);b=to_device(raw)
    with torch.no_grad():_,v,o=joint_loss(m,b)
    value=v['motion']+v['depth']+v['reprojection']+.1*v['masked_feature']
    if not np.isfinite(value):raise FloatingPointError('nonfinite development objective')
    values.append(value);objects+=1;with_motion+=int(len(raw['motion_source'])>0)
  rows.append({'id':r['id'],'observation_objective':float(np.mean(values)) if values else None,'object_frames':objects,'motion_object_frames':with_motion})
  save(B/'evaluation/v001/dev_progress.json',{'epoch':epoch,'completed':len(rows),'total':len(ds.rows)})
 valid=[v['observation_objective'] for v in rows if v['observation_objective'] is not None]
 if len(valid)!=len(ds.rows):raise RuntimeError('Development episode has no valid current surface; do not silently omit')
 result={'epoch':epoch,'time':now(),'criterion':float(np.mean(valid)),'episodes':rows,'human_ranges_used':False,'contact_truth_metrics':'unavailable'}
 save(B/'evaluation/v001'/('dev_epoch_%03d.json'%epoch),result);return result
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35)
 cfg=json.loads((B/'config.json').read_text());train=Dataset('train');dev=Dataset('dev')
 assert len(train.rows)==400 and len(dev.rows)==50
 state('waiting_for_full_observations')
 while not (B/'manifests/observation_complete.json').exists():
  if (B/'observation_failure.json').exists():raise RuntimeError('Observation pipeline failed; training not started')
  time.sleep(10)
 assert json.loads((B/'manifests/network_smoke.json').read_text())['ok']
 assert json.loads((B/'manifests/online_smoke.json').read_text())['ok']
 state('locking_all_training_and_development_inputs')
 inputs={}
 for r in train.rows+dev.rows:
  for path in train.paths(r):
   assert path.exists()
   inputs[str(path.relative_to(P))]={'sha256':sha(path),'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
 save(B/'runs/v001/input_locked.json',{'time':now(),'files':inputs,'reserved_content_read':False})
 state('calibrating_train_observation_scale')
 sigma=calibrate(train);dev.sigma=sigma
 save(B/'runs/v001/measurement_scale_locked.json',{'sha256':sha(B/'manifests/train_measurement_scale.json'),'sigma_m':sigma})
 # No old weights and no continuation from the engineering smoke.
 torch.manual_seed(2026);np.random.seed(2026);random.seed(2026)
 m=JointModel().cuda();ema=copy.deepcopy(m).eval()
 for p in ema.parameters():p.requires_grad_(False)
 opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
 run=B/'runs/v001';initial=run/'initial.pt'
 assert not initial.exists(),'Formal run already initialized; explicit resume required'
 torch.save(m.state_dict(),initial)
 source_hash={str(p.relative_to(P)):sha(p) for p in (B/'src').glob('*.py')}
 save(run/'source_locked.json',{'time':now(),'files':source_hash,'config':sha(B/'config.json'),'initial_sha256':sha(initial)})
 sampler=[];coverage={r['id']:set() for r in train.rows}
 for epoch in range(1,cfg['epochs']+1):
  rng=np.random.default_rng(2026+epoch)
  for quarter in range(4):
   for ri in rng.permutation(len(train.rows)):
    r=train.rows[int(ri)];fraction=(quarter+(epoch-1)/max(1,cfg['epochs']-1))/4;t=1+min(r['frames']-3,int(fraction*(r['frames']-2)))
    sampler.append({'epoch':epoch,'quarter':quarter,'row_index':int(ri),'frame':t})
    coverage[r['id']].update(range(max(0,t-15),min(r['frames'],t+2)))
 assert all(len(coverage[r['id']])==r['frames'] for r in train.rows),'Window schedule misses original timestamps'
 save(run/'schedule.json',{'windows':sampler,'coverage':{k:len(v) for k,v in coverage.items()},'windows_are_not_independent_events':True})
 history=[];best=float('inf');updates=0
 for epoch in range(1,cfg['epochs']+1):
  for rel,info in inputs.items():
   st=(P/rel).stat();assert st.st_size==info['size'] and st.st_mtime_ns==info['mtime_ns'],'input changed since hash lock'
  # Full video online refresh from current EMA, never fixed offline contact labels.
  state('online_target_refresh',epoch=epoch,completed=0,total=400)
  summary=[]
  if epoch>1:
   for r in train.rows:
    summary.append(refresh_episode(ema,train,r,epoch,point_limit=128))
    state('online_target_refresh',epoch=epoch,completed=len(summary),total=400)
  save(run/('online_epoch_%03d.json'%epoch),{'time':now(),'episodes':summary,'target_generation':'EMA learned relation plus observation uncertainty','teacher_weights_sha256':sha(initial) if epoch==1 else sha(run/'last.pt')})
  m.train();opt.zero_grad(set_to_none=True);metrics=[];skipped=[];accum=0;last_batch_ids=[]
  plan=[x for x in sampler if x['epoch']==epoch]
  state('training',epoch=epoch,window=0,total=len(plan))
  for wi,item in enumerate(plan):
   r=train.rows[item['row_index']];t=item['frame'];ids=train.frame_objects(r,t)
   if not ids:skipped.append({'id':r['id'],'frame':t,'reason':'no observed current object proposal'});continue
   oid=ids[(epoch*4+item['quarter'])%len(ids)]
   raw=train.get(r,t,oid,ci=(epoch+item['quarter'])%2);b=to_device(raw);online=target_for(raw,epoch)
   loss,v,o=joint_loss(m,b,online)
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite training objective')
   (loss/cfg['batch_episodes']).backward();accum+=1;last_batch_ids.append(r['id']);metrics.append(v)
   if accum==cfg['batch_episodes'] or wi==len(plan)-1:
    assert len(set(last_batch_ids))==len(last_batch_ids),'effective batch repeats an episode'
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True);updates+=1
    with torch.no_grad():
     for pe,pm in zip(ema.parameters(),m.parameters()):pe.mul_(cfg['ema']).add_(pm,alpha=1-cfg['ema'])
    accum=0;last_batch_ids=[]
   if wi%20==0:state('training',epoch=epoch,window=wi+1,total=len(plan),optimizer_updates=updates)
  if accum:
   torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True);updates+=1
   with torch.no_grad():
    for pe,pm in zip(ema.parameters(),m.parameters()):pe.mul_(cfg['ema']).add_(pm,alpha=1-cfg['ema'])
  state('development_evaluation',epoch=epoch)
  ev=evaluate(m,dev,epoch)
  averages={k:float(np.mean([v[k] for v in metrics])) for k in metrics[0]} if metrics else {}
  record={'epoch':epoch,'time':now(),'optimizer_updates':updates,'training_windows':len(metrics),'skipped_observations':skipped,'train':averages,'development_criterion':ev['criterion']}
  history.append(record);save(run/'history.json',history)
  checkpoint={'model':m.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'optimizer_updates':updates,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'schedule_sha256':sha(run/'schedule.json'),'online_epoch':epoch,'measurement_sigma':sigma,'config':cfg}
  tmp=run/'last.partial.pt';torch.save(checkpoint,tmp);tmp.replace(run/'last.pt')
  if averages.get('active_score_points',0)>0 and ev['criterion']<best:
   best=ev['criterion'];tmp=run/'best.partial.pt';torch.save(checkpoint,tmp);tmp.replace(run/'best.pt');save(run/'best_selection.json',{'epoch':epoch,'criterion':best,'rule':'minimum fixed development observed reconstruction objective'})
  for rel,h in source_hash.items():assert sha(P/rel)==h,'training source changed during run'
  with (P/'docs/无接触标签联合学习_LIBERO-Object网络与训练验证方案.md').open('a') as f:
   f.write('\n\n### E2 实际训练进度 — '+now()+'\n\n第'+str(epoch)+'/'+str(cfg['epochs'])+'轮完成；累计优化器更新'+str(updates)+'，本轮有效窗口'+str(len(metrics))+'，观测缺失窗口'+str(len(skipped))+'。开发观测重建目标='+str(ev['criterion'])+'。这是无接触真值的观测指标，不是抓取成功率或接触精度；结果以固定规则选择best，未读取人工范围或留出视频。\n')
 freeze={'time':now(),'best_sha256':sha(run/'best.pt'),'last_sha256':sha(run/'last.pt'),'source_hashes':source_hash,'config_sha256':sha(B/'config.json'),'epochs':cfg['epochs'],'trained_episodes':400,'development_episodes':50,'test_case':'pick_up_the_bbq_sauce_and_place_it_in_the_basket__demo_40','test_inputs_read':False}
 save(run/'frozen_for_test.json',freeze);state('complete_ready_for_frozen_test',epochs=cfg['epochs'],optimizer_updates=updates)
if __name__=='__main__':
 try:main()
 except Exception as e:
  (B/'logs/training_traceback.log').write_text(traceback.format_exc());save(B/'training_failure.json',{'time':now(),'exception':type(e).__name__,'detail':str(e)[:300]});state('failed')
