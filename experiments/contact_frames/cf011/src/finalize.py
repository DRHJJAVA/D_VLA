from common import *
import numpy as np,torch,h5py,sys,ast
sys.path.insert(0,str(ROOT/'src/v3'))
from stage2_probe_motion_fingerpads import cloud
def phash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.cpu().numpy().tobytes())
 return h.hexdigest()
def check_sources():
 checks=[]
 for ep in CFG['train_episodes']:
  old=ROOT/'experiments/contact_frames/cf005/data'/ep
  c=json.loads((old/'config.json').read_text())
  with np.load(old/'artifacts/reference.npz') as ref:
   init_xyz=ref['surface_xyz'];init_uv=ref['surface_uv'].astype(np.int64)
  with np.load(old/'artifacts/dataset.npz') as z:state=z['states'].copy();sv=z['state_valid'].copy()
  with np.load(BASE/'data'/f'{ep}.npz') as z:
   T=len(z['frames']);xyz=z['xyz'];pixels=z['pixels'];valid=z['point_valid'];mask=z['mask'];target=z['target'];center=z['center']
   assert np.allclose(z['states'][:,3:],np.nan_to_num(state[:,3:]),atol=0,rtol=0)
   assert np.allclose(z['states'][:,:3],np.nan_to_num((state[:,:3]-center)/.1),atol=1e-6,rtol=0)
   assert np.array_equal(z['eligible'],np.flatnonzero((np.arange(T)>=3)&(np.arange(T)<T-4)&sv&mask.any((1,2))))
   for j,k in enumerate(CFG['horizons']):
    assert np.array_equal(mask[:-k,j],valid[:-k]&valid[k:])
    assert not mask[-k:,j].any()
    assert np.array_equal(target[:-k,j],xyz[k:]-xyz[:-k])
   maxerr=0.
   with h5py.File(ROOT/c['rgbd'],'r') as h:
    d=h['agentview/depth_m'][0]
    a=cloud(d,h['agentview/K'][0],h['agentview/camera_to_world'][0]).reshape(*d.shape,3)
    initerr=float(np.max(np.abs(a[init_uv[:,1],init_uv[:,0]]-init_xyz)))
    assert initerr<1e-7
    for t in range(T):
     d=h['agentview/depth_m'][t];K=h['agentview/K'][t].astype(np.float64);C=h['agentview/camera_to_world'][t].astype(np.float64)
     uv=pixels[t];dep=d[uv[:,1],uv[:,0]]
     rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T
     world=(rays*dep[:,None])@C[:3,:3].T+C[:3,3]
     if valid[t].any():maxerr=max(maxerr,float(np.max(np.abs(world[valid[t]]-xyz[t,valid[t]]))))
    assert maxerr<1e-6
   checks.append(dict(episode=ep,initial_surface_source_max_m=initerr,independent_depth_reconstruction_max_m=maxerr,target_and_valid_mask_exact=True))
 return checks
def main():
 summary=json.loads((BASE/'summary.json').read_text())
 assert summary['n_runs']==24
 inputs=verify(json.loads((BASE/'input_locked.json').read_text()))
 data_count=verify(json.loads((BASE/'data_locked.json').read_text()))
 sources=verify(json.loads((BASE/'source_locked.json').read_text()))
 reconstruction=check_sources()
 checks=[];table=[];strata=[]
 for ep in CFG['train_episodes']:
  metrics={arm:[] for arm in CFG['arms']}
  with np.load(ROOT/'experiments/contact_frames/cf005/data'/ep/'artifacts/dataset.npz') as z:distance=z['distance'].copy()
  for seed in CFG['seeds']:
   fold=BASE/'runs'/ep/f'seed_{seed}'
   plan=json.loads((fold/'plan.json').read_text())
   assert ep not in plan['train'] and set(plan['train'])==set(CFG['train_episodes'])-{ep}
   assert len(plan['steps'])==600 and all(item[0] in plan['train'] for batch in plan['steps'] for item in batch)
   init=torch.load(fold/'initialization.pt',map_location='cpu',weights_only=False)
   assert phash(init)==plan['initialization_sha256']
   cnt={e:sum(batch[0][0]==e for batch in plan['steps']) for e in plan['train']}
   assert set(cnt.values())=={200}
   arrays={}
   for arm in CFG['arms']:
    out=fold/arm;hist=json.loads((out/'history.json').read_text())
    assert len(hist)==600 and [r['step'] for r in hist]==list(range(1,601))
    assert [r['episode'] for r in hist]==[batch[0][0] for batch in plan['steps']]
    ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    assert ck['steps']==600 and ck['init_hash']==plan['initialization_sha256']
    with np.load(out/'predictions.npz') as z:arrays[arm]={k:z[k].copy() for k in ['frames','mean','sigma','target','mask','nll']}
    a=arrays[arm];m=a['mask']
    assert np.isfinite(a['mean']).all() and np.isfinite(a['sigma']).all()
    assert a['sigma'].min()>=.002-1e-8 and a['sigma'].max()<=.02+1e-8
    # Independently recompute NLL in the registered 10 mm units.
    nll=.5*np.sum(((a['mean']-a['target'])/a['sigma'][...,None])**2,axis=-1)+3*np.log(a['sigma']/.01)+1.5*np.log(2*np.pi)
    assert np.allclose(nll[m],a['nll'][m],atol=.003,rtol=2e-5)
    mt=json.loads((out/'metrics.json').read_text())['rows'];metrics[arm].append(mt)
    checks.append(dict(episode=ep,seed=seed,arm=arm,updates=600,initialization_and_plan=True,nll_reconstructed=True,source_hash=sha(out/'last.pt')))
   a=arrays['object_history'];b=arrays['robot_conditioned']
   for k in ['frames','target','mask']:assert np.array_equal(a[k],b[k])
   dist=distance[a['frames']]
   bins=[('0-3mm',0,.003),('3-10mm',.003,.01),('10-30mm',.01,.03),('over30mm',.03,float('inf'))]
   for label,lo,hi in bins:
    frame=(dist>=lo)&(dist<hi)&np.isfinite(dist)
    for j,h in enumerate(CFG['horizons']):
     v=a['mask'][:,j]&frame[:,None]
     if not v.any():strata.append(dict(episode=ep,seed=seed,bin=label,horizon=h,count=0));continue
     ea=np.linalg.norm(a['mean'][:,j]-a['target'][:,j],axis=-1)
     eb=np.linalg.norm(b['mean'][:,j]-b['target'][:,j],axis=-1)
     strata.append(dict(episode=ep,seed=seed,bin=label,horizon=h,count=int(v.sum()),nll_gain=float((a['nll'][:,j]-b['nll'][:,j])[v].mean()),error_gain_mm=float((ea-eb)[v].mean()*1000)))
  for j,h in enumerate(CFG['horizons']):
   row=dict(episode=ep,horizon=h)
   for arm in CFG['arms']:
    ar=[x[j]['all'] for x in metrics[arm]]
    row[arm]={key:float(np.mean([x[key] for x in ar])) for key in ['error_mean_mm','nll','sigma_mean_mm','axis95_coverage','zero_motion_mean_mm']}
   row['nll_gain']=row['object_history']['nll']-row['robot_conditioned']['nll']
   row['error_gain_mm']=row['object_history']['error_mean_mm']-row['robot_conditioned']['error_mean_mm']
   table.append(row)
 save(BASE/'audits/independent_reconstruction.json',dict(time=now(),rows=reconstruction))
 save(BASE/'audits/completion.json',dict(time=now(),input_files=inputs,data_files=data_count,training_source_files=sources,checks=checks,total_updates=14400,heldout_episode_isolation=True,human_ranges_used=False,reference_initial_surface_verified=True,method='pseudo-correspondence dynamics, not contact truth'))
 save(BASE/'comparison.json',dict(time=now(),table=table,strata=strata,positive_gain='robot-conditioned has lower error/NLL',distance='old measured geometry distance; diagnostic binning only, never contact labels',strata_limits='same distance bin does not ensure matched action/scene; not causal effect'))
 winerr=sum(r['error_gain_mm']>0 for r in table);winnll=sum(r['nll_gain']>0 for r in table)
 avgerr=float(np.mean([r['error_gain_mm'] for r in table]));avgnll=float(np.mean([r['nll_gain'] for r in table]))
 lines=['\n\n### CF011 完成结果（'+now()+'）\n',
 '完成4个留一示范折、2个输入对照、3种子，共24次训练，每次600步，总14400实际更新。未读取人工接触范围进行训练、选模型或本轮评价；预留测试未访问。冻结轨迹只作为监督伪对应。所有训练使用末步检查点。\n',
 '| 留出示范 | 预测跨度/帧 | 视觉历史误差/mm | 加机器人状态误差/mm | NLL改善（正为改善） |\n|---|---:|---:|---:|---:|']
 for r in table:lines.append('|'+r['episode']+'|'+str(r['horizon'])+'|'+format(r['object_history']['error_mean_mm'],'.3f')+'|'+format(r['robot_conditioned']['error_mean_mm'],'.3f')+'|'+format(r['nll_gain'],'.3f')+'|')
 lines += ['\n表中误差为有效伪对应点的平均欧氏误差，再对三种子等权；12行是4事件×3预测跨度，不是12独立接触事件。平均误差改善 '+format(avgerr,'.3f')+' mm，NLL改善 '+format(avgnll,'.3f')+'；误差改善行数 '+str(winerr)+'/12，NLL改善 '+str(winnll)+'/12。不能将这些数值解释为接触准确率。',
 '\n两组误差、零位移基线、预测标准差和95%轴向覆盖率保存在 comparison.json 和各run metrics.json。距离分层仅复用已锁定的几何距离做诊断，分层不同于同场景干预，动作条件增益不是因果证明。历史速度基线使用离线轨迹身份，仅为诊断。',
 '\n输入/源/数据哈希核对、24个计划/初始化/600步、逐点实际深度重建、初始表面来源、标签位移与可见掩码、独立NLL复算通过。预处理输入先锁；训练源码锁在优化前建立，但准备脚本的锁晚于其启动，不称全部源码都在数据准备前锁定。',
 '\n本轮未训练新的接触分类器，未覆盖被遮挡区域、未新增物理接触标签、未修改CF010页面，第二步仍未验收。下一步必须依据逐事件误差、零位移基线及不确定性检查决定是否采用该动态证据，不能直接将正NLL改善阈值化成接触。\n']
 with (ROOT/'docs/小型验证方案_夹持运动证据与区域表示_v1.md').open('a') as f:f.write('\n'.join(lines))
 stage('complete_dynamics_control',runs=24,contact_classifier_trained=False,error_gain_mm=avgerr,nll_gain=avgnll)
 save(BASE/'completed_locked.json',lock([x for x in BASE.rglob('*') if x.is_file() and x.suffix not in ['.log','.pyc'] and x.name not in ['completed_locked.json','runner.json']]))
 print(json.dumps(dict(stage='complete',runs=24,error_gain_mm=avgerr,nll_gain=avgnll,error_improved_rows=winerr,nll_improved_rows=winnll)))
if __name__=='__main__':
 try:main()
 except Exception as e:failure(e);raise SystemExit(1)
