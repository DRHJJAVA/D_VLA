from common12 import *
import numpy as np,torch,itertools
def readpred(p):
 with np.load(p) as z:return {k:z[k].copy() for k in ['frames','mean','sigma','target','mask','nll']}
def main():
 done=json.loads((BASE/'training_complete.json').read_text());assert done['runs']==36 and done['updates']==21600
 oldfiles=verify(json.loads((PREV/'completed_locked.json').read_text()))
 src=verify(json.loads((BASE/'source_locked.json').read_text()))
 verify(json.loads((BASE/'input_locked.json').read_text()))
 verify(json.loads((BASE/'baseline_source_locked.json').read_text()));verify(json.loads((BASE/'baseline_input_locked.json').read_text()));verify(json.loads((BASE/'baseline_locked.json').read_text()))
 verify(json.loads((BASE/'calibration_locked.json').read_text()))
 checks=[];norm_checks=[]
 for pi,pair in enumerate(itertools.combinations(CFG['episodes'],2)):
  states=[]
  for ep in pair:
   with np.load(PREV/'data'/f'{ep}.npz') as z:states.append(z['states'][z['eligible']])
  st=np.concatenate(states);mu=st.mean(0);sd=st.std(0,ddof=1).clip(.01)
  for seed in CFG['seeds']:
   fold=BASE/'inner'/f'pair_{pi}'/f'seed_{seed}';plan=json.loads((fold/'plan.json').read_text())
   assert plan['training']==list(pair) and set(plan['evaluation'])==set(CFG['episodes'])-set(pair)
   assert len(plan['steps'])==600 and all(len(x)==4 for x in plan['steps'])
   assert all(it[0] in pair for batch in plan['steps'] for it in batch)
   assert all(sum(batch[0][0]==ep for batch in plan['steps'])==300 for ep in pair)
   assert np.allclose(mu,plan['norm_mean'],atol=2e-5,rtol=2e-5)
   assert np.allclose(sd,plan['norm_std'],atol=2e-5,rtol=2e-5)
   init=torch.load(fold/'initialization.pt',map_location='cpu',weights_only=False)
   ih=old.parameter_hash(init);assert ih==plan['initialization_hash']
   norm_checks.append(dict(pair=pi,seed=seed,only_training_events=True))
   for arm in CFG['arms']:
    out=fold/arm;hist=json.loads((out/'history.json').read_text())
    assert len(hist)==600 and [x['step'] for x in hist]==list(range(1,601))
    assert [x['episode'] for x in hist]==[x[0][0] for x in plan['steps']]
    ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert ck['steps']==600 and ck['init_hash']==ih
    assert old.parameter_hash(init)==ih
    checks.append(dict(pair=pi,seed=seed,arm=arm,steps=600,initialization_hash=ih))
 frozen=json.loads((BASE/'calibration_frozen.json').read_text());alpha_checks=[]
 for r in frozen['records']:
  held=r['held_out'];arm=r['arm'];qs=[];seen=[]
  for s in r['sources']:
   assert held not in s['training'] and held!=s['calibration_episode']
   assert s['calibration_episode'] not in s['training']
   assert set(s['training'])|{s['calibration_episode']}==set(CFG['episodes'])-{held}
   assert sha(s['prediction'])==s['prediction_sha256'] and sha(s['checkpoint'])==s['checkpoint_sha256']
   a=readpred(s['prediction']);q=[]
   for j in range(3):
    v=a['mask'][:,j];e=a['mean'][:,j].astype(np.float64)-a['target'][:,j]
    q.append(float(((e**2).sum(-1)/(3*a['sigma'][:,j].astype(np.float64)**2))[v].mean()))
   qs.append(q);seen.append((s['calibration_episode'],s['seed']))
  assert len(set(seen))==9 and len(qs)==9
  alpha=np.sqrt(np.mean(qs,axis=0))
  assert np.allclose(alpha,r['alpha'],rtol=1e-6,atol=1e-7)
  alpha_checks.append(dict(held_out=held,arm=arm,alpha=alpha.tolist(),no_outer_event_in_training_or_calibration=True))
 calibrated=json.loads((BASE/'calibrated_summary.json').read_text())['runs'];assert len(calibrated)==24
 summary=[];baseline_summary=[];raw_nll=[];cal_nll=[];cov_raw=[];cov_cal=[];width_raw=[];width_cal=[]
 for ep in CFG['episodes']:
  with np.load(BASE/'causal_baseline'/f'{ep}.npz') as z:vel=z['mean'];vvalid=z['valid']
  for arm in CFG['arms']:
   selected=[r for r in calibrated if r['held_out']==ep and r['arm']==arm];assert len(selected)==3
   for seed in CFG['seeds']:
    a=readpred(PREV/'runs'/ep/f'seed_{seed}'/arm/'predictions.npz')
    b=readpred(BASE/'calibrated'/ep/f'seed_{seed}'/arm/'predictions.npz')
    for key in ['frames','mean','target','mask']:assert np.array_equal(a[key],b[key])
    alpha=next(r['alpha'] for r in frozen['records'] if r['held_out']==ep and r['arm']==arm)
    assert np.allclose(b['sigma'],a['sigma']*np.array(alpha)[None,:,None],atol=1e-10,rtol=1e-7)
    m=b['mask'];e=b['mean'].astype(np.float64)-b['target'];s=b['sigma']
    ll=.5*((e/s[...,None])**2).sum(-1)+3*np.log(s/.01)+1.5*np.log(2*np.pi)
    assert np.allclose(ll[m],b['nll'][m],atol=.003,rtol=1e-5)
    for j,h in enumerate(CFG['horizons']):
     good=a['mask'][:,j]&vvalid[a['frames']]
     delta=a['target'][:,j];net=a['mean'][:,j];speed=vel[a['frames'],j]
     n=int(good.sum());total=int(a['mask'][:,j].sum())
     baseline_summary.append(dict(episode=ep,seed=seed,arm=arm,horizon=h,count=n,total=total,valid_fraction=n/total,network_mm=float(np.linalg.norm(net-delta,axis=-1)[good].mean()*1000),causal_velocity_mm=float(np.linalg.norm(speed-delta,axis=-1)[good].mean()*1000),zero_mm=float(np.linalg.norm(delta,axis=-1)[good].mean()*1000)))
   for j,h in enumerate(CFG['horizons']):
    rr=[r['rows'][j] for r in selected]
    line=dict(episode=ep,arm=arm,horizon=h,alpha=float(rr[0]['alpha']),error_mm=float(np.mean([r['error_mm'] for r in rr])))
    for method in ['raw','calibrated']:
     line[method]={key:float(np.mean([r[method][key] for r in rr])) for key in rr[0][method]}
    summary.append(line);raw_nll.append(line['raw']['nll']);cal_nll.append(line['calibrated']['nll'])
    cov_raw.append(line['raw']['coverage95']);cov_cal.append(line['calibrated']['coverage95'])
    width_raw.append(line['raw']['width95_mm']);width_cal.append(line['calibrated']['width95_mm'])
 save(BASE/'audits/independent_completion.json',dict(time=now(),old_cf011_locked_files_verified=oldfiles,source_files_verified=src,training_checks=checks,normalization=norm_checks,calibration_sources=alpha_checks,all_prediction_means_unchanged=True,all_valid_masks_unchanged=True,calibration_formula_and_nll_recomputed=True,updates=21600))
 result=dict(time=now(),rows=summary,baseline=baseline_summary,aggregate=dict(raw_nll=float(np.mean(raw_nll)),calibrated_nll=float(np.mean(cal_nll)),raw_coverage95=float(np.mean(cov_raw)),calibrated_coverage95=float(np.mean(cov_cal)),raw_width95_mm=float(np.mean(width_raw)),calibrated_width95_mm=float(np.mean(width_cal))),semantics='development pseudo-correspondence dynamics; no contact classification or physical truth')
 save(BASE/'comparison.json',result)
 baseline_complete=json.loads((BASE/'baseline_complete.json').read_text())
 lines=['\n\n### CF012 实际结果（'+now()+'）\n',
 '36次内层训练全部完成，各600步，共21600更新；先锁定8组校准参数（4外层事件×2分支、每组3预测跨度），再应用24份CF011原预测。原模型均值、目标、有效掩码逐项未变。原CF011全部149个锁定文件、36初始化/计划/步数、内层归一化、校准来源事件隔离、尺度公式与NLL独立复算通过。',
 '\n| 留出示范 | 分支 | 跨度 | alpha | NLL 原→校准 | 95%轴向覆盖 原→校准 | 95%区间全宽/mm 原→校准 |\n|---|---|---:|---:|---:|---:|---:|']
 for x in summary:
  a=x['raw'];b=x['calibrated']
  lines.append('|'+x['episode']+'|'+x['arm']+'|'+str(x['horizon'])+'|'+format(x['alpha'],'.3f')+'|'+format(a['nll'],'.3f')+' → '+format(b['nll'],'.3f')+'|'+format(100*a['coverage95'],'.1f')+'% → '+format(100*b['coverage95'],'.1f')+'%|'+format(a['width95_mm'],'.2f')+' → '+format(b['width95_mm'],'.2f')+'|')
 a=result['aggregate']
 lines+=['\n按事件、跨度、分支、种子等权：NLL '+format(a['raw_nll'],'.3f')+' → '+format(a['calibrated_nll'],'.3f')+'；95%逐坐标覆盖 '+format(100*a['raw_coverage95'],'.1f')+'% → '+format(100*a['calibrated_coverage95'],'.1f')+'%；区间全宽 '+format(a['raw_width95_mm'],'.2f')+' → '+format(a['calibrated_width95_mm'],'.2f')+' mm。更宽区间不能单独算接触识别收益；也没有新的接触分数。各事件必须分别检查，不能用总体均值掩盖。',
 '\n#### 因果速度基线（相同有效评价点）\n',
 '| 留出示范 | 跨度 | 有效比例 | 零位移/mm | 因果恒速/mm | 视觉历史网络/mm | 机器人条件网络/mm |\n|---|---:|---:|---:|---:|---:|---:|']
 for ep in CFG['episodes']:
  for h in CFG['horizons']:
   rr=[x for x in baseline_summary if x['episode']==ep and x['horizon']==h]
   fn=lambda key:float(np.mean([x[key] for x in rr]))
   e0=float(np.mean([x['network_mm'] for x in rr if x['arm']=='object_history']))
   ea=float(np.mean([x['network_mm'] for x in rr if x['arm']=='robot_conditioned']))
   lines.append('|'+ep+'|'+str(h)+'|'+format(100*fn('valid_fraction'),'.1f')+'%|'+format(fn('zero_mm'),'.3f')+'|'+format(fn('causal_velocity_mm'),'.3f')+'|'+format(e0,'.3f')+'|'+format(ea,'.3f')+'|')
 lines += ['\n该基线读取完整t−1/t RGB做稠密流，再在固定评价位置取样，不读取未来图像或离线轨迹来构造流。评价掩码仍使用共同的伪对应目标；这是相同点集上的预测对照，不证明追踪身份和物理接触。缺失流保持未知，未用于负例。',
 '\n本轮未引入人工范围训练，未读预留测试，未改变CF010接触输出，第二步仍未验收。下一步应先依据逐事件NLL/宽度和因果速度基线决定动态表示是否足够，再设计接触头对照；不得仅把alpha或NLL增益当接触阈值。\n']
 with (ROOT/'docs/小型验证方案_夹持运动证据与区域表示_v1.md').open('a') as f:f.write('\n'.join(lines))
 stage('complete_calibration_control',inner_runs=36,contact_classifier_trained=False,aggregate=a)
 save(BASE/'completed_locked.json',lock([x for x in BASE.rglob('*') if x.is_file() and x.suffix not in ['.log','.pyc'] and x.name not in ['completed_locked.json']]))
 print(json.dumps(dict(stage='complete',aggregate=a)))
if __name__=='__main__':
 try:main()
 except Exception as e:failure(e);raise SystemExit(1)
