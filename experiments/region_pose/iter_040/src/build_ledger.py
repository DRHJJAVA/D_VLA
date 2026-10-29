from pathlib import Path
import json,hashlib,datetime,statistics,collections,traceback
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/region_pose/iter_040'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 cfg=json.loads((D/'config.json').read_text());locks=json.loads((D/'input_locked.json').read_text())
 assert all(sha(R/k)==v for k,v in locks.items())
 source=json.loads((D/'source_locked.json').read_text());assert all(sha(R/k)==v for k,v in source.items())
 data={k:json.loads(Path(v).read_text()) for k,v in cfg['inputs'].items()}
 refs={k:str(Path(v).relative_to(R)) for k,v in cfg['inputs'].items()}
 roster={}
 for x in data['direction']['rows']:
  e=x['id'];s=x['split'];assert s in ['train','val']
  if e not in roster:roster[e]={'episode':e,'split':s,'attempts':[]}
  assert roster[e]['split']==s;roster[e]['attempts'].append(x)
 assert len(roster)==24 and sum(x['split']=='train' for x in roster.values())==20
 baseline={x['episode']:x for x in data['baseline_training']['rows']}
 assert set(baseline)=={e for e,x in roster.items() if x['split']=='train'}
 corr={}
 for k in ['tracking_train','tracking_val']:
  for x in data[k]['rows']:assert x['episode'] not in corr;corr[x['episode']]=(k,x)
 sup={x['episode']:x for x in data['supervision']['rows']}
 wide={}
 for key in ['wide_train','wide_val']:
  for x in data[key]['episodes']:wide[x['episode']]=(key,x)
 model={}
 for k in ['model029_manifest','model031_manifest']:
  model[k]=collections.defaultdict(list)
  for x in data[k]['rows']:model[k][x['episode']].append(x)
 assert data['model029_state']['stage']=='complete' and data['model031_state']['stage']=='complete'
 assert set(model['model029_manifest'])==set(model['model031_manifest'])
 early={}
 for key in ['early_train','early_val']:
  for x in data[key]['rows']:early.setdefault(x['episode'],[]).append(x)
 rows=[];attempt_rows=[]
 for e,base in sorted(roster.items()):
  row={'episode':e,'split':base['split'],'original_attempt_count':len(base['attempts']),'source_identity_verified':False,'physical_success_label':None,'references':[refs['direction']]}
  for x in base['attempts']:
   attempt_rows.append({'episode':e,'split':base['split'],'attempt':x['attempt'],'preclose_frames':x['frames'],'world_vertical_axis_angle_deg':x['vertical_axis_angle_deg'],'historical_status_not_physical_truth':x['status'],'source':refs['direction']})
  if e in baseline:
   row['old_003_011_held_case_available']=baseline[e]['group']!='no_011_held_case';row['references'].append(refs['baseline_training'])
  else:row['old_003_011_held_case_available']=e in corr
  row['full_volume_reaudit']=None
  if e in wide:
   key,x=wide[e];row['full_volume_reaudit']={'source':refs[key],'attempts':x['attempts'],'any_sustained_candidate':any(bool(a['candidate_accepted']) for a in x['attempts']),'old_motion_reused_not_independent_identity':True};row['references'].append(refs[key])
  row['latest_held_candidate_available']=row['full_volume_reaudit']['any_sustained_candidate'] if row['full_volume_reaudit'] is not None else row['old_003_011_held_case_available']
  row['correspondence_early_pass']=None;row['correspondence_not_run_reason']=None
  if e in corr:
   key,x=corr[e];row['correspondence_early_pass']=bool(x['new_early_valid']);row['correspondence_metadata']={'initial':x.get('initial'),'reason':x.get('reason'),'endpoints':x.get('endpoints'),'dual_bridge_difference':x.get('dual_bridge_difference')};row['references'].append(refs[key])
   if x.get('initial',{}).get('total',999)<8:row['correspondence_not_run_reason']='fewer_than_8_initial_points'
  if e==data['white6_correspondence']['episode']:
   x=data['white6_correspondence'];row['correspondence_early_pass']=len(x['supported_predefined_endpoints'])>=2;row['correspondence_metadata']={'supported_predefined_endpoints':x['supported_predefined_endpoints'],'first_failure_endpoint':x['first_failure_endpoint'],'later82_diagnostic_only':x['later82_diagnostic_only']};row['references'].append(refs['white6_correspondence'])
  row['stable_supervision']=None
  if e in sup:
   x=sup[e];row['stable_supervision']={k:x[k] for k in ['query','selected_frames','full_positive','full_negative','sample_positive_sources','sample_negative_sources','eligible']};row['references'].append(refs['supervision'])
  if e==data['white6_supervision']['episode']:
   x=data['white6_supervision'];row['stable_supervision']={'query':x['query'],'selected_frames':x['frames'],'full_positive':x['full']['positive'],'full_negative':x['full']['negative'],'sample_positive_sources':x['fps']['distinct_positive_query_indices'],'sample_negative_sources':x['fps']['negative'],'eligible':x['fps']['quantity_threshold_pass']};row['references'].append(refs['white6_supervision'])
  if row['stable_supervision'] is not None:
   x=row['stable_supervision'];assert x['eligible']==(x['sample_positive_sources']>=3 and x['sample_negative_sources']>=16)
  row['early_queries']=[{k:x[k] for k in ['event','query','offset','eligible','positive_sources','negative_sources']} for x in early.get(e,[])]
  if row['early_queries']:row['references'].append(refs['early_train' if base['split']=='train' else 'early_val'])
  m=model['model029_manifest'].get(e,[]);n=model['model031_manifest'].get(e,[])
  assert sorted((x['query_frame'],x['split']) for x in m)==sorted((x['query_frame'],x['split']) for x in n)
  row['actual_029_expanded_031_use']={'role':('training' if base['split']=='train' else 'old_validation') if m else 'excluded_unknown','query_count':len(m),'queries':[x['query_frame'] for x in m],'event_ids':sorted(set(x['event'] for x in m)),'completed_runs_confirmed_by_stage_state':True,'control6_not_claimed_all10':True}
  if m:
   assert row['stable_supervision']['eligible'] and len(m)==4
   row['references'] += [refs[k] for k in ['model029_manifest','model031_manifest','model029_state','model031_state']]
   row['current_category']='included_training' if base['split']=='train' else 'included_old_validation'
  elif not row['latest_held_candidate_available']:row['current_category']='no_sustained_observed_volume_support'
  elif row['correspondence_not_run_reason']:row['current_category']='too_few_initial_points'
  elif row['correspondence_early_pass'] is not True:row['current_category']='correspondence_quality_not_passed'
  else:
   assert row['stable_supervision'] is not None and not row['stable_supervision']['eligible']
   row['current_category']='too_few_stable_positive_query_sources'
  limits=[]
  if e=='alphabet_soup_demo_5':limits.append('持物端点只有8/9刚体内点；多正查询来源不能消除弱对应')
  if e=='porcelain_mug_demo_5':limits.append('58端点高留一敏感未删除；部分来源由其他合格时刻支持')
  if e in ['white_yellow_mug_demo_4','white_yellow_mug_demo_1']:limits.append('参考FPS后仅3正查询来源，刚好达到数量门槛，不代表充分监督')
  if e=='white_yellow_mug_demo_6':limits.append('22跨时区域格仅生成1稳定正查询来源，不并入训练')
  if e=='white_yellow_mug_demo_5':limits.append('低留一变动不等于可见正监督，0正来源，不作全负事件')
  if e=='tomato_sauce_demo_4':limits.append('完整有限体积交集最长连续3帧不足5，分散5帧不能合并；来源未获身份真值')
  if e=='tomato_sauce_demo_3':limits.append('首次与后次尝试分开，后次持物候选不被首次空夹否定；查询补点仅5，不能追踪')
  row['limitations']=limits;row['references']=sorted(set(row['references']));rows.append(row)
 counts={s:dict(collections.Counter(x['current_category'] for x in rows if x['split']==s)) for s in ['train','val']}
 angles={}
 for s in ['train','val']:
  a=[v for x in attempt_rows if x['split']==s for v in x['world_vertical_axis_angle_deg']]
  angles[s]={'episodes':sum(x['split']==s for x in rows),'attempts':sum(x['split']==s for x in attempt_rows),'query_observations':len(a),'min_deg':min(a),'median_deg':statistics.median(a),'max_deg':max(a),'above_30deg':sum(v>30 for v in a)}
 assert counts['train']['included_training']==10 and counts['val']['included_old_validation']==3
 assert sum(counts['train'].values())==20 and sum(counts['val'].values())==4
 assert sum(x['actual_029_expanded_031_use']['query_count'] for x in rows if x['split']=='train')==40
 assert sum(x['actual_029_expanded_031_use']['query_count'] for x in rows if x['split']=='val')==12
 anchor=[];unanchored=[]
 for rel,h in locks.items():
  p=R/rel;parent=next(a for a in p.parents if a.name.startswith('iter_'));lp=parent/'completed_locked.json'
  if lp.exists():
   old=json.loads(lp.read_text())
   if rel in old:assert old[rel]==h;anchor.append(rel)
   else:unanchored.append(rel)
  else:unanchored.append(rel)
 save(D/'episode_ledger.json',{'unit':'original episode','rows':rows})
 save(D/'attempt_ledger.json',{'unit':'closing attempt; not independent episode','rows':attempt_rows})
 summary={'episode_denominators':{'train':20,'old_val':4},'category_counts':counts,'actual_sample_counts':{'train_events':10,'train_queries':40,'old_val_events':3,'old_val_queries':12},'held_candidate_metadata_available':{s:sum(x['latest_held_candidate_available'] for x in rows if x['split']==s) for s in ['train','val']},'correspondence_early_pass':{s:sum(x['correspondence_early_pass'] is True for x in rows if x['split']==s) for s in ['train','val']},'full_volume_reaudited_episode_count':sum(x['full_volume_reaudit'] is not None for x in rows),'direction':angles,'no_new_training_or_labels':True,'no_test_read':True,'not_success_rate':True,'stage2_accepted':False}
 save(D/'summary.json',summary)
 save(D/'audits/input_provenance.json',{'old_completed_lock_matches':anchor,'not_in_old_completed_lock':unanchored,'all_current_inputs_prelocked_and_unchanged':True,'note':'后补元数据未在所属旧完成锁内时不冒称旧锁覆盖'})
 save(D/'audits/completion.json',{'roster_24_equals_train20_oldval4':True,'attempts_retained':len(attempt_rows),'every_episode_category_unique':True,'supervision_gate_matches_counts':True,'actual029_031_query_membership_equal':True,'only_json_metadata_read':True,'current_input_hashes_unchanged':all(sha(R/k)==v for k,v in locks.items()),'source_hashes_unchanged':all(sha(R/k)==v for k,v in source.items())})
 labels={'included_training':'纳入训练','included_old_validation':'纳入旧验证','no_sustained_observed_volume_support':'缺持续有限区域支持','too_few_initial_points':'初始点不足未追踪','correspondence_quality_not_passed':'对应质量未过','too_few_stable_positive_query_sources':'稳定正查询来源不足'}
 lines=['# 第040轮冻结证据覆盖清单','仅元数据汇总，不新增监督或网络结果。未知不是失败。表中按episode计，尝试另表；训练10事件40查询，旧验证3事件12查询。','| episode | 划分 | 原尝试数 | 持物候选 | 早期对应 | FPS正来源 | 实际查询数 | 当前状态 |','|---|---|---:|---|---|---:|---:|---|']
 for x in rows:
  sp=x['stable_supervision'];pos=str(sp['sample_positive_sources']) if sp else '未构建';c='通过数值规则' if x['correspondence_early_pass'] else ('未过/未运行' if x['correspondence_early_pass'] is False else '未运行')
  lines.append('|'+ '|'.join([x['episode'],x['split'],str(x['original_attempt_count']),'有候选' if x['latest_held_candidate_available'] else '未满足',c,pos,str(x['actual_029_expanded_031_use']['query_count']),labels[x['current_category']]])+'|')
 lines+=['','全体积重提来源只覆盖033五训练和038一旧验证；其余候选仍来自历史证据，不能说全部24段均重新提取。每行出处、尝试与限制见episode_ledger.json；闭合尝试明细见attempt_ledger.json。','方向统计：'+json.dumps(angles,ensure_ascii=False),'世界竖直参考不是仿真物体姿态，也不证明局部抓取方向。所有78个时刻均不足30度，同事件3帧不是独立抓取。','未读取预留测试、RGB-D、模型权重或分数。']
 (D/'覆盖清单.md').write_text('\n\n'.join(lines))
 assert all(sha(R/k)==v for k,v in locks.items())
 print(json.dumps({'stage':'aggregated','counts':counts,'direction':angles,'old_lock_matches':len(anchor),'not_in_old_lock':unanchored},ensure_ascii=True))
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(D/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback_remote_only':traceback.format_exc()});print(json.dumps({'failed':True,'type':type(e).__name__}))
