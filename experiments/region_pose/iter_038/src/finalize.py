from pathlib import Path
import json,hashlib,datetime,ast,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA');E=R/'experiments/region_pose';I=E/'iter_038'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
s=json.loads((I/'summary.json').read_text());assert len(s['episodes'])==1 and s['episodes'][0]['split']=='val'
old_unit=s['unit'];s['unit']='1 old-validation episode, 1 attempt; frames are not independent grasps'
save(I/'summary_corrected.json',s)
assert s['episodes']==json.loads((I/'summary.json').read_text())['episodes']
save(I/'audits/count_correction.json',{'original_preserved':True,'original_sha256':sha(I/'summary.json'),'new_sha256':sha(I/'summary_corrected.json'),'changed_field':'unit','old_unit':old_unit,'new_unit':s['unit'],'episode_data_equal':True,'no_recomputation':True})
z=np.load(I/'cases/tomato_sauce_demo_4.npz');offs=z['frame_offsets'];reps=z['representative_indices'];rows=[]
for name,start,end in [('before_attempt',1,49),('attempt',50,113),('after_attempt',114,132)]:
 lo,hi=int(offs[start]),int(offs[end+1]);rows.append({'phase':name,'start':start,'end':end,'observations':hi-lo,'representatives':int(((reps>=lo)&(reps<hi)).sum()),'camera_observations':{str(c):int((z['camera'][lo:hi]==c).sum()) for c in [0,1]}})
save(I/'audits/phase_counts.json',rows)
checks={}
for name in ['source_locked.json','input_locked.json','finalize_source_locked.json']:
 d=json.loads((I/name).read_text())
 for k,v in d.items():assert sha(Path(k))==v
 checks[name]=len(d)
for name in ['033','037']:
 d=json.loads((E/f'iter_{name}/completed_locked.json').read_text())
 for k,v in d.items():assert sha(R/k)==v
 checks['old'+name]=len(d)
audit=json.loads((I/'audits/independent_reconstruction.json').read_text())
assert audit['episodes'][0]['attempts'][0]['full_candidate_accepted']==0
assert json.loads((I/'audits/completion_audit.json').read_text())['all_original_narrow_points_pixels_cameras_sides_counts_equal']
save(I/'待人工判断.json',{'stage2_approved':False,'unresolved_old_validation':'tomato_sauce_demo_4','reason':'full finite volume longest intersection run 3 < 5; no held supervision restored','not_grasp_failure':True,'pending':['来源真实身份不可由几何/连续性诊断确认','侧向示范数据范围'],'rest_no_question':True})
save(I/'audits/final_integrity.json',{'passed':True,'checks':checks,'summary_metadata_correction_only':True,'training_freeze_before_validation':True,'validation_episode_count':1,'test_access':False,'new_training_labels':False})
now=datetime.datetime.now(datetime.timezone.utc).isoformat();save(I/'stage_state.json',{'stage':'complete','utc':now})
with (R/'docs/STATUS.md').open('a') as f:f.write('\n\n第038轮完成（'+now+'）：冻结033方法检查旧验证tomato4唯一尝试50–113。窄垫复现相同；完整体积单帧交集64/68/70/71/72、最长70–72三帧，未达连续5，仍0接受。全段10995观测主要闭合前，尝试内53全主相机；不当成功/失败真值、不恢复标签。独立重投影/连续性及22输入、旧033/037锁通过。038.1仅更正静态unit为1旧验证episode/1尝试，原件保留。下一步039固定来源查看，不读测试；第二步未验收。\n')
with (E/'迭代索引.md').open('a') as f:f.write('\n- 038：旧验证tomato4冻结有限体积来源范围对照完成；最长交集连续3帧，0接受，仍缺监督；仅summary单位描述附录更正，见iter_038/结果与下一步.md。\n')
files=[p for p in I.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['completed_locked.json','finalize.log']]
lock={str(p.relative_to(R)):sha(p) for p in sorted(files)};save(I/'completed_locked.json',lock);assert all(sha(R/k)==v for k,v in lock.items())
print(json.dumps({'complete':True,'locked_files':len(lock)}))
