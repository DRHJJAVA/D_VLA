from pathlib import Path
import json,hashlib,time
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/region_pose';B=P/'iter_026';V=P/'reviews/iter_026'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
counts={}
for p in [B/'source_locked.json',B/'source_final_addendum.json',B/'input_locked.json',P/'iter_024/completed_locked.json',P/'iter_025/completed_locked.json',V/'immutable_locked.json']:
 d=json.loads(p.read_text())
 for k,h in d.items():assert sha(R/k)==h,k
 counts[str(p.relative_to(R))]=len(d)
rows=[]
for p in (B/'cases').glob('*.json'):
 d=json.loads(p.read_text());out=[]
 for row in d['frames']:
  rr=[r for r in d['source_edges'] if r['frame']==row['frame']];good=[r for r in rr if r['positive_support']]
  out.append(dict(frame=row['frame'],supports=len(good),distinct_region_sources=len({r['region_source_id'] for r in good}),main_camera_support=sum(r['source_camera']==0 for r in good),wrist_camera_support=sum(r['source_camera']==1 for r in good),unstable_final_positive=sum(not r['stable'] for r in rr),outside_robust1_final_positive=sum(not r['robust_inside'] for r in rr)))
 rows.append(dict(episode=d['episode'],frames=out))
assert rows==json.loads((B/'audits/source_count_diagnostic.json').read_text())
dump(B/'audits/browser_review.json',dict(time=time.time(),page='docs/第026轮_自动监督来源与逐时刻支持.html',target_images_reviewed=6,source_images_reviewed=7,rigid_images_reviewed=1,loaded_page_and_links=True,temporary_tab_closed=True,observations='罐头局部左右分片；杯子杯口附近少量正点；白黄杯验证来自腕相机。未见整体落到机械臂主体；小物体遮挡与标号密集不能逐点确认身份。',manual_label_changes=False,user_stage2_approval=False))
dump(B/'audits/final_integrity.json',dict(time=time.time(),hash_counts=counts,source_count_reconstruction=True,labels_unchanged=True,all_artifacts_remote=True))
dump(B/'stage_state.json',dict(stage='complete',time=time.time(),cases=6,frames=24,images=60,scoring_training=False,stage2_user_approved=False))
note='\n\n## 026 自动监督来源与逐时刻支持完成\n6案例24端点60图保留远程。瓷杯5的58帧对最终17正来源支持0，未删帧；白黄杯验证3正点四端点均由腕相机3个不同最近源观测支持，仍非独立重复。后台检查14图与页面，目标未改、无评分训练，旧024/025锁147/108不变。页面：第026轮_自动监督来源与逐时刻支持.html；第二步未验收。下一步另登记027训练更早查询传播，先核查旧014可复用对应，保留质量与因果门槛。\n'
for f in [R/'docs/STATUS.md',P/'迭代索引.md']:
 with f.open('a',encoding='utf8') as o:o.write(note)
files=[p for p in B.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.log' and p.name!='completed_locked.json']
dump(B/'completed_locked.json',{str(p.relative_to(R)):sha(p) for p in files})
print('COMPLETE',len(files),'review_files',counts[str((V/'immutable_locked.json').relative_to(R))])
