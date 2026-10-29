
import pathlib,json,hashlib,datetime,math,traceback
B=pathlib.Path('/mnt/nvme1/drh/D_VLA');R=B/'experiments/contact_joint_object/runs/v002/preflight/P1_interaction_v1';A=B/'docs/lcjp002_interaction_v1_assets'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ['seconds','association_source_sha256','video']}
 if isinstance(x,list):return [clean(z) for z in x]
 return x
def save(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2));q.replace(p)
try:
 assert (R/'association_all_complete.json').exists()
 d=json.loads((A/'results.json').read_text());old=json.loads((R/'before_current_only_geometry/results.json').read_text());oldmap={e['id']:e for e in old['episodes']}
 eq=[]
 for ep in d['episodes']:
  ref=oldmap[ep['id']];same=clean(ep)==clean(ref);assert same
  ep['video']=ref['video'];eq.append({'episode':ep['id'],'all_numeric_decisions_equal':same,'frames':len(ep['frames'])})
 hashes=json.loads((R/'before_current_only_geometry/image_hashes.json').read_text())
 changed=[k for k,v in hashes.items() if sha(A/k)!=v];assert not changed
 now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
 report={'time':now,'episodes':eq,'compared_images':len(hashes),'changed_images':changed,'all_equal':True,'current_geometry_guard_report':'current_only_geometry_check.json'}
 save(R/'causal_fix_equivalence.json',report)
 findings=[
 '三段全部处理完：508个时间点，1016个相机画面；三个MP4逐帧解码帧数均一致。',
 '番茄酱第247帧原先缺失的外部候选已补回：腕部投影的203个共同可见像素中198个被新外部候选覆盖。',
 '橙汁第50至119帧连续关联编号8；番茄酱关联编号9、烧烤酱编号3，但两段存在短暂关联中断。',
 '抽查番茄酱第143/200帧、橙汁第80帧、烧烤酱第80帧，关联轮廓对应实际被操作物体；没有据此计算全视频准确率。',
 '首次静态接触不能新锁定；背景、物体局部重复候选和短暂失锁仍存在。190个关联帧不是正确率，其余318帧也不等于接触失败。',
 '因果读取修复后重算三段，全部关联字段与3048张图逐项一致；未来帧输入已由前缀视图阻止。'
 ]
 d['findings']=findings;d['summary_title']='三段完整视频已生成，目标跟随已有改善，仍有失锁';d['training_updates']=0;d['updated']=now;d['status']='complete';d['status_label']='处理完成'
 d['method']=['冻结SAM2.1 tiny：32×32自动分割，每16帧补检测，中间视频跟踪。','双视角真实深度变换到世界坐标，根据遮挡检查后的投影匹配统一编号。','仅两根夹指表面距离和可信世界运动提供关联证据；未训练关系网络，不输出部位评分。','共同运动支持首次关联，静止时可保持已有身份；不是所有接触类型的检测器。']
 save(A/'results.json',d)
 audit={'time':now,'reviewer':'assistant_visual_review','scope':'sampled object identity and browser playback, no pixel ground truth','samples':[{'episode':'ketchup','frames':[143,200],'correct_identity':True},{'episode':'orange_juice','frames':[80],'correct_identity':True},{'episode':'bbq_sauce','frames':[80],'correct_identity':True}],'browser':{'image_sequence_play_observed':{'episode':'bbq_sauce','from_frame':80,'observed_frame':84,'ended_frame':133,'camera_frames_matched':True},'mp4_play_observed':True,'mp4_duration_seconds':6.7},'no_full_video_accuracy_claim':True}
 save(R/'visual_and_browser_check.json',audit)
 md=next((B/'docs').glob('LCJP002_*.md'))
 with md.open('a',encoding='utf8') as f:
  f.write('\n\n### 双视角交互验证实际结果 '+now+'\n')
  f.write('\n'.join('- '+x for x in findings)+'\n')
  f.write('\n完整覆盖：番茄酱248帧（73帧关联同一编号9），橙汁126帧（70帧关联同一编号8），烧烤酱134帧（47帧关联同一编号3）。共9083个有效实例掩膜；无空视图仅表示至少存在一个候选，不表示所有物体均分对。\n')
  f.write('\n页面：[双视角交互物体锁定](LCJP002_P1_双视角交互物体锁定.html)。页面支持同步逐帧、原图/全部实例/关联物体切换和完整MP4。三段视频逐帧解码248/126/134帧，1024×576，20FPS离线回放。\n')
  f.write('\n严格因果修复记录：原辅助函数的未使用finger_displacement读取下一帧位姿，已通过CurrentGeometryView阻断；9个实际帧几何最大差0，7项边界检查通过；三段重算关联字段完全一致、3048张展示图hash一致。协议阈值和原始src未改。相关报告均保存在runs/v002/preflight/P1_interaction_v1。\n')
 print(json.dumps({'equivalence':report,'result_summary':d['summary'],'recorded':now},ensure_ascii=True))
except Exception as ex:
 out={'exception':type(ex).__name__,'frames':[{'file':pathlib.Path(t.filename).name,'line':t.lineno} for t in traceback.extract_tb(ex.__traceback__)]}
 save(R/'final_check_failure.json',out);print(json.dumps(out));raise SystemExit(1)
