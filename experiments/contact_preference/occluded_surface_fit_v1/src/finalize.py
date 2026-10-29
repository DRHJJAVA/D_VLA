from pathlib import Path
import json,hashlib,re
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA');O=R/'experiments/contact_preference/occluded_surface_fit_v1';page=R/'docs/接触偏好_遮挡表面定位与两表面贴合估计.html';doc=R/'docs/夹爪物体接触偏好学习_完整方案.md'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def main():
 time=now();summ=json.loads((O/'summary.json').read_text());audit=json.loads((O/'audits/surface_audit.json').read_text());images=set();frames=0;maxWebDiff=0.;oldcounts=[];table=[];totals={'surface_estimates':0,'active_gripper_occluded':0,'any_robot_occluded':0,'confirmed_physical_contact':0}
 for ci,c in enumerate(summ['cases']):
  w=json.loads((O/'webdata'/f'case{ci}.json').read_text());images.add(w['query_image']);cc=[]
  for f,ar in zip(w['frames'],audit['cases'][ci]['rows']):
   t=f['frame'];frames+=1;z=np.load(O/'points'/f"{c['key']}_frame{t}.npz");old=np.load(R/'experiments/contact_preference/stage3_pilot_v2/points_evidence'/f"{c['key']}_frame{t}.npz")
   assert np.array_equal(f['old_class'],old['display_class'])
   assert np.array_equal(f['old_positive'],old['positive'])
   active=sum(p['active_occluded'] for p in f['points']);assert active==ar['active_gripper_occluded']==f['active_occluded_count']
   for p in f['points']:
    idx=p['id'];maxWebDiff=max(maxWebDiff,abs(p['gap']-z['gap_m'][idx]*1000),abs(p['lower']-z['gap_lower_m'][idx]*1000),abs(p['upper']-z['gap_upper_m'][idx]*1000))
    assert p['status']==int(z['status'][idx])
   images.update(v['image'] for v in f['views']);cc.append(int((old['display_class']==0).sum()))
   a=f['summary'];totals['surface_estimates']+=len(f['points']);totals['active_gripper_occluded']+=active;totals['any_robot_occluded']+=a['robot_occluded_estimates']
   table.append(f"| {c['title']} | {t} | {len(f['points'])} | {active} | {a['gap_mm_quantiles'][2]:.3f} | {a['uncertainty_mm_quantiles'][1]:.3f} | 未知 |")
  oldcounts.append({'key':c['key'],'green_counts':cc})
 assert maxWebDiff==0
 assert all((R/'docs'/p.lstrip('/')).is_file() for p in images)
 for link in re.findall(r'href="([^"]+)"',page.read_text()):
  if link.startswith('/'):assert (R/'docs'/link.lstrip('/')).is_file(),link
 projection=[]
 for c in summ['cases']:
  z=np.load(R/'experiments/region_pose/iter_024/train'/f"{c['key']}.npz");T=np.linalg.inv(z['query_camera_to_world']);x=z['query_xyz']@T[:3,:3].T+T[:3,3];K=z['query_K'];uv=x[:,:2]/x[:,2,None]*np.array([K[0,0],K[1,1]])+np.array([K[0,2],K[1,2]]);err=float(abs(uv-z['query_pixel']).max());assert err<.001;projection.append({'key':c['key'],'max_pixel_error':err})
 save(O/'audits/web_data_audit.json',{'time':time,'frames':frames,'image_files':len(images),'missing':0,'old_classification_preserved':oldcounts,'web_gap_max_difference':maxWebDiff,'query_reprojection':projection,'totals':totals})
 save(O/'audits/browser_review.json',{'time':time,'page':str(page),'cases_reviewed':['porcelain55 main','porcelain66 wrist and cloud/gap','tomato68 main and gap','tomato70 empty occluded selection and show-all','whiteyellow68 wrist, cloud focus, gap, reference image'],'interactions':['case/frame/camera switching','occlusion-only checkbox','empty selection','local cloud focus'],'limits':'Not every endpoint visually inspected; numerical display integrity covers all 12. Visual checks do not prove actual contact.','observation':'No gross image/cloud offset in reviewed views after CV projection fix. Rings intentionally overlay hidden locations. Whiteyellow68 image fully loaded and rechecked.'})
 text=f"""

### {time} 阶段3：遮挡表面定位与两表面贴合估计——实算结果与验收

本轮仅同三个训练案例、12个既有端点完成局部表面估计、两表面间距核查和网页对照。未训练评分网络，未新增物理接触正标签。原双相机融合、夹爪专属支持和遮挡期间保持三个旧版本及锁均保留。

**技术流程已完成；“可靠找出真实被遮挡接触面”的效果目标尚未通过。** 能给出少量位于夹指遮挡后的几何位置估计，但不确定范围同时容纳接触和分离，全部保持未知。橙环不是正支持，位于夹指之后也不证明贴合。

#### 实际实施与模块作用

1. 真实物体源点与定位：复用已核查的多时刻主/腕相机1mm真实代表点，运动沿用独立RGB-D对应变换W=H×Q及LOO。没有将物体固定在手上，不把夹爪贴合当配准目标。原来源筛选要求接近参考表面，所以仍是局部插值，不是未见背面补全。
2. 物体曲面：参考锚点3mm与5mm邻域分别拟合平面、二次曲面，共四模型。每邻域至少12个真实格、最多160个；稳健拟合3轮，系数仅由物体点决定。仅允许源点切平面凸包内交点，禁止外推。薄壁两侧混合及法线方向仍可能存在歧义，不能称身份真值。
3. 两表面比较：每端点4个夹指/垫面几何体各采样512点，沿外法线与物体曲面求交；正间距为分离，负间距为穿入。四模型均须有凸包内、法线相对的交点。±8mm只是搜索范围，不是接触门槛；没有移动物体去贴合夹爪，基座/安装壳体不参与接触面查询。
4. 不确定性：四模型间距包络，加上sqrt(max(旧sigma_proxy,估计点LOO位移)^2+最大拟合RMS^2)。旧sigma_proxy包含对应残差、像素足迹及机器人视觉/碰撞几何代理差异，**不是实测真实误差或校准置信区间**。使用当前双相机实际深度检查可见与自由空间冲突，另以夹指三角面射线核查遮挡。
5. 未知：区间含零仅表示不能排除贴合，仍未知；无源点支持、凸包外、模型冲突、身份/深度冲突也未知。没有强行贴合，不把缺失表面当负例；其他物体或机器人部位接触不生成正支持。

#### 全部端点结果（距离为毫米）

数量为每端点2048个夹指查询位置中的估计数，不是面积、唯一表面数量或独立接触次数。“表面交点”包含之后深度/身份不支持而标灰紫色未知的位置。

| 案例 | 帧 | 四模型表面交点 | 夹指遮挡估计 | 名义间距中位 | 合成误差尺度中位 | 接触结论 |
|---|---:|---:|---:|---:|---:|---|
"""+'\n'.join(table)+f"""

合计{totals['surface_estimates']}个端点—位置观测，其中{totals['active_gripper_occluded']}个通过夹指/垫面遮挡核查（瓷杯44、番茄酱罐2、白黄杯1），同事件多帧不能当独立接触。任意机器人遮挡共{totals['any_robot_occluded']}个，不能混同夹指遮挡。确认为物理接触的点为0，不代表视频实际没有接触。

局部拟合可以平滑、插值观测，但不能补足物体整体定位和夹指几何精度。四模型间距差异小也不代表位置准确；当前误差模型偏保守可能影响结论，尚未独立标定，不能仅缩小它来制造正支持。

#### 验收和完整性

- 41输入、拟合脚本哈希未变；旧融合65、旧夹爪支持59、上一版保持47文件锁核对通过。
- 独立IRLS系数重建最大差0；射线曲面方程最大残差{audit['checks']['independent_ray_surface_equation_max_m']:.3e}m；凸包、变换、区间分类检查通过。已知正/零/负间距平面及凸包外拒绝测试通过，仅证明实现一致，不证明真实接触。
- 每五个源点留出一个的局部曲面绝对残差中位/95分位（mm）：瓷杯0.334/0.940，番茄酱罐0.107/0.399，白黄杯0.300/0.655。该诊断采用固定最小二乘，不用于改标签；相关局部点并非独立测试。
- 当前相机深度一致、前方自由空间冲突、后方遮挡分别保存，不只报低残差子集。它是重投影自一致，不证明身份真值。
- 展示最初误用OpenGL坐标方向，原展示数据保留；v2改回实际CV正z深度、正y图像向下，三案例原始像素反投影最大误差<0.001px。表面拟合和间距没改。旧绿色直接读取原分类，不重设显示阈值。
- 12端点网页数值等于保存结果；27张引用图片全部存在。浏览器已检查三个案例的主/腕图、点云、间距剖面和空结果筛选；透过遮挡绘制的环不是可见实测像素。
- **数值实现和来源隔离通过；真实接触面定位与物理贴合效果未通过。** 本轮用于分析和查看，不能直接生成评分网络正标签。

#### 网页、产物和后续

[遮挡表面定位与两表面贴合估计](接触偏好_遮挡表面定位与两表面贴合估计.html)。左侧保留旧候选，右侧橙环是未知位置估计；下方可旋转点云、定位夹持前源点、查看两表面间距与范围。所有产物仅在远程。

目录：experiments/contact_preference/occluded_surface_fit_v1/。保存逐点NPZ、锚点曲面与真实源索引、核查、网页数据和完成锁。
拟合代码SHA256：{sha(O/'src/fit_surfaces.py')}
核查代码SHA256：{sha(O/'src/audit_and_webdata_v2.py')}

若继续，需先拆分并独立验证物体定位、相机/夹爪几何误差、表面覆盖缺口，再决定是否生成可靠正支持。不能把加点、缩小误差范围、强制贴合当解决。本轮不训练评分网络，不宣布阶段3或最终第二步通过。
"""
 with doc.open('a') as f:f.write(text)
 save(O/'document_update.json',{'time':time,'document':str(doc),'document_sha256_at_update':sha(doc)})
 save(O/'completion_audit.json',{'time':time,'computation_complete':True,'independent_audit_passed':True,'browser_review_done':True,'same_document_updated':True,'scoring_network_training':False,'physical_contact_localization_accepted':False,'totals':totals,'page':str(page)})
 save(O/'stage_state.json',{'time':time,'status':'complete','effect_acceptance':'contact_surface_still_unknown','scoring_training':False})
 files=[p for p in O.rglob('*') if p.is_file() and p.suffix!='.log' and p.name!='completed_locked.json'];files.append(page)
 save(O/'completed_locked.json',{str(p):sha(p) for p in files})
 assert all(sha(p)==h for p,h in json.loads((O/'completed_locked.json').read_text()).items())
 print(json.dumps({'status':'complete','time':time,'locked_files':len(files),'totals':totals},ensure_ascii=False))
if __name__=='__main__':main()
