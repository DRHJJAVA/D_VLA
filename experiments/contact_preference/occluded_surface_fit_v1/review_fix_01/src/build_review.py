from pathlib import Path
import re,json,hashlib
R=Path('/mnt/nvme1/drh/D_VLA');D=R/'experiments/contact_preference/occluded_surface_fit_v1/review_fix_01'
s=(R/'docs/接触偏好_遮挡表面定位与两表面贴合估计.html').read_text()
s=s.replace('<title>接触偏好 · 遮挡表面定位与两表面间距</title>','<title>遮挡表面 · 坐标朝向与几何分歧核查</title>').replace('<h1>遮挡表面定位 · 两表面间距</h1>','<h1>遮挡表面 · 坐标与几何核查</h1>')
s=re.sub(r'<div class="verdict">.*?</div>','<div class="verdict"><b>本次修正：不再把“没显示”解释成“没有接触”。</b><br>默认显示全部表面估计；点云朝向与当前RGB相机一致，切换帧或案例时复位。demo7 的68、73帧原标记存在明显几何分歧，用粉色×和不确定虚线说明；没有将其强行改成接触点。demo9及所有旧计算值保留。</div>',s,flags=re.S)
s=s.replace('<a href="/接触偏好_双相机跨时融合与遮挡区域.html"','<a href="/接触偏好_遮挡表面定位与两表面贴合估计.html" target="_blank">修正前网页</a> · <a href="/接触偏好_双相机跨时融合与遮挡区域.html"',1)
s=s.replace('id="hidden" checked','id="hidden"').replace('只看夹爪遮挡估计','仅看夹指遮挡估计')
s=s.replace('02 / 本轮：独立物体表面估计','02 / 名义位置与不确定范围')
s=s.replace('橙色空心环：间距不确定','橙环：未知估计；粉色×：两几何分歧')
s=s.replace('白色线框只含夹指和垫面。橙点是估计表面，短线连接夹爪表面与物体估计点。','灰白实体是夹指可视几何；可选橙线框是简化碰撞外壳。粉色×是几何分歧未知；选中虚线是法线间距范围。真实源点按原RGB着色。')
s=s.replace('<button id="reset">复位视角</button>','<button id="reset">与RGB同朝向</button>')
s=s.replace('<button id="focus">聚焦选中表面</button>','<button id="focus">聚焦选中表面</button><label><input type="checkbox" id="collision">对照简化碰撞外壳</label>')
s=s.replace('四种拟合均支持的位置估计','本帧全部表面估计（未知）')
s=s.replace('下方可旋转点云','下方可旋转点云')
s=s.replace('/图示/遮挡表面贴合/case','/图示/遮挡表面核查/case')
s=s.replace("color=p=>p.status===1?","color=p=>p.display_geometry_conflict?'#d33c83':p.status===1?")
s=s.replace("selected=list.length?list[0].i:-1;updateDetail()","selected=list.length?(list.find(x=>x.p.active_occluded)?.i??list[0].i):-1;if(selected>=0)$('point').value=selected;updateDetail()")
s=s.replace("cam=0,selected=-1","cam=0,selected=-1")
# Frame/case switch resets camera/zoom; data itself is unchanged.
s=s.replace("populate();drawImages();drawCloud();drawQuery()}","cam=+$('camera').value;resetCloud();populate();drawImages();drawCloud();drawQuery()}",1)
s=re.sub(r'function transform\(p,w,h\).*?(?=\n\$\(\'case\'\))',(D/'src/cloud_renderer.js').read_text(),s,flags=re.S)
s=re.sub(r'function drawImages\(\).*?(?=\nfunction drawQuery)',(D/'src/image_renderer.js').read_text(),s,flags=re.S)
s=re.sub(r'function updateDetail\(\).*?(?=\nfunction ring)',(D/'src/detail_renderer.js').read_text(),s,flags=re.S)
s=s.replace("$('camera').onchange=drawImages","$('camera').onchange=()=>{cam=+$('camera').value;resetCloud();drawImages()}")
s=s.replace("$('reset').onclick=()=>{angle=[-.6,.25];zoom=1;center=[0,0,.085];drawCloud()}","$('reset').onclick=resetCloud;$('collision').onchange=drawCloud")
s=s.replace("ctx.fillText('夹爪表面 0'","ctx.fillText('碰撞外壳 0'")
s=s.replace('0 是夹爪表面','0 是简化碰撞外壳表面')
s=s.replace('未知；未确认贴合</td>','未知；未确认贴合</td>')
# Explain recorded diagnosis without claiming uncertain reconstruction is fixed.
s=s.replace('<details><summary>这次具体做了什么，哪些仍没有解决</summary>','<details open><summary>为什么有些帧原来没有标记，为什么demo7会错位？</summary><p>demo7的70/78帧仍有37/34个估计，原筛选将它们全隐藏。demo4的66/71帧仍各有22个估计；71帧有4个任意机器人遮挡位置，但没有通过夹指表面遮挡核查。可见、遮挡、接触是三件不同的事；本页面没有用运输运动自动填正。</p><p>demo7的68/73帧同一名义位置，距简化碰撞外壳约1–2mm，距夹指可视表面约12.5mm。它在夹指较靠近根部的近似几何上，不能因为碰撞外壳近就说真实接触。保留未知并展示分歧，尚未解决这两帧真实接触面的定位。</p><p>demo4旧3D图使用独立手坐标观察方向，切换案例还保留旋转/缩放，杯柄容易与RGB左右相反。本页默认使用实际相机外参确定方向、深度排序和源RGB；并非把点云人为镜像。可手动旋转，点击按钮恢复。</p></details><details><summary>拟合方法与限制</summary>')
p=R/'docs/接触偏好_遮挡表面定位_坐标与几何核查.html';p.write_text(s)
link=R/'docs/图示/遮挡表面核查'
if not link.exists():link.symlink_to(D/'webdata',target_is_directory=True)
# Structural regressions before browser execution.
checks={'default_all_estimates':'id="hidden" checked' not in s,'uses_actual_camera_transform':'hand_to_camera' in s,'no_old_fixed_view_projection':'xx*5000*zoom' not in s,'resets_on_frame_change':'resetCloud();populate()' in s,'collision_optional':'id="collision"' in s,'depth_sort':'primitives.sort((a,b)=>b.z-a.z)' in s}
assert all(checks.values()),checks
(D/'audits/display_structure.json').write_text(json.dumps(checks,indent=2));print('page_written')
