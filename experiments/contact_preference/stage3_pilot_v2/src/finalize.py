from pathlib import Path
import json,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_preference/stage3_pilot_v2')
B=R.parents[2]
D=B/'docs/夹爪物体接触偏好学习_完整方案.md'
page=B/'docs/接触偏好_阶段3_夹爪专属正负支持对照.html'
now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((R/'audits/completion.json').read_text())
browser={'time':now,'all_12_endpoints_images_loaded':True,'all_12_display_counts_match':True,'visual_inspection':['porcelain55 cloud/selected source','tomato68 raw views/green projection','whiteyellow71 raw views/classified cloud'],'overlay_toggle_checked':True,'unknown_gray':True,'positive_saturated_green':True,'mount_exclusion_separate':True,'note':'Visual location review is not physical contact truth; green projections may be occluded by robot pixels.'}
(R/'audits/browser_review.json').write_text(json.dumps(browser,ensure_ascii=False,indent=2))
hashes={p.name:sha(p) for p in (R/'src').glob('*.py')}
entry=f"""

## {now}｜阶段3第二次小试结果：局部接触证据与夹爪范围分开处理

本轮仅对已有瓷杯 demo9、番茄酱 demo7、白黄杯 demo4 的12个固定端点生成弱支持；没有新增训练集、没有训练评分网络、没有读取预留测试或使用人工区间作监督。旧版保留，所有新产物在远程 experiments/contact_preference/stage3_pilot_v2。当前页面：[阶段3夹爪专属正负支持对照](接触偏好_阶段3_夹爪专属正负支持对照.html)。

### 根因与实际变化

旧版三类分类确实没有正支持占优的绿色点，连续正支持又被先验与尺度构造限制在约0.115以内，不能只解释成颜色太浅。本轮先尝试学习全物体混合先验，但36个拟合的接触比例退化到约6.14e-6～7.45e-6，几乎全判分离；这个失败已保存在 summary.json、fitted_models.json、points/，没有用它当最终结果。

修订后采用此前登记的局部条件似然证据：接触解释 fC 与正间隙分离解释 fS 分别生成密度，正支持为 max(fC-fS,0)/(fC+fS)，分离支持对称，重叠部分归未知。保留两种分离分布、两种误差模型、按其他事件重采样拟合的敏感性；所有模型均排除当前被评事件。跨帧不按独立重复缩小误差，当前帧与事件聚合取保守共同支持。这样把“整个物体大部分表面不接触”的比例从逐点局部证据中分离出来。**这是监督量定义的更改，不是校准概率修复，也不能把正支持变大当作准确率提高。** 条件分布的相对归一化和模型族仍是设计选择，没有宣称完全无假设。

只允许目标物体来源点与左右夹指/指垫（已知机器人几何3/4/6/7）产生正支持，含夹指外侧，未限定为两指之间。安装壳体1不参与接触表面；它比夹指更近时，点独立标记“范围排除”，正、负支持均0，不作为其他接触的未知候选。两尺度表面法向必须相向才保留正支持，不相向并不自动判负。过深穿透作为几何冲突未知。没有加入抬升、动作类别或放宽接触距离阈值。

这里的“范围排除”是保守处理，不能解读为已证明发生壳体接触；其他物体/桌面也没有用作接触对象。物体来源仍继承自动前景与跨帧对应，尚未独立证明所有来源身份正确。

### 同一批案例结果

| 案例 | 端点 | 原绿色点/帧 | 新绿色点/帧 | 分离支持占优/帧 | 未知占优/帧 | 安装壳体最近而排除/帧 |
|---|---|---:|---:|---:|---:|---:|
| 瓷杯 demo9 | 55/57/60/65/66 | 0 | 28–40 | 1367–1383 | 54–66 | 1266–1278 |
| 番茄酱 demo7 | 68/70/73/78 | 0 | 80–89 | 685–689 | 71–80 | 335–339 |
| 白黄杯 demo4 | 66/68/71 | 0 | 11–13 | 2126–2143 | 44–47 | 1520–1537 |

合计29602个点时刻观测：正支持占优546、分离支持占优16030、未知736、范围排除12290、深穿透冲突0。跨帧同一个来源会重复计数，不能当独立接触数、真实面积或成功率。绿色中安装壳体最近或法向不符的点均为0。

瓷杯55帧旧未知中有109点最近的是安装壳体，现不再把它们画作待定接触。新绿色34点中8点当前深度一致、26点当前遮挡或不一致；后者来自夹前真实表面映射，不是用当前机器人像素作为物体表面。遮挡/不一致还可能包含映射误差，不能全部认定为真实遮挡接触。

### 可视化修正

默认只突出固定饱和绿色正支持占优点，其余灰色；可切换红色分离与灰色未知的分类视图。范围排除单独计数，明确不是接触候选。点选可查夹前像素、最近夹指部件、几何间距、误差尺度、法向与可见性。当前图上的绿色是来源点投影，可能被夹爪遮挡；提供关闭投影查看原图的开关，不用投影颜色证明像素属于物体。12个端点的36张显示图片加载与计数均检查通过，实际复用旧27张不同图片，没有重复生成原图。

### 验收与未解决项

工程核查通过：12端点源XYZ/像素/RGB/变换后位置与旧版逐数组相等；旧版85文件完成锁、本轮输入与源码锁无差异。独立重算支持差0，网页舍入最大差4.998e-7；用数值积分核对两种条件密度，最大相对误差1.53e-15；全部36个模型排除了当前案例。修正了两个继承的可见绿色计数字段仍为0的问题，现等于逐点独立计数，支持数组没改。详见 audits/completion.json 与 evidence_summary.json 的更正记录。

**物理接触可靠性尚未验收通过。** 合成分离检查中3倍误差尺度和10倍误差尺度的分离点正支持为0；但仅1倍误差尺度的真实正间隙仍可得到约0.496–0.627正支持，两组甚至正支持占优。当前误差尺度约2.63–3.07mm，无法据此区分毫米级靠近与真正接触。绿色代表相对支持接触解释，不能称确认接触或用于宣称精度提高。这是本轮遗留的核心可辨识性问题，不能靠降低阈值或单纯加深绿色解决。

本轮结论：完成正支持数值构造、夹爪专属范围及显示的小规模修订；不宣布监督已足够可靠，不进入评分网络训练。下一步若继续，应先针对同一来源的接近—保持—离开变化提供额外时序区分证据，并用可见分离反例检验误报；不能继续仅用绝对间距自证接触。本轮未运行这项后续工作。

源码SHA256：
"""
entry+='\n'.join('- '+k+'：'+v for k,v in hashes.items())+'\n'
with D.open('a',encoding='utf-8') as f:f.write(entry)
state={'stage':'complete_pilot_likelihood_support','time':now,'scoring_network_trained':False,'engineering_audit':'passed','physical_contact_reliability':'not_accepted','open_issue':'sub_noise_separation_not_identifiable_from_current_evidence'}
(R/'stage_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
final={'time':now,'same_document':str(D),'document_sha256_at_completion':sha(D),'page':str(page),'page_sha256':sha(page),'source_hashes':hashes,'completion_audit_sha256':sha(R/'audits/completion.json'),'browser_review_sha256':sha(R/'audits/browser_review.json'),'no_scoring_training':True}
(R/'audits/final_integrity.json').write_text(json.dumps(final,ensure_ascii=False,indent=2))
files=[p for p in R.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix not in ['.log','.pyc'] and '__pycache__' not in str(p) and p.name!='completed_locked.json']
lock={str(p):sha(p) for p in files};lock[str(page)]=sha(page)
(R/'completed_locked.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2))
assert all(sha(Path(p))==h for p,h in lock.items())
print(json.dumps({'time':now,'files_locked':len(lock),'document_updated':D.is_file(),'page_ready':page.is_file(),'state':state},ensure_ascii=False))
