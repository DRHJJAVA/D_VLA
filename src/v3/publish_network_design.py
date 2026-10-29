from pathlib import Path
import markdown,json,html,datetime
R=Path("/mnt/nvme1/drh/D_VLA");d=R/"docs"
name="完整示范抓取区域学习_网络设计与验证路线"
md=(d/(name+".md")).read_text()
report=R/"audits/v3/full_sequence_input_check.json"
if report.exists():
 r=json.loads(report.read_text())
 note="\n### 本轮实际检查结果\n\n已逐段核对 "+str(r["episodes"])+" 段原始示范长度，共 "+str(r["synchronized_time_steps"])+" 个同步时刻、"+str(r["camera_frames"])+" 帧相机观测；全部深度帧数值有限，连续分块没有遗漏原始帧。此脚本未重新验证 RGB 语义和视觉对齐，未训练模型。\n"
 if "### 本轮实际检查结果" not in md:md+=note;(d/(name+".md")).write_text(md)
renderer=markdown.Markdown(extensions=["tables","toc","fenced_code"])
body=renderer.convert(md)
# Keep the architecture and method choice visible; long reference/implementation sections can be expanded.
import re
for heading in ["2. 从哪些已有工作借鉴","4. 网络怎样设计","5. 损失函数中的信号来自哪里"]:
 pat=r'(<h2[^>]*>'+re.escape(heading)+r'</h2>)(.*?)(?=<h2|\Z)'
 match=re.search(pat,body,flags=re.S)
 if match:
  sid=re.search(r'id="([^"]+)"',match.group(1)).group(1)
  body=body[:match.start()]+'<details id="'+sid+'"><summary>'+heading+'</summary>'+match.group(2)+'</details>'+body[match.end():]
css="""
:root{color-scheme:light;--ink:#213a3d;--muted:#5c7174;--line:#d8e1df;--accent:#226958}
*{box-sizing:border-box}body{margin:0;background:#f4f7f5;color:var(--ink);font:17px/1.85 "Noto Sans CJK SC","Microsoft YaHei",sans-serif}
header{padding:18px 5vw;border-bottom:1px solid var(--line);background:white;display:flex;justify-content:space-between}
header strong{font-size:15px;letter-spacing:.06em}header span{font-size:14px;color:var(--muted)}
main{max-width:1320px;margin:24px auto;background:white;padding:42px 52px;box-shadow:0 8px 30px #19352c08}
h1{font-size:34px;line-height:1.35;margin:0 0 14px}h2{font-size:25px;margin-top:48px;border-top:1px solid var(--line);padding-top:24px}h3{font-size:20px;margin-top:28px}
a{color:var(--accent);text-underline-offset:3px}p{max-width:1100px}img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:10px;margin:25px 0}
table{width:100%;border-collapse:collapse;font-size:15px;line-height:1.7;margin:24px 0}
th{text-align:left;background:#edf4ef}td,th{border:1px solid var(--line);padding:12px 15px;vertical-align:top}
li{margin:7px 0}details{margin:24px 0;border:1px solid var(--line);border-radius:8px;padding:20px}
summary{cursor:pointer;font-weight:700;font-size:21px}.banner{border-left:4px solid var(--accent);padding:12px 18px;background:#edf5f0;margin-bottom:30px}
footer{font-size:14px;color:var(--muted);padding:25px 0 0;border-top:1px solid var(--line)}
@media(max-width:800px){main{padding:24px 18px;margin:0}h1{font-size:26px}table{font-size:13px}td,th{padding:8px}}
"""
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整示范抓取区域学习：网络设计</title><style>'+css+'</style><header><strong>D_VLA · 网络设计与方法选择</strong><span>远程文档 · 2026-10-23</span></header><main><div class="banner">官方序列原样保留；一段录像可含多次、不同类型或不完整的操作。第一步只读检查，不补动作。</div>'+body+'<footer>设计图、源文件、数据与报告均保存在远程服务器。当前图是候选网络结构，不是训练结果。</footer></main></html>'
(d/(name+".html")).write_text(page)
oldname="D_VLA抓取区域学习实验方案（分阶段验收版）"
history=d/"历史版本";history.mkdir(exist_ok=True)
for ext in [".md",".html"]:
 old=d/(oldname+ext);backup=history/("抓取区域学习方案_v3.0_已停用"+ext)
 if old.exists() and not backup.exists():backup.write_bytes(old.read_bytes())
(d/(oldname+".md")).write_text("# 旧方案已停用\n\n请阅读 [完整示范抓取区域学习：网络设计与验证路线]("+name+".md)。\n\n用户已明确完整 RGB-D 序列参与训练，取消旧红蓝区域图及逐点答案验收。旧 v3.0 仅保留历史记录，不再执行。\n")
(d/(oldname+".html")).write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url='+name+'.html"><title>方案已更新</title><a href="'+name+'.html">打开完整示范抓取区域学习网络设计</a></html>')
status=d/"STATUS.md"
tag="## 2026-10-23：完整示范学习设计修订"
s=status.read_text()
if tag not in s:
 s+='\n\n'+tag+'\n\n完整 RGB-D 示范是训练输入，包含抓取前至放置松开后；无机器人动作预测。旧红蓝区域图和人工区域验收取消，旧 v3.0 方案停止执行并存入历史版本。本轮交付网络图与方法比较设计，优先测试自监督特征与视觉交互学习，是否加自动关系约束由验证决定。完整序列数据通路检查与训练分开；尚未开展本设计训练。仍在第一阶段，未经阶段验收不进入第二阶段。旧数值指标不代表此方案效果。\n\n新文档：'+name+'.md / .html。图：图示/完整示范抓取区域学习_网络设计图.png。\n'
 status.write_text(s)
print(json.dumps({"html":str(d/(name+".html")),"previous_plan":"archived_and_superseded","training_started":False},ensure_ascii=False))
