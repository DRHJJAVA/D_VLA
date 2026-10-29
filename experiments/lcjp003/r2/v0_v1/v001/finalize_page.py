from pathlib import Path
import json,runpy
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/lcjp003/r2/v0_v1/v001'
runpy.run_path(str(P/'render_results.py'),run_name='__main__')
f=R/'docs/LCJP003_R2_V0_V1检查.html';s=f.read_text()
s=s.replace('V1 需要注意：','V1 发现输入问题，尚未通过：')
s=s.replace('候选中也可能存在桌面、机械臂碎片及重复区域。','已有明确例子：milk / demo45 / 第0帧的牛奶盒可见，却没有独立候选。候选中还存在桌面和机器人轮廓附近的残余碎片。')
s=s.replace('边界框及掩膜来自已有分割缓存；#编号是候选 ID。','实际掩膜轮廓来自旧分割缓存；#编号是旧候选 ID，跨视角身份尚未独立验证，不能直接作为R2合并依据。图上数字是该视角先验；表中是旧ID各视角均值，仅作诊断。')
s=s.replace('核心接口检查只证明能传递数据和梯度；','V0几何已改用P0修正后的深度反投影，保留单视角候选分支；两相机回算最大差5.96e-8米。t=0/3/15/20观测读取守卫均未访问未来。核心接口检查只证明能传递数据和梯度；')
panel='''<section><h2>漏候选定位 · 只改变提示点密度</h2><p>同一个 SAM2.1 tiny、相同质量阈值，8×8 与 32×32 提示点对比；新旧掩膜都保留，未更新模型权重。</p><p><label>样本 <select id="denCase"><option value="60">milk · demo45 · 第0帧</option><option value="10">bbq sauce · demo45 · 第0帧</option></select></label><label>视角 <select id="denView"><option value="0">外部相机</option><option value="1">腕部相机</option></select></label></p><div class="views"><div><div class="viewtitle">8×8 提示点</div><img id="den8"></div><div><div class="viewtitle">32×32 提示点</div><img id="den32"></div></div><p id="denMeta" class="muted"></p><p>新增轮廓不等于语义已经识别正确；这里只排查可见物体有没有进入候选。密度对照仅覆盖2帧×2相机，不能代替100帧修复后的覆盖验收。</p></section>'''
s=s.replace('<section><h2>先验是否真的看区域？</h2>',panel+'<section><h2>先验是否真的看区域？</h2>')
js="""let DD;
async function density(){DD=await(await fetch(root+'density_probe.json')).json();$('denCase').onchange=denShow;$('denView').onchange=denShow;denShow();}
function denShow(){let cc=+$('denCase').value,vv=+$('denView').value,rs=DD.records.filter(r=>r.case===cc&&r.camera===vv);for(let r of rs)$('den'+r.grid).src=root+r.image;$('denMeta').textContent=rs.map(r=>r.grid+'×'+r.grid+'：'+r.nonempty_after_robot_mask+'个候选，'+r.seconds.toFixed(3)+'秒').join('；')+'。含冷启动差异，不作为公平吞吐基准。'}
density();
"""
s=s.replace("init().catch",js+"\ninit().catch")
f.write_text(s)
print(json.dumps({'page_updated':True,'v0_current_result_loaded':True,'density_panel':True}))
