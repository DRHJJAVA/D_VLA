
from pathlib import Path
import sys,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib import font_manager
ROOT=Path('/mnt/nvme1/drh/D_VLA')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'docs/LCJP003_R4_design_assets';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(HERE/'qa_tools'))
from audit_panel_alignment import require_matplotlib_panel_alignment
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
C={'visual':'#e8eff9','score':'#e0f1ed','action':'#fff0dc','train':'#efe8f5','neutral':'#f1f3f6'}
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
def canvas(title,sub):
 f,a=plt.subplots(figsize=(18,10));f.subplots_adjust(left=.025,right=.975,bottom=.035,top=.97)
 a.set(xlim=(0,18),ylim=(0,10));a.axis('off')
 a.text(.1,9.55,title,fontsize=22,weight='bold',color='#20334c')
 a.text(.1,9.05,sub,fontsize=11,color='#526276')
 return f,a
def box(a,x,y,w,h,title,body='',kind='neutral'):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=.10',facecolor=C[kind],edgecolor='#8291a5',lw=1))
 a.text(x+w/2,y+h-.29,title,ha='center',va='center',fontsize=12,weight='bold',color='#20334c')
 if body:a.text(x+w/2,y+(h-.48)/2,body,ha='center',va='center',fontsize=10,linespacing=1.65,color='#263b4e')
def arrow(a,x1,y1,x2,y2,dash=False):
 a.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,lw=1.4,color='#8c6bac' if dash else '#405e78',linestyle='--' if dash else '-'))
def export(f,name):
 f.canvas.draw()
 require_matplotlib_panel_alignment(f,json_out=str(OUT/(name+'.alignment.json')))
 f.savefig(str(OUT/name)+'.png',dpi=300,facecolor='white')
 f.savefig(str(OUT/name)+'.svg',facecolor='white')
 f.savefig(str(OUT/name)+'.pdf',facecolor='white')
 plt.close(f)
f,a=canvas('LCJP003 · R4    直接识别目标 → 三维部位评分 → 动作','网络总图  /  LCJP003-R4-design-v1  /  实线：推理路径     虚线：训练监督')
nodes=[
(.15,6.1,2.25,2.2,'双视角 RGB-D','外部相机＋腕部相机\n当前图像与深度\n任务指令 / 机器人历史','neutral'),
(2.9,6.1,2.25,2.2,'共享视觉特征','DINO 384 → 256\n两路各编码一次\n特征供分割与点云复用','visual'),
(5.65,6.1,2.4,2.2,'直接图文分割','4 × Cross-Attention\n256维 / 8头\n上采样 256 → 128 → 64','visual'),
(8.55,6.1,2.4,2.2,'目标三维表面','预测掩膜 × 当前深度\n双视角可见性融合\n2048真实点 × 268维','visual'),
(11.45,6.1,2.5,2.2,'状态条件部位评分','64局部组 × 32邻点\n独立组分数＋点内细化\n夹爪位置 / 朝向 / 开口','score'),
(14.45,6.1,3.35,2.2,'部位门控动作专家','8个查询读取64组\n可学习 λ 融合 → 256→720\n16层专家＋LoRA → 16步动作','action')]
for args in nodes:box(a,*args)
for left,right in zip(nodes,nodes[1:]):arrow(a,left[0]+left[2]+.04,7.2,right[0]-.07,7.2)
box(a,2.9,3.85,2.25,1.45,'指令编码','文字 960 → 256\n目标 / 放置位置查询','visual')
arrow(a,5.2,4.6,6.3,6.04)
box(a,5.65,3.85,2.4,1.45,'明确的掩膜输出','目标＋放置位置\n各视角存在 / 可见性','visual')
arrow(a,6.85,6.05,6.85,5.35)
box(a,8.55,3.85,2.4,1.45,'状态与两指几何','16×27历史 → 256\n两指64×6 → 256','neutral')
arrow(a,11,4.6,12.1,6.04)
box(a,11.45,3.85,2.5,1.45,'共享粗场景上下文','放置位置 / 障碍关系\n不重复编码密集目标几何','neutral')
arrow(a,14,4.6,15.1,6.04)
box(a,14.45,3.85,3.35,1.45,'执行与下一帧','运动6维＋开合2类\n剩余计划返回下一次预测','action')
arrow(a,16.1,6.05,16.1,5.35)
a.text(.15,2.95,'仅训练阶段：离线候选与示范监督',fontsize=14,weight='bold',color='#77548d')
box(a,.15,.85,4.0,1.55,'监督来源','SAM 32×32候选 / 示范轨迹\n语义＋运动＋跨视角绑定；冲突忽略','train')
box(a,4.75,.85,3.65,1.55,'目标掩膜监督','BCE＋Dice / 存在性\n不是模型自身选择的循环标签','train')
box(a,9,.85,3.65,1.55,'部位支持监督','有效交互范围 / 未知忽略\n替换固定25mm全物体扩散','train')
box(a,13.25,.85,4.55,1.55,'动作与时序监督','Drifting＋开合分类＋切换辅助\n相邻计划重叠一致性','train')
arrow(a,4.2,1.6,4.7,1.6,True)
arrow(a,6.7,2.45,6.7,3.8,True)
a.plot([11.2,11.2,12.7],[2.45,5.65,5.65],color='#8c6bac',lw=1.4,ls='--')
arrow(a,12.7,5.65,12.7,6.05,True)
arrow(a,16.5,2.45,16.5,3.8,True)
a.text(.15,.25,'推理不执行密集候选搜索；未来示范只参与训练监督。R4为结构迭代，图中不含实验结果。',fontsize=10,color='#526276')
export(f,'r4_network')
f,a=canvas('R4 · 从“少数高分点”到“部位分数真正影响动作”','评分与动作接口细节  /  独立分数 ≠ 注意力权重  /  所有64个局部组参与读取')
box(a,.2,6.1,3.5,2.1,'几何局部组','2048目标点 → 64局部组\n每组最多32邻点\n距离与表面边界有效性掩码','visual')
box(a,4.4,6.1,3.7,2.1,'组评分头','组特征＋指令＋夹爪状态\nMLP 512→256→128→1\n独立分数 s = sigmoid(u)','score')
box(a,8.8,6.1,3.7,2.1,'点内细化','点残差 r：256→128→1\nlogit = Σ w·u + 0.25 tanh(r)\n邻近表面内插值 → 点分数','score')
box(a,13.2,6.1,4.6,2.1,'目标表面的评分图','固定0–1分段颜色\n允许整块部位同时高分\n不再强制只有最高5%显示红色','score')
for x1,x2 in [(3.75,4.35),(8.15,8.75),(12.55,13.15)]:arrow(a,x1,7.15,x2,7.15)
box(a,.2,2.5,3.5,2.1,'特征先归一化','v = Wv(LayerNorm(h))\n保留全部64组\n取消Top-8硬选组','neutral')
box(a,4.4,2.5,3.7,2.1,'分数显式门控','Fpart = Σ a · s · v\na：动作查询注意力\ns：独立部位分数','score')
box(a,8.8,2.5,3.7,2.1,'可学习 λ 融合','F = (1−λ)Fbase + λFpart\n基础分支：粗场景与状态\n门控后不单独LayerNorm','action')
box(a,13.2,2.5,4.6,2.1,'动作输出与直接对照','16层 / 720维动作专家\n运动头＋开合头＋切换辅助头\n正常 / 均匀 / 打乱 / 错误部位','action')
for x1,x2 in [(3.75,4.35),(8.15,8.75),(12.55,13.15)]:arrow(a,x1,3.55,x2,3.55)
arrow(a,1.95,6.05,1.95,4.65);arrow(a,6.25,6.05,6.25,4.65)
a.text(.2,1.55,'监督：独立部位支持＋可信表面局部一致性；未知点不强行当负样本，不能通过平滑把整物体抹成高分。',fontsize=12,color='#526276')
a.text(.2,.85,'验收：正确评分必须改善动作误差或闭环成功率；梯度非零、λ增大、热图更红，都不能单独证明有效。',fontsize=12,color='#526276')
export(f,'r4_score_action')
(OUT/'figure_contract.json').write_text(json.dumps({'version':'R4-design-v1','backend':'python','type':'network schematic','claim':'direct segmentation and region-score gated action architecture','quantitative_results':False,'alignment':'each figure is a single schematic axes; no comparable quantitative panels','exports':['png','svg','pdf']},ensure_ascii=False,indent=2))
print(json.dumps({'figures':['r4_network','r4_score_action']}))
