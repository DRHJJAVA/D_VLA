from pathlib import Path
import sys,json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties,fontManager
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"design_qa"))
from audit_panel_alignment import require_matplotlib_panel_alignment
ROOT=Path("/mnt/nvme1/drh/D_VLA")
OUT=ROOT/"docs"/"图示"
OUT.mkdir(parents=True,exist_ok=True)
fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Noto Sans CJK JP","Arial","DejaVu Sans"],"font.size":16,"pdf.fonttype":42,"svg.fonttype":"none","axes.unicode_minus":False})
fig,ax=plt.subplots(figsize=(19.2,15.0))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.set(xlim=(0,1920),ylim=(1500,0))
ax.axis("off")
C={"ink":"#203337","muted":"#52676b","line":"#b9c8c8","green":"#e5f1ec","purple":"#eeebf5","gold":"#fbf0d8","white":"#ffffff","gray":"#f4f7f7"}
def txt(x,y,s,size=16,weight="normal",color=None,ha="left"):
 return ax.text(x,y,s,fontsize=size,fontweight=weight,color=color or C["ink"],ha=ha,va="center",linespacing=1.65)
def box(x,y,w,h,title,body="",fill="gray",ts=18,bs=14):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=13",facecolor=C[fill],edgecolor=C["line"],linewidth=1))
 txt(x+22,y+32,title,ts,"bold")
 if body:txt(x+22,y+h/2+20,body,bs,color=C["muted"])
def arrow(a,b,label=None,dashed=False,col=None):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=17,linewidth=1.7,color=col or C["muted"],linestyle="--" if dashed else "-"))
 if label:txt((a[0]+b[0])/2,(a[1]+b[1])/2-15,label,13,ha="center")
txt(60,54,"完整示范 → 物体抓取区域评分",29,"bold")
txt(60,99,"候选网络设计  ·  训练读取完整 RGB-D 序列  ·  推理只看物体  ·  方法由对照实验决定",17,color=C["muted"])
# Full episode input, all phases remain.
box(50,145,1820,170,"01  官方录像原样读取；一段录像不等于一次交互",fill="gray",ts=20)
stages=["原始开始帧","操作或静止","可有多次交互","动作类型可不同","原始结束帧"]
for i,s in enumerate(stages):
 x=75+i*355
 box(x,209,325,66,s,fill="white",ts=17)
 if i<4:arrow((x+327,242),(x+350,242))
txt(75,294,"保留提供的全部帧；不强制动作起止或固定流程，不补帧、不改官方数据；深度来源单独记录。",14)
arrow((960,315),(960,345))
# Shared encoders.
box(50,350,1190,135,"02A  RGB-D 双路编码","RGB：DINOv2 ViT-S/14 自监督预训练特征，先冻结主干\n深度：反投影 → 可见点云 → PointNet++；保留真实尺度与视觉表面对应",fill="green")
box(1310,350,560,135,"02B  外观与几何融合","图像特征投到点云，融合为 256 维\n局部点特征保留，每帧压缩为 32 个 token",fill="green")
arrow((1240,418),(1304,418))
arrow((1590,485),(1590,525))
# Sequence learner row direction right to left clean.
box(1310,530,560,170,"03  全序列教师分支","4 层时序 Transformer，隐藏维度 256\n读取所有提供的帧与时间顺序\n学习跨帧对应、遮挡信息和交互变化",fill="purple")
box(680,530,560,170,"04  局部交互学习","遮挡夹爪提示，匹配真实局部上下文\n识别物体部位，排除夹爪与背景捷径\n抓取关系仅用于抓取子实验，做对照",fill="purple")
box(50,530,560,170,"05  时间级证据与候选交互","一段录像可有多次交互；边界待学习\n可靠事件分别聚合；不强制固定阶段\n看不到、对应不可靠的表面记为未知",fill="purple")
arrow((1310,615),(1246,615))
arrow((680,615),(616,615))
# Learning signals on one band.
box(50,740,1820,120,"训练信号来自哪里？",fill="gold",ts=18)
txt(75,801,"自监督：遮挡特征预测 + 跨视角对应    ｜    交互学习：局部上下文匹配    ｜    可选自动约束：夹指与物体的粗关系",15,color=C["muted"])
txt(75,835,"无人工逐点标签；不输入机器人动作；不同操作不能都解释成抓取概率，事件发现须验证。",14)
# Student input and output.
txt(60,901,"06  把序列中的知识传给只看物体的学生分支",20,"bold")
box(50,945,400,145,"物体 RGB-D","训练：同一示范中无遮挡观测\n测试：已见物体 / 未见同类物体",fill="green",ts=18,bs=14)
box(520,945,400,145,"物体编码器","复用图像与点云特征结构\n输入中排除夹爪和背景捷径",fill="green",ts=18,bs=14)
box(990,945,400,145,"逐点评分头","局部点特征 + 多模式原型\n同一物体允许多个高分区域",fill="green",ts=18,bs=14)
box(1460,945,410,145,"输出","可见物体点云的区域评分\n同时给出可见性与证据置信度",fill="green",ts=18,bs=14)
for x1,x2 in [(450,520),(920,990),(1390,1460)]:arrow((x1,1018),(x2-5,1018))
# Dashed transfer path outside content, down left to band.
ax.plot([35,35,960,960],[700,917,917,1018],color=C["muted"],linestyle="--",linewidth=1.5)
arrow((960,1018),(984,1018),dashed=True)
txt(620,934,"教师验证后，冻结目标并蒸馏",13,ha="center")
arrow((330,700),(35,700),dashed=True)
# Experiment design card.
box(50,1130,1820,170,"用同一数据划分选择方法",fill="gray",ts=19)
txt(77,1200,"A  自监督特征 + 局部上下文学习",17,"bold")
txt(700,1200,"B  A + 自动局部交互约束",17,"bold")
txt(1305,1200,"C  B + 完整时序 / 多模式聚合",17,"bold")
txt(77,1255,"固定点数预算比较热点命中与包裹覆盖；整段示范、物体资产独立划分；测试时没有夹爪提示。",16,color=C["muted"])
txt(60,1352,"关键待验证：是否覆盖包裹表面、是否迁移到新尺寸和形状；共运动与注意力本身都不能证明抓取区域正确。",16)
txt(60,1403,"当前建议：自监督为底座的示范观察学习。强化学习留给未来“主动试抓、优化成功率”的另一个问题。",16)
txt(60,1460,"设计示意，不是模型结果  |  依据：Hands as Probes · Interaction Hotspots · LOCATE · PointContrast · VideoMAE / V-JEPA 2",12,color=C["muted"])
base=OUT/"完整示范抓取区域学习_网络设计图"
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=str(base)+".alignment.json",strict=True)
fig.savefig(str(base)+".png",dpi=300)
fig.savefig(str(base)+".svg")
fig.savefig(str(base)+".pdf")
plt.close(fig)
print(json.dumps({"figure":str(base),"status":"design_only","backend":"matplotlib","archetype":"single_canvas_schematic","quantitative_results":False},ensure_ascii=False))
