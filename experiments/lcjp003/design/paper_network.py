
import sys,json,datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle,Circle,Polygon
from matplotlib.font_manager import FontProperties
root=Path('/mnt/nvme1/drh/D_VLA')
out=root/'docs/LCJP003_layer_blueprint_assets'
sys.path.insert(0,str(root/'experiments/lcjp003/design/figure_qa'))
from audit_panel_alignment import require_matplotlib_panel_alignment
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.size':10,'font.family':'sans-serif','font.sans-serif':['DejaVu Sans']})
fig,ax=plt.subplots(figsize=(23,11.5))
fig.patch.set_facecolor('white');ax.set_xlim(0,24);ax.set_ylim(0,12);ax.axis('off')
ink='#243449';blue='#3573a1';teal='#288c87';orange='#ce8545';purple='#8371ae';gray='#788795'
def text(x,y,t,sz=12,c=ink,ha='center',weight='normal'):
 return ax.text(x,y,t,fontproperties=font,fontsize=sz,color=c,ha=ha,va='center',weight=weight,linespacing=1.5,zorder=5)
def arrow(x1,y1,x2,y2,c=ink,dashed=False,rad=0):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,lw=1.5,color=c,linestyle='--' if dashed else '-',connectionstyle='arc3,rad='+str(rad),zorder=2))
def box(x,y,w,h,t,fill='#edf4f9',edge=blue,sz=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02,rounding_size=.08',facecolor=fill,edgecolor=edge,lw=1.0,zorder=3))
 text(x+w/2,y+h/2,t,sz,edge)
def stack(x,y,n=4,w=.15,h=.8,c=blue):
 for i in range(n):ax.add_patch(Rectangle((x+i*(w+.055),y),w,h,facecolor=c,alpha=.3+i*.1,edgecolor=c,lw=.7))
def cloud(cx,cy,scale=.65,color=teal,heat=False,form='bottle'):
 # Synthetic geometry used solely as an explanatory icon, not an experiment result.
 z=np.linspace(-1,1,17); ang=np.linspace(0,2*np.pi,17,endpoint=False);z,a=np.meshgrid(z,ang)
 rad=np.where(z>.45,.30,.52) if form=='bottle' else np.full_like(z,.52)
 xx=cx+scale*(rad*np.cos(a)+.24*rad*np.sin(a)); yy=cy+scale*(z+.16*rad*np.sin(a))
 if heat:
  val=np.exp(-((z-.55)/.44)**2);cc=plt.cm.viridis(val.ravel())
 else:cc=color
 ax.scatter(xx.ravel(),yy.ravel(),s=8,c=cc,alpha=.88,linewidths=0,zorder=4)
def scene(x,y,flip=False):
 ax.add_patch(Rectangle((x,y),1.65,1.15,facecolor='#f1f1ec',edgecolor='#bcc7ce',lw=.8))
 ax.add_patch(Polygon([(x+.08,y+.12),(x+1.53,y+.12),(x+1.3,y+.75),(x+.25,y+.75)],facecolor='#e2e4dc',edgecolor='none'))
 ax.add_patch(Rectangle((x+.35,y+.28),.2,.35,facecolor='#70a4a0',edgecolor='none'))
 ax.add_patch(Rectangle((x+1.04,y+.25),.25,.21,facecolor='#ba9aca',edgecolor='none'))
 ax.add_patch(Rectangle((x+.77,y+.63),.1,.40,facecolor='#84939e',edgecolor='none'))
 for dx in [.64,.94]:ax.add_patch(Rectangle((x+dx,y+.58),.08,.18,facecolor='#84939e',edgecolor='none'))
def group(x,w,title,num):
 ax.add_patch(FancyBboxPatch((x,4.85),w,5.65,boxstyle='round,pad=.04,rounding_size=.12',facecolor='#fbfcfe',edgecolor='#c3cdd6',lw=.85,linestyle=(0,(4,4)),zorder=0))
 text(x+.1,10.95,num+'  '+title,15,ha='left',weight='bold')
text(.4,11.72,'状态条件三维部位评分与动作联合学习',23,ha='left',weight='bold')
text(23.6,11.72,'LCJP003 · 修订结构',12,c=gray,ha='right')
group(.4,4.4,'多视角与机器人状态','①')
group(5.15,5.15,'语义目标选择','②')
group(10.65,5.6,'目标内部位评分','③')
group(16.6,7.0,'特征融合与动作生成','④')
scene(.65,8.5);scene(.65,6.85,True)
text(1.45,8.27,'外部 RGB-D',10,c=gray);text(1.45,6.62,'腕部 RGB-D',10,c=gray)
box(2.8,8.6,1.65,.85,'冻结视觉编码\nDINO / Smol',sz=11)
box(2.8,6.95,1.65,.85,'分割 + 3D关联\n排除机器人',sz=11)
arrow(2.35,9.03,2.77,9.03);arrow(2.35,7.43,2.77,7.43)
text(2.5,5.65,'夹爪几何 + 16帧状态历史',11,c=gray)
text(2.5,6.02,'指令：“拿起烧烤酱”',12,c=blue)
ax.plot([4.1,4.62,4.62,8.15],[6.02,6.02,10.14,10.14],color=blue,lw=1.2)
arrow(8.15,10.14,8.15,9.48,c=blue)
text(6.45,10.32,'指令特征',9,c=blue)
# Candidate object tokens
cloud(5.82,8.95,.47,teal);cloud(6.63,8.95,.47,purple,form='box')
stack(5.65,7.10,5,.12,.65,blue)
text(6.1,6.63,'K个统一物体身份\n每物体256维',10,c=gray)
arrow(4.45,9.02,5.35,9.02)
arrow(4.45,7.40,5.35,7.40)
box(7.38,8.18,1.55,1.25,'文字 / 物体\n交叉注意力\n256维 · 4头',sz=11)
box(7.38,6.53,1.55,1.15,'历史 / 物体\n交叉注意力\n256维 · 4头',sz=11)
arrow(6.82,8.80,7.34,8.80);arrow(8.15,8.13,8.15,7.73)
for j,(length,col) in enumerate([(.28,gray),(.68,teal),(.19,gray)]):
 ax.add_patch(Rectangle((9.34,8.82-j*.5),length,.24,facecolor=col,edgecolor='none',zorder=3))
text(9.65,9.45,'目标分布',10,c=gray)
arrow(8.97,7.1,9.65,7.1);arrow(9.65,7.1,9.65,7.7)
text(7.7,5.55,'选择对象 / 目的地两个角色\n训练软选择；推理明确目标身份',11,c=blue)
# Target -> point scorer -> score field
cloud(11.4,8.35,.69,teal)
text(11.48,9.5,'目标物体',11,c=teal)
arrow(10.1,8.55,10.95,8.55)
box(12.45,8.10,1.7,1.35,'夹爪条件注意力\n+ 历史、文字\n3个注意力块',fill='#e5f2f0',edge=teal,sz=11)
arrow(11.9,8.55,12.41,8.55)
box(12.45,6.50,1.7,1.0,'点级评分解码\n512→256→128→1',fill='#e5f2f0',edge=teal,sz=10)
arrow(13.3,8.04,13.3,7.55)
cloud(15.15,8.35,.69,heat=True)
text(15.12,9.5,'部位分布 α',11,c=teal)
arrow(14.18,7.0,14.55,7.0);arrow(14.55,7.0,14.98,7.65)
text(15.12,6.48,'2048真实点\n位置 + 形状 + 特征',10,c=gray)
text(13.45,5.55,'只在所选目标表面评分\n同一3D分数投影回两个相机',11,c=teal)
# score and base branches
box(17.0,8.68,1.75,.8,'基础特征\n8×256',fill='#edf4f9',edge=blue)
box(17.0,6.93,1.75,.8,'评分部位特征\n8×256',fill='#e5f2f0',edge=teal)
arrow(16.0,8.55,16.65,8.55);arrow(16.65,8.55,16.65,7.32);arrow(16.65,7.32,16.95,7.32)
# Independent base branch from selected target, routed above content.
arrow(11.45,9.9,17.85,9.9,c=blue);arrow(17.85,9.9,17.85,9.51,c=blue)
text(14.7,10.16,'目标基础特征旁路',9,c=blue)
ax.add_patch(Circle((19.45,8.22),.40,facecolor='#fff0d9',edgecolor=orange,lw=1.4,zorder=4))
text(19.45,8.22,'+',12,c=orange)
arrow(18.79,9.07,19.27,8.57,c=blue);arrow(18.79,7.33,19.27,7.88,c=teal)
text(19.23,9.48,'1−λ',10,c=blue);text(19.25,7.05,'λ',12,c=teal)
for j in [2,1,0]:
 ax.add_patch(FancyBboxPatch((19.95+j*.08,7.36+j*.08),1.90,1.70,boxstyle='round,pad=.02,rounding_size=.05',facecolor='#efeaf6',edgecolor=purple if j==0 else 'none',lw=.9,zorder=3))
text(20.90,8.26,'动作专家\nTransformer ×16\n720维 + LoRA',10,c=purple)
arrow(19.86,8.22,19.92,8.22)
arrow(21.99,8.22,22.10,8.22)
tt=np.linspace(0,1,30)
ax.plot(22.15+.95*tt,7.9+.65*tt+.12*np.sin(tt*5),color=purple,lw=2,zorder=3)
ax.scatter(22.15+.95*tt[::4],7.9+.65*tt[::4]+.12*np.sin(tt[::4]*5),s=18,c=purple,zorder=4)
text(22.65,7.35,'16步动作\n+ 夹爪开合',11,c=purple)
text(20.7,6.35,'噪声 z → 单次 Drifting 前向',11,c=purple)
text(20.05,5.55,'F = (1−λ) F基础 + λ F部位',12,c=orange)
# Robot/history conditioning bus below inference groups, short upward feeds.
arrow(2.5,5.17,2.5,4.48,c=gray)
ax.plot([2.5,20.8],[4.48,4.48],color=gray,lw=1.2)
for x in [8.15,13.3,19.45,20.8]:arrow(x,4.48,x,4.79,c=gray)
text(12.1,4.16,'当前及历史状态条件：送入目标选择、部位评分、λ与动作专家',10,c=gray)
# One joint training band (not separate stages)
ax.add_patch(FancyBboxPatch((.4,.45),23.2,3.1,boxstyle='round,pad=.04,rounding_size=.12',facecolor='#fffaf1',edgecolor='#d5bc94',lw=1,zorder=0))
text(.75,3.18,'同一次联合训练',15,c=orange,ha='left',weight='bold')
text(23.15,3.18,'下方未来信息仅用于监督，不进入推理输入',11,c=orange,ha='right')
box(.85,1.45,3.1,1.05,'成功示范视频 + 动作序列\n构造物体与部位证据',fill='#fff6e6',edge=orange,sz=11)
box(5.3,1.45,3.5,1.05,'目标角色监督\n名称 ↔ 被操作物体',fill='#fff6e6',edge=orange,sz=12)
box(10.15,1.45,4.3,1.05,'部位监督 + 空间预测\n实际采用部位 ↔ 状态与动作',fill='#fff6e6',edge=orange,sz=12)
box(16.1,1.45,5.8,1.05,'Drifting动作损失 + 夹爪损失 + 连续性\n共同反传至可训练模块',fill='#fff6e6',edge=orange,sz=12)
ax.plot([4.0,4.55,4.55,19.0],[1.97,1.97,2.75,2.75],color=orange,lw=1.2,linestyle='--')
for bx in [7.05,12.3,19.0]:arrow(bx,2.75,bx,2.54,c=orange,dashed=True)
text(12,.89,'目标选择器无需单独预训练：从同一训练开始学习；早期偏重基础分支，部位逐渐有用后由λ调整融合。',12,c=ink)
text(.5,.10,'结构示意图；点云与轨迹为说明性绘制，不是模型预测结果。实线：前向特征；虚线：训练监督。',9,c=gray,ha='left')
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=str(out/'paper_network.alignment.json'),strict=True)
fig.savefig(out/'paper_network.png',dpi=300,bbox_inches='tight',facecolor='white')
fig.savefig(out/'paper_network.svg',bbox_inches='tight',facecolor='white')
fig.savefig(out/'paper_network.pdf',bbox_inches='tight',facecolor='white')
print(json.dumps({'figure':'paper_network.png','size':[23,11.5],'single_joint_training':True}))
