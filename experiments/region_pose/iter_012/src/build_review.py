from pathlib import Path
import json,hashlib,html
import numpy as np,h5py
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/region_pose/iter_012';V=B.parent/'reviews/iter_012';D=R/'docs';V.mkdir(exist_ok=False);(V/'images').mkdir()
refs={x['id']:x for x in json.loads((B.parent/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
def png(arr,uv,colors,path):
 im=Image.fromarray(arr).resize((672,672));dr=ImageDraw.Draw(im);factor=672/arr.shape[0]
 for (u,v),color in zip(uv,colors):
  x,y=float(u)*factor,float(v)*factor;dr.ellipse((x-2,y-2,x+2,y+2),fill=color)
 im.save(path)
def link(name):return '../experiments/region_pose/reviews/iter_012/images/'+name
parts=['<!doctype html><meta charset="utf-8"><title>第012轮 夹持区域回映射审计</title><style>body{font:16px sans-serif;max-width:1380px;margin:30px auto;background:#f5f7fa;color:#182130}section{background:white;padding:24px;margin:25px 0}img{max-width:100%}.row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.two{display:grid;grid-template-columns:repeat(2,1fr);gap:15px}figure{margin:8px}figcaption{padding:8px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style><h1>第012轮：夹持区域回映射审计</h1><p>这些是完整示范产生的自动区域证据，不是新模型预测，也不是物理接触真值。没有新训练。夹前图仅显示单物体；夹持来源画面用于解释区域如何得到。</p><p>绿色：距夹前可见物体表面≤3毫米且有至少2个持物时刻支持；橙色：≤6毫米但未满足绿色条件；紫色：距夹前点云&gt;6毫米，可能不可见或对应错误。颜色表示证据状态，不判断杯内壁/杯柄抓法是否合理。</p><p>三维图不插值补面。1毫米网格选择真实来源点作代表，多个时刻不是多次抓取。对应端点遇首次失败便停止；未审计的中间帧不宣称可靠。</p>']
for sp in ['val','train']:
 for c in json.loads((B/sp/'summary.json').read_text())['cases']:
  name=c['event']+'_query'+str(c['query']);z=np.load(B/sp/(name+'.npz'))
  qp=z['query_pixel'];rgb=z['query_rgb'];near=z['distance_to_query']<=.006;good=z['supported_candidate'];colors=np.full(len(good),'#b153d1',object);colors[near]='#ec9b22';colors[good]='#16a05a'
  with h5py.File(refs[c['id']]['rgbd'],'r') as f:
   qimg=f['agentview/rgb'][c['query']];mask=np.full_like(qimg,210);mask[qp[:,1],qp[:,0]]=rgb
   lo=qp.min(0);hi=qp.max(0)+1;pad=12;lo=np.maximum(lo-pad,0);hi=np.minimum(hi+pad,[qimg.shape[1],qimg.shape[0]])
   im=Image.fromarray(mask);crop=im.crop((*lo,*hi));crop.resize((448,448)).save(V/'images'/(name+'_query.png'))
   pred=z['mapped_query_xyz'];T=z['query_camera_to_world'];K=z['query_K'];cam=(pred-T[:3,3])@T[:3,:3]
   uv=cam[:,:2]/cam[:,2,None]*[K[0,0],K[1,1]]+K[:2,2]
   dr=ImageDraw.Draw(im)
   # Display only candidate points consistent with visible query surface on this overlay; others retained in 3-D and sources.
   for (u,v),ok,col in zip(uv,z['distance_to_query']<=.003,colors):
    if ok:dr.ellipse((u-1,v-1,u+1,v+1),fill=col)
   im.crop((*lo,*hi)).resize((448,448)).save(V/'images'/(name+'_mapped.png'))
   fig=plt.figure(figsize=(6,5));ax=fig.add_subplot(111,projection='3d');query=z['query_xyz'];center=query.mean(0);a=(query-center)*1000
   ax.scatter(*a.T,c='#999999',s=1,alpha=.35);b=(pred-center)*1000
   ax.scatter(*b.T,c=colors.tolist(),s=9)
   radius=max(np.ptp(a,axis=0).max()/2,20);mid=(a.max(0)+a.min(0))/2
   for fun,mu in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],mid):fun(mu-radius,mu+radius)
   ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)');ax.set_zlabel('Z (mm)');ax.view_init(22,-65);fig.tight_layout();fig.savefig(V/'images'/(name+'_3d.png'),dpi=130);plt.close(fig)
   parts.append('<section id="'+html.escape(c['id'])+'"><h2>'+html.escape(c['id'])+' · '+('旧验证' if sp=='val' else '训练')+'</h2><p>夹前帧 '+str(c['query'])+'；持物来源端点 '+str(c['selected_frames'])+'；原始观测 '+str(c['raw_points'])+'，≤3毫米 '+str(c['near_query_points'])+'，跨时支持1毫米格 '+str(c['supported_1mm_voxels'])+'，未见或未支持 '+str(c['unsupported_or_unseen_points'])+'。</p><div class="row">')
   for suffix,label in [('_query.png','单物体夹前输入参考（没有模型评分）'),('_mapped.png','回映射的可见表面附近候选（≤3毫米）'),('_3d.png','完整候选与夹前点云；紫色未知点保留')]:
    parts.append('<figure><img src="'+link(name+suffix)+'"><figcaption>'+label+'</figcaption></figure>')
   parts.append('</div><p>留一变换敏感性最大帧95分位：'+format(max(x['leave_one_out_pointwise_max_shift_mm']['p95'] or 0 for x in c['per_frame']),'.3f')+'毫米；不是测量真值误差。</p><details><summary>查看所有双相机来源画面（有限两指体积内原始深度点）</summary>')
   for t in c['selected_frames']:
    parts.append('<h3>持物帧 '+str(t)+'</h3><div class="two">')
    for ci,camname in enumerate(['agentview','robot0_eye_in_hand']):
     ix=(z['frame']==t)&(z['camera']==ci);fn=name+'_'+str(t)+'_'+camname+'.png'
     png(f[camname+'/rgb'][t],z['pixel'][ix],colors[ix],V/'images'/fn)
     parts.append('<figure><img loading="lazy" src="'+link(fn)+'"><figcaption>'+camname+'；'+str(int(ix.sum()))+'个实测来源点</figcaption></figure>')
    parts.append('</div>')
   parts.append('</details></section>')
page=D/'第012轮_夹持区域回映射与来源.html';page.write_text('\n'.join(parts))
files=list((V/'images').glob('*.png'))+[page,Path(__file__)]
(V/'locked.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print('REVIEW_IMAGES',len(files)-2,'PAGE',page)
