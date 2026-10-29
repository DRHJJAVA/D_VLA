"""Remote-only visual review of actual frozen student predictions. Yellow dots are top-20% points."""
from pathlib import Path
import html,json,sys
import h5py,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pointnet2_grasp import PointNet2Grasp
from stage2_train_visual_interaction_student import R,DATA,S,tensor
E={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
OUT=R/'docs/图示/第二步_学生真实预测'
OUT.mkdir(parents=True,exist_ok=True)
net=PointNet2Grasp().eval()
ck=torch.load(R/'outputs/v3_stage2/visual_interaction_student/seed_2026/best.pt',map_location='cpu',weights_only=False)
net.load_state_dict(ck['state'])
evalrows={x['id']:x for x in json.loads((R/'outputs/v3_stage2/visual_interaction_student/seed_2026/review/student_contact_test.json').read_text())['rows']}
def make(id):
 e=E[id];aid=e['asset_group'];ep=e['source_episode']
 sample=DATA/'test_inference'/(id+'.npz');graph=DATA/'graphs/test_inference'/(id+'.npz')
 with np.load(sample) as f:s={k:f[k].copy() for k in f.files if k in ('xyz','dino_feat','pixel','source_box')}
 s['target']=np.zeros(len(s['xyz']),dtype=np.float32)
 with np.load(graph) as f:g={k:f[k].copy() for k in f.files}
 with torch.inference_mode():
  xyz,feat,_,gg=tensor((id,s,g),'cpu')
  score=net(xyz,feat,gg).numpy()
 budget=max(1,int(np.ceil(.2*len(score))))
 selected=np.argpartition(score,-budget)[-budget:]
 rows=json.loads((R/'data/v2/stage1'/aid/(ep+'_states_contacts.json')).read_text())['rows']
 grip=max(range(len(rows)),key=lambda i:len(rows[i]['contacts']))
 with h5py.File(e['rgbd_path']) as f:
  first=Image.fromarray(f['agentview/rgb'][0])
  held=Image.fromarray(f['agentview/rgb'][grip])
 overlay=first.copy().convert('RGBA');layer=Image.new('RGBA',first.size,(0,0,0,0));draw=ImageDraw.Draw(layer)
 for u,v in s['pixel'][selected]:
  u=int(u);v=int(v);draw.ellipse((u-2,v-2,u+2,v+2),fill=(255,225,0,170))
 overlay=Image.alpha_composite(overlay,layer).convert('RGB')
 x1,y1,x2,y2=map(int,s['source_box']);pad=14
 box=(max(0,x1-pad),max(0,y1-pad),min(first.width,x2+pad),min(first.height,y2+pad))
 def crop(im):return im.crop(box).resize((448,448),Image.Resampling.NEAREST)
 canvas=Image.new('RGB',(448*3+32,448*2+60),(255,255,255))
 for i,im in enumerate((first,overlay,held)):
  canvas.paste(im,(i*464,28))
 for i,im in enumerate((crop(first),crop(overlay))):
  canvas.paste(im,(i*464,504))
 labels=['INPUT: first RGB-D frame','MODEL: top 20% yellow points','DEMO: later grasp frame','INPUT: object crop','MODEL: object crop']
 pen=ImageDraw.Draw(canvas)
 for x,y,label in [(0,8,labels[0]),(464,8,labels[1]),(928,8,labels[2]),(0,484,labels[3]),(464,484,labels[4])]:
  pen.text((x+4,y),label,fill=(0,0,0))
 name=id+'.jpg';canvas.save(OUT/name,quality=92)
 return dict(id=id,file=name,grasp_frame=grip,points=len(score),selected_points=budget,coverage=evalrows[id]['budget_20']['student'])
def main():
 items=[make(id) for id in S['test']]
 parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>第二步：模型真实预测人工审核</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1440px;margin:auto;padding:24px;color:#222}h1{font-size:26px}img{max-width:100%;height:auto}section{border-top:1px solid #ccc;padding:20px 0}.note{background:#fff8da;padding:12px}</style><h1>第二步：真实模型预测人工审核</h1><p class="note">每行左侧为测试段首帧原始 RGB-D 中的 RGB，中间黄点是冻结 PointNet++ 模型对首帧物体点云评分最高的前 20%，右侧为原示范后续夹持画面；下方是同一物体局部放大。黄点是模型输出，不是人工或仿真标签。深度作为三维点和模型输入使用。示范后续画面只供你审核，从未送入测试时的学生模型。旧仿真接触仅用于下方数字诊断，且尚未经人工复核。这里没有显示“包裹区域”真值，也未完成未见物体验证。</p>']
 for q in items:
  parts.append('<section><h2>'+html.escape(q['id'])+'</h2><p>首帧可见点 '+str(q['points'])+'；黄点 '+str(q['selected_points'])+'；示范夹持帧 '+str(q['grasp_frame'])+'；旧接触参考的 top20% 覆盖 '+format(q['coverage'],'.3f')+'（仅诊断）</p><img src="图示/第二步_学生真实预测/'+html.escape(q['file'])+'"></section>')
 parts.append('</html>')
 target=R/'docs/第二步_真实模型预测人工审核.html'
 target.write_text(''.join(parts),encoding='utf-8')
 (OUT/'manifest.json').write_text(json.dumps(dict(status='await_user_visual_review',checkpoint_epoch=ck['epoch'],point_color='yellow means top20 model score; no calibrated probability',test_only_first_frame=True,rows=items),indent=2,ensure_ascii=False))
 print(json.dumps(dict(page=str(target),count=len(items),checkpoint_epoch=ck['epoch'])))
if __name__=='__main__':main()
