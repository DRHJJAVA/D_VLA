"""Remote-only review page of locked fresh-test predictions, including all eight cases."""
from pathlib import Path
import html,json,math
import h5py,numpy as np
from PIL import Image,ImageDraw
R=Path('/mnt/nvme1/drh/D_VLA')
MANIFEST=R/'data/v3/manifests/stage2_fresh_blind_8.json'
OUT=R/'outputs/v3_stage2/fresh_blind_8'
LOCK=OUT/'predictions_locked.json'
EVAL=OUT/'fresh_blind_contact_diagnostic.json'
CONTACT=R/'data/v3/evaluation/fresh_blind_contacts'
FIG=R/'docs/图示/第二步_新留出示范真实预测'
PAGE=R/'docs/第二步_新留出示范模型预测人工审核.html'
def make(e,audit):
    id=e['id']
    with np.load(OUT/'predictions'/(id+'.npz')) as f:
        score=f['score'].copy();pixel=f['pixel'].copy();box=f['source_box'].copy()
    k=max(1,math.ceil(.2*len(score)))
    top=np.argpartition(score,-k)[-k:]
    state=json.loads((CONTACT/e['asset_group']/(e['source_episode']+'_states_contacts.json')).read_text())
    rows=state['rows']
    grip=max(range(len(rows)),key=lambda i:len(rows[i]['contacts']))
    with h5py.File(e['rgbd_path']) as f:
        first=Image.fromarray(f['agentview/rgb'][0]).convert('RGB')
        held=Image.fromarray(f['agentview/rgb'][grip]).convert('RGB')
    overlay=first.convert('RGBA')
    layer=Image.new('RGBA',first.size,(0,0,0,0));pen=ImageDraw.Draw(layer)
    for u,v in pixel[top]:
        u=int(u);v=int(v)
        pen.ellipse((u-2,v-2,u+2,v+2),fill=(255,220,0,175))
    overlay=Image.alpha_composite(overlay,layer).convert('RGB')
    x1,y1,x2,y2=map(int,box);pad=16
    rect=(max(0,x1-pad),max(0,y1-pad),min(first.width,x2+pad),min(first.height,y2+pad))
    def crop(im):return im.crop(rect).resize((448,448),Image.Resampling.NEAREST)
    width=448*3+32
    canvas=Image.new('RGB',(width,448*2+60),(255,255,255))
    for i,im in enumerate((first,overlay,held)):canvas.paste(im,(i*464,28))
    for i,im in enumerate((crop(first),crop(overlay))):canvas.paste(im,(i*464,504))
    draw=ImageDraw.Draw(canvas)
    labels=[(4,7,'INPUT: first RGB-D frame (RGB shown)'),(468,7,'MODEL: highest 20% yellow points'),(932,7,'DEMO: later contact frame'),(4,484,'INPUT: object crop'),(468,484,'MODEL: object crop')]
    for x,y,label in labels:draw.text((x,y),label,fill=(0,0,0))
    FIG.mkdir(parents=True,exist_ok=True)
    img=id+'.jpg';canvas.save(FIG/img,quality=92)
    return dict(id=id,image=img,contact_frame=grip,visible_points=len(score),selected_points=k,diagnostic=audit)
def main():
    assert LOCK.exists() and EVAL.exists()
    lock=json.loads(LOCK.read_text());assert lock['status']=='predictions_locked_before_evaluation'
    report=json.loads(EVAL.read_text())
    byid={x['id']:x for x in report['rows']}
    entries=json.loads(MANIFEST.read_text())['rows']
    assert len(entries)==8 and len(byid)==8
    items=[make(e,byid[e['id']]) for e in entries]
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>第二步：新留出示范真实预测人工审核</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1440px;margin:auto;padding:24px;color:#222}h1{font-size:26px}img{max-width:100%;height:auto}section{border-top:1px solid #bbb;padding:20px 0}.note{background:#fff7d6;padding:14px}small{color:#555}</style><h1>第二步：新留出示范的真实模型预测</h1><p class="note">以下完整列出预先锁定的 8 段新测试示范。左图为输入首帧 RGB-D 的 RGB 画面（深度已用于三维点云），中图黄点为三个冻结 PointNet++ 模型平均评分最高的前 20% 物体点，右图为原示范后续接触画面，仅供人工审核。黄点是模型输出，不是标签，也不是校准的抓取概率。测试时物体定位、特征和评分均仅使用首帧；模型训练用完整训练示范 RGB-D 序列提取自动视觉线索。下方数值仅以尚未人工复核的旧仿真接触做独立诊断，无法验证夹爪包裹区域或未见物体迁移。未经您审核，第二步不通过，也不进入第三步。</p>']
    for x in items:
        r=x['diagnostic'];b=r.get('budget_20')
        if b:metrics='旧接触 top20% 覆盖 '+format(b['student_ensemble'],'.3f')+'；随机 '+format(b['random'],'.3f')
        else:metrics='旧接触不可评估：'+html.escape(r.get('reason','未知'))
        parts.append('<section><h2>'+html.escape(x['id'])+'</h2><p>首帧物体点 '+str(x['visible_points'])+'；黄点 '+str(x['selected_points'])+'；示范接触帧 '+str(x['contact_frame'])+'；'+metrics+'</p><img src="图示/第二步_新留出示范真实预测/'+html.escape(x['image'])+'"></section>')
    parts.append('</html>')
    assert not PAGE.exists()
    PAGE.write_text(''.join(parts),encoding='utf-8')
    (FIG/'manifest.json').write_text(json.dumps(dict(status='await_user_visual_review',prediction_lock=str(LOCK),all_8_cases=True,scope='same assets, new episode IDs, first-frame query',rows=items),indent=2,ensure_ascii=False))
    print(json.dumps(dict(page=str(PAGE),cases=len(items)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
