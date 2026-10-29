
"""Epoch-five development preview; never accesses heldout content."""
import sys,pathlib,ast,json,time,hashlib,traceback,base64,os
import numpy as np,torch,h5py
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object');P=B.parents[1]
sys.path[:0]=[str(B/'src'),str(B/'engineering')]
import data,evaluate_frozen,review
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
CAMS=['agentview','robot0_eye_in_hand']
COLORS=np.array([[49,54,149],[116,173,209],[254,224,144],[244,109,67],[165,0,38]],dtype=np.uint8)
PAGE='LCJP001_第5轮开发集评分视频预览.html'
def main():
    import fcntl
    handle=open(B/'engineering/preview.lock','w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.2)
    best=B/'runs/v001/preview_epoch005.pt'
    request=json.loads((B/'engineering/preview_request.json').read_text())
    assert hashlib.sha256(best.read_bytes()).hexdigest()==request['checkpoint_sha256']
    assert hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()==request['preview_script_sha256']
    ck=torch.load(best,map_location='cpu',weights_only=False);assert ck['epoch']==5
    dataset=data.Dataset('dev')
    row=next(r for r in dataset.rows if r['id']==request['case'])
    assert row['split']=='dev' and row['episode']=='demo_45'
    out=B/'evaluation/preview_epoch005';out.mkdir(parents=True,exist_ok=True)
    _,render=dataset.paths(row)
    # Reuse the exact current-only inference body; omit all heldout loading/render adapters.
    tree=ast.parse((B/'src/evaluate_frozen.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    nodes=[n for n in fn.body if 51<=n.lineno<=72]
    class DevOnly(ast.NodeTransformer):
        def visit_Call(self,n):
            if isinstance(n.func,ast.Name) and n.func.id=='Dataset':
                return ast.copy_location(ast.Call(func=n.func,args=[ast.Constant('dev')],keywords=[]),n)
            if isinstance(n.func,ast.Attribute) and n.func.attr=='set_per_process_memory_fraction':
                n.args=[ast.Constant(.2)]
            return self.generic_visit(n)
        def visit_Constant(self,n):
            if n.value=='bbq_sauce_demo_40_scores.h5':return ast.copy_location(ast.Constant('bbq_sauce_demo_45_scores.h5'),n)
            return n
    module=DevOnly().visit(ast.Module(body=nodes,type_ignores=[]))
    scope=dict(vars(evaluate_frozen));scope.update(best=best,row=row,out=out,render=render,lock={'epochs':5})
    started=time.time()
    exec(compile(ast.fix_missing_locations(module),'<development_current_only_preview>','exec'),scope)
    summary=scope['summary'];pred=scope['pred']
    summary.update(split='development preview; already used for development objective, not final heldout test',
       checkpoint_epoch=5,training_continues=True,preview_only=True,
       limitations=['Development preview at epoch 5; not final evaluation','Same assets; not unseen-shape generalization','Gray denotes unknown, not a confirmed low score','Preference scores are not contact probabilities'],
       inference_seconds=time.time()-started)
    summary.pop('best_epoch',None)
    summary['preview_script_sha256']=request['preview_script_sha256']
    # Reuse original 3-D review and visibility logic in an isolated preview destination.
    rt=ast.parse((B/'src/review.py').read_text())
    class PreviewPaths(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,str):
                value=n.value.replace('LCJP001_LIBERO_Object联合训练逐点评分.html',PAGE).replace('review/v001','review/preview_epoch005').replace('lcjp001_assets','lcjp001_preview005_assets')
                value=value.replace('冻结后测试','第5轮开发集预览').replace('冻结测试','第5轮开发集预览')
                return ast.copy_location(ast.Constant(value),n)
            return n
    rs={'__name__':'development_preview_review'}
    exec(compile(ast.fix_missing_locations(PreviewPaths().visit(rt)),'<preview_review>','exec'),rs)
    summary['view']=rs['build_review'](row,pred,render,summary)
    directory=B/'review/preview_epoch005';video=directory/'epoch005_bbq_demo45.mp4'
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font=ImageFont.truetype(font_path,21);small=ImageFont.truetype(font_path,17)
    counts=[];start=time.time()
    with h5py.File(pred,'r') as predictions,h5py.File(render,'r') as rendered:
        writer=imageio.get_writer(str(video),fps=20,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-movflags','+faststart'])
        try:
            for t in range(row['frames']):
                canvas=Image.new('RGB',(1088,704),(16,22,34));draw=ImageDraw.Draw(canvas)
                draw.text((24,12),'第5轮开发集预览 | BBQ sauce / demo_45 | 训练继续运行',font=font,fill='white')
                draw.text((24,45),'当前帧输入评分；完整视频逐帧推理，非在线相机延迟测试',font=small,fill=(180,194,212))
                arrays=[np.array(rendered[c+'/rgb'][t],dtype=np.uint8) for c in CAMS]
                layers=[Image.fromarray(a).convert('RGBA') for a in arrays]
                overlays=[Image.new('RGBA',(256,256),(0,0,0,0)) for _ in CAMS]
                painters=[ImageDraw.Draw(im) for im in overlays]
                frame_count={'frame':t,'displayed':0,'unknown':0,'finite_points':0}
                for name in predictions[str(t)]:
                    g=predictions[str(t)][name];xyz=g['xyz'][:];scores=g['score'][:];lo=g['lower'][:];hi=g['upper'][:]
                    assert np.isfinite(scores).all()
                    selected=np.argsort(scores)[-int(np.ceil(.8*len(scores))):]
                    frame_count['finite_points']+=len(scores)
                    for ci,cam in enumerate(CAMS):
                        C=rendered[cam+'/camera_to_world'][t];K=rendered[cam+'/K'][t];depth=rendered[cam+'/depth_m'][t]
                        pc=(xyz-C[:3,3])@C[:3,:3]
                        px=pc[:,:2]/np.maximum(pc[:,2,None],1e-8)*np.array([K[0,0],K[1,1]])+K[:2,2]
                        for j in selected:
                            u,v=np.rint(px[j]).astype(int)
                            if pc[j,2]<=0 or not(0<=u<256 and 0<=v<256) or abs(float(depth[v,u])-float(pc[j,2]))>.008:continue
                            unknown=(hi[j]-lo[j])>=.8
                            color=(119,119,119) if unknown else tuple(int(x) for x in COLORS[min(int(scores[j]*5),4)])
                            painters[ci].ellipse((u-1,v-1,u+1,v+1),fill=(*color,60 if unknown else 185))
                            frame_count['displayed']+=1;frame_count['unknown']+=int(unknown)
                for ci in range(2):
                    im=Image.alpha_composite(layers[ci],overlays[ci]).convert('RGB').resize((512,512),Image.Resampling.NEAREST)
                    canvas.paste(im,(24+ci*528,105))
                    draw.text((24+ci*528,77),['主相机','腕部相机'][ci],font=font,fill='white')
                draw.text((24,625),f'帧 {t:03d}/{row["frames"]-1}  时间 {t/20:.2f}s  每个物体提议显示前80%真实点',font=small,fill='white')
                for k,c in enumerate(COLORS):
                    x=24+k*150;draw.rectangle((x,663,x+18,681),fill=tuple(int(v) for v in c))
                    draw.text((x+25,658),f'{k*.2:.1f}–{(k+1)*.2:.1f}',font=small,fill='white')
                draw.rectangle((800,663,818,681),fill=(119,119,119));draw.text((825,658),'未知 / 宽区间',font=small,fill='white')
                writer.append_data(np.asarray(canvas));counts.append(frame_count)
        finally:writer.close()
    reader=imageio.get_reader(str(video));decoded=0
    for frame in reader:
        assert frame.shape[:2]==(704,1088);decoded+=1
    reader.close();assert decoded==row['frames']
    summary.update(video_frames=decoded,video_fps=20,video_seconds=decoded/20,video_path=str(video),video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),video_render_seconds=time.time()-start,video_projection_counts=counts)
    page=P/'docs'/PAGE
    html=page.read_text()
    insert='<section style="padding:20px;background:#101622;color:white"><h2>第5轮开发集预览视频（训练未停止）</h2><p>主相机与腕部相机同步。固定绝对分数颜色；未知为灰。评分为模型偏好估计，不是接触概率。下方可逐帧查看点云。</p><video controls loop playsinline style="width:100%;max-width:1088px" src="data:video/mp4;base64,'+base64.b64encode(video.read_bytes()).decode()+'"></video></section>'
    import re
    html=re.sub(r'(<body[^>]*>)',lambda m:m.group(1)+insert,html,count=1) if re.search(r'<body[^>]*>',html) else html.replace('<h1',insert+'<h1',1)
    html=html.replace('冻结模型的当前帧预测','第5轮检查点的当前帧开发集预览').replace('实际训练与测试记录','开发集预览记录')
    assert '<video controls' in html and 'lcjp001_preview005_assets' in html
    page.write_text(html)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    (B/'engineering/preview_complete.json').write_text(json.dumps({'time':time.time(),'page':str(page),'video':str(video),'frames':decoded,'checkpoint_epoch':5,'needs_visual_review':True}))
    from datetime import datetime
    from zoneinfo import ZoneInfo
    with (P/'docs/无接触标签联合学习_LIBERO-Object网络与训练验证方案.md').open('a') as f:
        f.write('\n\n### '+datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')+' 第5轮开发集视频预览\n未中断训练，复制第5轮检查点对bbq_sauce/demo_45全部'+str(decoded)+'帧做当前输入预测。完整双相机评分视频与逐帧点云页面：'+PAGE+'。不是最终留出测试。评分分位数'+str(summary['score_quantiles'])+'；宽区间比例'+str(summary['wide_interval_fraction'])+'。默认每提议前80%排名，固定五段绝对色带，未知独立灰色。视频已逐帧解码核对，仍需浏览器视觉检查。\n')
if __name__=='__main__':
    try:main()
    except Exception as e:
        (B/'engineering/preview_failure.json').write_text(json.dumps({'error_type':type(e).__name__,'locations':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]}))
        with (B/'engineering/preview_traceback.log').open('w') as f:traceback.print_exc(file=f)
        sys.exit(1)
