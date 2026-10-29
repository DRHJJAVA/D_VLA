"""Internal visual sanity check of development-only automatic localization (no simulator truth)."""
from pathlib import Path
import html,json
import h5py
from PIL import Image,ImageDraw
R=Path('/mnt/nvme1/drh/D_VLA')
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
OUT=R/'docs/图示/第二步_定位开发诊断'
PAGE=R/'docs/第二步_定位开发诊断.html'
def main():
    rows=json.loads(M.read_text())['rows'];OUT.mkdir(parents=True,exist_ok=True)
    items=[]
    for e in rows:
        id=e['id'];aid=e['asset_group'];ep=e['source_episode']
        with h5py.File(e['rgbd_path']) as f:
            n=len(f['agentview/rgb']);first=Image.fromarray(f['agentview/rgb'][0]).convert('RGB');later=Image.fromarray(f['agentview/rgb'][int(n*.65)]).convert('RGB')
        ref=json.loads((R/'data/v3/proposals/motion_rgbd_rerank'/aid/(ep+'.json')).read_text())['selected']['box']
        ori=json.loads((R/'data/v3/proposals/first_frame_templates/dev'/(id+'.json')).read_text())['candidates'][0]['box']
        masked=json.loads((R/'data/v3/proposals/first_frame_masked_templates/dev'/(id+'.json')).read_text())['candidates'][0]['box']
        left=first.copy();d=ImageDraw.Draw(left)
        for b,color in ((ref,'lime'),(ori,'yellow'),(masked,'magenta')):d.rectangle(tuple(b),outline=color,width=3)
        canvas=Image.new('RGB',(920,480),'white')
        canvas.paste(left,(0,26));canvas.paste(later,(464,26))
        pen=ImageDraw.Draw(canvas);pen.text((3,5),'first: green full-sequence visual; yellow raw template; magenta masked template',fill='black');pen.text((467,5),'later demonstration frame: audit only',fill='black')
        file=id+'.jpg';canvas.save(OUT/file,quality=90)
        items.append(dict(id=id,file=file,frame=int(n*.65)))
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>第二步定位开发诊断</title><style>body{font:16px system-ui;max-width:1000px;margin:auto;padding:20px}section{border-top:1px solid #ccc;padding:15px}img{max-width:100%}</style><h1>定位开发集：8段</h1><p>仅供技术诊断。左图绿色是完整序列的自动视觉候选（不等于真值），黄色是原首帧模板，紫红色是前景遮罩模板；右图是后续画面。没有仿真/人工物体标签输入训练。</p>']
    for x in items:parts.append('<section><h2>'+html.escape(x['id'])+'</h2><img src="图示/第二步_定位开发诊断/'+html.escape(x['file'])+'"></section>')
    parts.append('</html>')
    assert not PAGE.exists();PAGE.write_text(''.join(parts))
    print(json.dumps(dict(page=str(PAGE),episodes=len(items)),ensure_ascii=False))
if __name__=='__main__':main()
