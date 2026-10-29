from pathlib import Path
import json,datetime,hashlib,base64,subprocess
import numpy as np,h5py
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/contact_frames/cf005_frozen_review_003'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
plan=json.loads((B/'plan.json').read_text());clips=[];checks=[]
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
ff=imageio_ffmpeg.get_ffmpeg_exe()
for ep in plan['cases']:
 cfg=json.loads((B/ep/'config.json').read_text());d=np.load(B/ep/'artifacts/dataset.npz');a=np.load(B/ep/'artifacts/frame_predictions.npz');n=len(d['states'])
 vid=R/'docs/图示'/('CF005复核_'+ep+'_无评分双视角.mp4');vid.parent.mkdir(exist_ok=True)
 proc=subprocess.Popen([ff,'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','896x552','-r','20','-i','-','-an','-vcodec','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(vid)],stdin=subprocess.PIPE,stderr=(B/ep/'encode.log').open('w'))
 with h5py.File(R/cfg['rgbd'],'r') as h:
  assert len(h['agentview/rgb'])==n and len(h['robot0_eye_in_hand/rgb'])==n
  for t in range(n):
   im=Image.new('RGB',(896,552),(14,22,38));draw=ImageDraw.Draw(im)
   draw.text((12,5),ep+'  |  第 '+str(t)+' / '+str(n-1)+' 帧',font=font,fill=(232,240,250))
   draw.text((12,32),'主相机（模型视觉输入）',font=small,fill=(185,200,222))
   draw.text((460,32),'腕相机（仅供查看）',font=small,fill=(185,200,222))
   im.paste(Image.fromarray(h['agentview/rgb'][t]),(0,58));im.paste(Image.fromarray(h['robot0_eye_in_hand/rgb'][t]),(448,58))
   draw.text((12,519),'CF005 冻结模型复核 · 分数在网页侧栏 · 闭合尝试不等于成功接触',font=small,fill=(185,200,222))
   proc.stdin.write(np.asarray(im).tobytes())
 proc.stdin.close();assert proc.wait()==0
 rows=[]
 for t in range(n):
  pr=a['probabilities'][:,t];state=0 if a['contact_candidate'][t] else 1 if a['observed_separation_candidate'][t] else 2
  target=float(d['target'][t]) if float(d['weight'][t])>0 else None
  reason='三个模型均高分，几何估计可用。' if state==0 else '三个模型均低分，并有几何分离证据。' if state==1 else '模型分数或种子一致性不足，保留未知。'
  if state==2 and not d['geometry_reliable'][t]:reason='几何证据不足，保留未知；高分不替代证据。'
  if not d['state_valid'][t]:reason='本帧状态不可用，保留未知。'
  if state==0 and target is not None and target<=.2:reason='模型高分，但与低几何弱目标冲突（弱目标非真值）。'
  rows.append({'frame':t,'time':t/20,'mean':float(pr.mean()),'scores':pr.astype(float).tolist(),'state':state,'reason':reason,'weak_target':target})
 attempts=[[52,69],[94,153]] if ep=='tomato_sauce_demo_3' else [[46,64],[80,158]]
 name='番茄酱 · demo3（两次闭合尝试）' if ep=='tomato_sauce_demo_3' else '罐头 · demo1（两次闭合尝试）'
 counts=[sum(x['state']==k for x in rows) for k in range(3)]
 clips.append({'id':ep,'name':name,'fps':20,'n':n,'human':None,'trained':False,'attempts':attempts,'note':'闭合尝试区间 '+str(attempts[0])+'、'+str(attempts[1])+' 仅供定位；不输入模型，也不是成功接触真值。','models':[{'name':'CF005 · 冻结三种子','rows':rows,'counts':counts}],'src':'data:video/mp4;base64,'+base64.b64encode(vid.read_bytes()).decode()})
 # decode the remote encoded video without copying it locally
 test=subprocess.run([ff,'-v','error','-i',str(vid),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 assert test.returncode==0 and len(test.stdout)==n*896*552*3
 checks.append({'episode':ep,'frames':n,'counts':counts,'video_sha256':sha(vid),'video_decode_frames_verified':n,'prediction_sha256':sha(B/ep/'artifacts/frame_predictions.npz')})
template=(B/'src/viewer_template.html').read_text()
page=template.replace('__CLIPS__',json.dumps(clips,ensure_ascii=False,separators=(',',':'))).replace('__RESULTS__','[]')
out=R/'docs/CF005_多次尝试视频接触预测查看.html';out.write_text(page)
save(B/'audits/display.json',{'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),'videos':checks,'html_hash':sha(out),'fixed_weights':'cf005/last600','no_retraining':True,'attempts_navigation_only':True})
save(B/'display_locked.json',{str(p):sha(p) for p in [out,B/'src/build_review.py',B/'src/viewer_template.html']+[R/'docs/图示'/('CF005复核_'+ep+'_无评分双视角.mp4') for ep in plan['cases']]})
print('review_page_ready')
