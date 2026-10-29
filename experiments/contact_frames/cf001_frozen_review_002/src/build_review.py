from pathlib import Path
import json,datetime,hashlib,base64,subprocess,traceback
import numpy as np,h5py,cv2,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/contact_frames/cf001_frozen_review_002'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def main():
 cases=json.loads((B/'plan.json').read_text())['cases'];clips=[]
 font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
 f=ImageFont.truetype(font,21);small=ImageFont.truetype(font,17)
 labels=['接触候选','可观测分离候选','未知']
 colors=[(245,191,58),(73,155,236),(153,164,181)]
 for ep in cases:
  c=B/ep;cfg=json.loads((c/'config.json').read_text());d=np.load(c/'artifacts/dataset.npz');p=np.load(c/'artifacts/frame_predictions.npz');n=len(p['mean'])
  fps=float(1/np.median(np.diff(d['frame_time'])))
  assert np.max(np.abs(np.diff(d['frame_time'])-1/fps))<1e-5
  rows=[]
  for t in range(n):
   state=0 if p['contact_candidate'][t] else 1 if p['observed_separation_candidate'][t] else 2
   rows.append({'frame':t,'seconds':float(d['frame_time'][t]),'score':float(p['mean'][t]),'low':float(p['low'][t]),'high':float(p['high'][t]),'state':state,'state_label':labels[state],'weak_target':float(d['target'][t]) if d['weight'][t]>0 else None})
  save(c/'frames.json',rows)
  output=R/f'docs/图示/CF001_{ep}_冻结预测.mp4'
  cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','896x608','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
  with (c/'encode.log').open('w') as log,h5py.File(R/cfg['rgbd']) as h:
   proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=log)
   for t,row in enumerate(rows):
    im=Image.new('RGB',(896,608),(17,24,39));dr=ImageDraw.Draw(im)
    dr.text((16,10),ep+'  |  第 '+str(t)+' / '+str(n-1)+' 帧',font=f,fill='white')
    dr.text((16,40),'主相机（模型视觉输入）',font=small,fill=(180,190,206));dr.text((464,40),'腕相机（仅供查看）',font=small,fill=(180,190,206))
    for x,key in [(0,'agentview'),(448,'robot0_eye_in_hand')]:im.paste(Image.fromarray(h[key+'/rgb'][t]).resize((448,448)),(x,64))
    dr.text((16,524),row['state_label']+'  |  模型均分 '+format(row['score'],'.3f')+'  |  三模型范围 '+format(row['low'],'.3f')+'–'+format(row['high'],'.3f'),font=f,fill=colors[row['state']])
    dr.text((16,560),'源记录时间 '+format(row['seconds'],'.3f')+' 秒 · 冻结权重 · 未参与CF001训练 · 分数不是成功概率',font=small,fill=(190,199,215))
    proc.stdin.write(np.asarray(im).tobytes())
   proc.stdin.close();assert proc.wait()==0
  cap=cv2.VideoCapture(str(output));count=0
  while True:
   ok,frame=cap.read()
   if not ok:break
   assert frame.shape[:2]==(608,896);count+=1
  cap.release();assert count==n
  summary=json.loads((c/'inference_summary.json').read_text())
  clips.append({'id':ep,'name':{'white_yellow_mug_demo_5':'白黄杯 · demo5','alphabet_soup_demo_0':'罐头 · demo0','porcelain_mug_demo_9':'瓷杯 · demo9'}[ep],'fps':fps,'rows':rows,'summary':summary,'human':None,'trained':False,'src':'data:video/mp4;base64,'+base64.b64encode(output.read_bytes()).decode()})
 old=R/'experiments/contact_frames/cf001';d=np.load(old/'artifacts/dataset.npz');p=np.load(old/'artifacts/frame_predictions.npz');rows=[]
 for t in range(len(p['mean'])):
  state=0 if p['contact_candidate'][t] else 1 if p['observed_separation_candidate'][t] else 2
  rows.append({'frame':t,'seconds':float(d['frame_time'][t]),'score':float(p['mean'][t]),'low':float(p['low'][t]),'high':float(p['high'][t]),'state':state,'state_label':labels[state],'weak_target':float(d['target'][t]) if d['weight'][t]>0 else None})
 pos=np.arange(52,133);comparison={'confirmed_frames':len(pos),'candidate_within_confirmed':int(p['contact_candidate'][pos].sum()),'separation_within_confirmed':int(p['observed_separation_candidate'][pos].sum()),'unknown_within_confirmed':int(p['unknown'][pos].sum()),'outside_interval':'unreviewed; not negative'}
 save(B/'human_review_comparison.json',comparison)
 clips.append({'id':'tomato_sauce_demo_7','name':'原番茄酱 · demo7（含你的复核）','fps':20,'rows':rows,'summary':{'contact_candidate':int(p['contact_candidate'].sum()),'separation_candidate':int(p['observed_separation_candidate'].sum()),'unknown':int(p['unknown'].sum())},'human':[52,132],'trained':True,'src':'data:video/mp4;base64,'+base64.b64encode((R/'docs/图示/CF001_接触时段预测.mp4').read_bytes()).decode()})
 template=(B/'src/viewer_template.html').read_text()
 page=R/'docs/CF001_多视频接触预测查看.html'
 page.write_text(template.replace('__CLIPS__',json.dumps(clips,ensure_ascii=False)).replace('__TIME__',now()))
 locks={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in B.rglob('*') if p.is_file() and p.suffix not in ['.log','.pyc'] and p.name not in ['display_locked.json','stage_state.json']}
 locks[str(page.relative_to(R))]=hashlib.sha256(page.read_bytes()).hexdigest()
 for ep in cases:
  p=R/f'docs/图示/CF001_{ep}_冻结预测.mp4';locks[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
 save(B/'display_locked.json',locks)
 save(B/'stage_state.json',{'stage':'display_ready','timestamp':now(),'page':str(page)})
 print(json.dumps({'page':str(page),'comparison':comparison},ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(B/'display_failure.json',{'type':type(e).__name__,'message':str(e),'stack':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
