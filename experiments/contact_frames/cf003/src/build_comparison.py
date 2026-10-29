from pathlib import Path
import json,base64,subprocess,datetime,traceback,hashlib
import numpy as np,h5py,cv2,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/contact_frames/cf003';P=R/'experiments/contact_frames/cf001_frozen_review_002'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def main():
 cfg=json.loads((B/'config.json').read_text())
 assert json.loads((B/'stage_state.json').read_text())['stage']=='evaluation_complete'
 eps=cfg['fixed_review_episodes']+['tomato_sauce_demo_7'];clips=[]
 names={'white_yellow_mug_demo_5':'白黄杯 · demo5','alphabet_soup_demo_0':'罐头 · demo0','porcelain_mug_demo_9':'瓷杯 · demo9','tomato_sauce_demo_7':'番茄酱 · demo7（训练片）'}
 model_names=['CF001 原模型','CF002 去绝对位姿／单视频（未采用）','CF003 多视频／原网络','CF003 多视频／去绝对位姿']
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
 for ep in eps:
  original=R/'experiments/contact_frames/cf001' if ep=='tomato_sauce_demo_7' else P/ep
  d=dict(np.load(original/'artifacts/dataset.npz'));n=len(d['target']);fps=float(1/np.median(np.diff(d['frame_time'])))
  conf=json.loads((original/'config.json').read_text());preds=[]
  preds.append(np.load(original/'artifacts/frame_predictions.npz')['probabilities'])
  cf2=R/'experiments/contact_frames/cf002'
  preds.append(np.load(cf2/('artifacts/frame_predictions.npz' if ep=='tomato_sauce_demo_7' else ep+'/artifacts/frame_predictions.npz'))['probabilities'])
  for arm in ['full','no_world']:
   if ep=='tomato_sauce_demo_7':preds.append(np.array([np.load(B/f'runs/{arm}/seed_{seed}/{ep}_prediction.npz')['probability'] for seed in cfg['seeds']]))
   else:preds.append(np.load(B/f'evaluation_{arm}/{ep}/artifacts/frame_predictions.npz')['probabilities'])
  models=[]
  for name,p in zip(model_names,preds):
   lo=p.min(0);hi=p.max(0);valid=d['geometry_reliable']&d['state_valid'];pos=(lo>=.8)&valid;neg=(hi<=.2)&valid&((d['distance']-2*d['sigma'])>.006);unknown=~(pos|neg);rows=[]
   for t in range(n):
    state=0 if pos[t] else 1 if neg[t] else 2
    reason='三模型均高分，且几何证据可用' if state==0 else '三模型均低分，且有分离证据' if state==1 else '状态未知' if not d['state_valid'][t] else '几何证据不足' if not d['geometry_reliable'][t] else '三模型分歧或分数未达固定门槛'
    rows.append({'frame':t,'time':float(d['frame_time'][t]),'mean':float(p[:,t].mean()),'scores':p[:,t].tolist(),'state':state,'reason':reason,'weak_target':float(d['target'][t]) if d['weight'][t]>0 else None})
   models.append({'name':name,'rows':rows,'counts':[int(pos.sum()),int(neg.sum()),int(unknown.sum())]})
  video=R/f'docs/图示/CF003_{ep}_无评分双视角.mp4'
  ff=imageio_ffmpeg.get_ffmpeg_exe();cmd=[ff,'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','896x552','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
  with (B/f'encode_{ep}.log').open('w') as log,h5py.File(R/conf['rgbd']) as h:
   pr=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=log)
   for t in range(n):
    im=Image.new('RGB',(896,552),(15,23,40));dr=ImageDraw.Draw(im);dr.text((14,8),ep+'  |  第 '+str(t)+' / '+str(n-1)+' 帧',font=font,fill='white')
    dr.text((14,35),'主相机（模型视觉输入）',font=font,fill='#b5c1d5');dr.text((462,35),'腕相机（仅供查看）',font=font,fill='#b5c1d5')
    for x,key in [(0,'agentview'),(448,'robot0_eye_in_hand')]:im.paste(Image.fromarray(h[key+'/rgb'][t]).resize((448,448)),(x,64))
    dr.text((14,520),'评分与模型版本见右侧；切换模型保持同一视频帧',font=font,fill='#b5c1d5');pr.stdin.write(np.asarray(im).tobytes())
   pr.stdin.close();assert pr.wait()==0
  cap=cv2.VideoCapture(str(video));count=0
  while True:
   ok,a=cap.read()
   if not ok:break
   assert a.shape[:2]==(552,896);count+=1
  cap.release();assert count==n
  clips.append({'id':ep,'name':names[ep],'fps':fps,'n':n,'human':[56,95] if ep=='porcelain_mug_demo_9' else [52,132] if ep=='tomato_sauce_demo_7' else None,'trained':ep=='tomato_sauce_demo_7','models':models,'src':'data:video/mp4;base64,'+base64.b64encode(video.read_bytes()).decode()})
 page=R/'docs/CF003_接触网络优化对照.html';template=(B/'src/comparison_template.html').read_text()
 page.write_text(template.replace('__DATA__',json.dumps(clips,ensure_ascii=False)).replace('__TIME__',now()))
 save(B/'display_summary.json',{'timestamp':now(),'page':str(page),'clips':len(clips),'models':model_names,'videos_decoded':True,'human_labels_training':False})
 save(B/'display_locked.json',{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [page,*list((R/'docs/图示').glob('CF003_*_无评分双视角.mp4')),B/'src/build_comparison.py',B/'src/comparison_template.html']})
if __name__=='__main__':
 try:main()
 except Exception as e:save(B/'display_failure.json',{'type':type(e).__name__,'message':str(e),'stack':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
