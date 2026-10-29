import pathlib,json,html,time,subprocess
import numpy as np,cv2
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA')
def overlay(fr,result,t,trial,tag='step3000'):
 scores=result['scores'][0].detach().cpu().numpy()
 valid=fr['point_valid'].astype(bool)&np.isfinite(scores)
 colours=np.zeros((len(scores),3),np.uint8);selected=np.zeros(len(scores),bool)
 for oid in np.unique(fr['object_id'][valid]):
  ids=np.flatnonzero(valid&(fr['object_id']==oid));ids=ids[np.argsort(scores[ids],kind='stable')]
  ids=ids[-max(1,int(np.ceil(len(ids)*.8))):];selected[ids]=True
  for ix,q in zip(ids,(np.arange(len(ids))+.5)/len(ids)):
   colours[ix]=(245,70,65) if q>=.95 else (255,150,50) if q>=.9 else (250,220,80) if q>=.8 else (70,210,140) if q>=.6 else (70,150,245)
 panels=[]
 for ci in range(2):
  im=fr['rgb'][ci].copy()
  for idx in np.flatnonzero(selected&fr['view_visible'][:,ci]):
   u,v=fr['projected_uv'][idx,ci]
   if np.isfinite(u+v) and 0<=u<256 and 0<=v<256:
    cv2.circle(im,(int(round(u)),int(round(v))),1,tuple(int(x) for x in colours[idx]),-1)
  panels.append(cv2.resize(im,(512,512),interpolation=cv2.INTER_NEAREST))
 canvas=np.concatenate(panels,axis=1);banner=np.zeros((64,1024,3),np.uint8)
 cv2.putText(banner,f'Trial {trial+1:02d} | observed frame {t:04d} | {tag} | agentview / wrist',(14,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(230,230,230),1)
 cv2.putText(banner,'Top80% per object; rerank: blue <60 | green 60-80 | yellow 80-90 | orange 90-95 | red >=95',(14,50),cv2.FONT_HERSHEY_SIMPLEX,.47,(220,220,220),1)
 return cv2.cvtColor(np.concatenate([banner,canvas],axis=0),cv2.COLOR_RGB2BGR)
def encode_video(src,dst,expected):
 import imageio_ffmpeg
 cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',str(src),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-threads','2',str(dst)]
 p=subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 if p.returncode:raise RuntimeError('H264 encoding failed')
 cap=cv2.VideoCapture(str(dst));n=0
 while True:
  ok,im=cap.read()
  if not ok:break
  n+=1
 cap.release()
 if n!=expected:raise RuntimeError('Decoded frame count mismatch')
 return n
def page(results):
 rows=[]
 for x in results:
  label='成功' if x['success'] else '未成功'
  rows.append('<article><h2>试验 '+str(x['trial']+1)+' | '+html.escape(x['episode'])+' | '+label+'</h2><p>'+str(x['steps'])+' 控制步；'+html.escape(x['stop_reason'])+'</p><video controls preload="metadata" src="'+x['video_relative']+'"></video></article>')
 body='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>LCJP003 P1 闭环测试</title><style>body{margin:32px auto;max-width:1100px;padding:0 20px;background:#10151e;color:#e5eaf3;font:16px/1.6 system-ui}video{width:100%;background:#000}article{margin:24px 0;padding:18px;background:#1a2332;border-radius:12px}</style><h1>LCJP003 P1 · 真实闭环测试</h1><p>每次预测后执行首个动作，再读取双相机新观测。3000次检查点；20次训练初始状态测试。视频按20帧/秒播放，计算延迟另行记录。</p><p>左：外部相机；右：腕部相机。每个物体保留最高80%点，并在保留点内重新排名：最高5%红、90–95%橙、80–90%黄、60–80%绿、其余蓝。颜色表示相对排名。完整点、分数和失败保存在远程。遮挡点不投影到前景。</p><p>已完成 '+str(len(results))+'/20，成功 '+str(sum(x['success'] for x in results))+' 次。</p>'+''.join(rows)+'</html>'
 (ROOT/'docs/LCJP003_P1_闭环测试.html').write_text(body,encoding='utf-8')
