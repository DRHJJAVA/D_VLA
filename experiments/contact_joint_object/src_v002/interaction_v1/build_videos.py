
import pathlib,json,time,datetime,traceback,hashlib
import numpy as np,cv2,imageio_ffmpeg
B=pathlib.Path('/mnt/nvme1/drh/D_VLA');R=B/'experiments/contact_joint_object/runs/v002/preflight/P1_interaction_v1';A=B/'docs/lcjp002_interaction_v1_assets'
def save(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
try:
 assert (R/'association_all_complete.json').exists()
 d=json.loads((A/'results.json').read_text());checks=[]
 for ep in d['episodes']:
  out=A/(ep['id']+'.mp4')
  writer=imageio_ffmpeg.write_frames(str(out),(1024,576),fps=ep['fps'],codec='libx264',quality=8,pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=16,output_params=['-movflags','+faststart'])
  writer.send(None)
  for f in ep['frames']:
   imgs=[]
   for cam in ['agentview','robot0_eye_in_hand']:
    img=cv2.imread(str(B/'docs'/f['views'][cam]['associated']));assert img is not None
    imgs.append(cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC))
   canvas=np.zeros((576,1024,3),dtype=np.uint8);canvas[32:544]=np.hstack(imgs)
   cv2.putText(canvas,'External camera',(12,23),cv2.FONT_HERSHEY_SIMPLEX,.62,(235,235,235),1,cv2.LINE_AA)
   cv2.putText(canvas,'Wrist camera',(524,23),cv2.FONT_HERSHEY_SIMPLEX,.62,(235,235,235),1,cv2.LINE_AA)
   active=', '.join('#'+str(v) for v in f['active_ids']) or 'none'
   label=f"Frame {f['t']}/{ep['frames_total']-1}   Associated objects: {active}   (geometry + motion evidence)"
   cv2.putText(canvas,label,(12,566),cv2.FONT_HERSHEY_SIMPLEX,.52,(235,235,235),1,cv2.LINE_AA)
   writer.send(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
  writer.close()
  cap=cv2.VideoCapture(str(out));count=0;size=None
  while True:
   ok,im=cap.read()
   if not ok:break
   count+=1;size=[im.shape[1],im.shape[0]]
  cap.release();assert count==ep['frames_total']
  ep['video']='lcjp002_interaction_v1_assets/'+out.name
  checks.append({'episode':ep['id'],'decoded_frames':count,'size':size,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
 d['training_updates']=0;d['status']='complete';d['updated']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
 save(A/'results.json',d);save(R/'video_integrity_check.json',{'time':d['updated'],'videos':checks,'all_frames_decoded':True})
 print(json.dumps({'videos':checks},ensure_ascii=True))
except Exception as e:
 safe={'exception':type(e).__name__,'frames':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]}
 save(R/'video_failure.json',safe);print(json.dumps(safe))
 raise SystemExit(1)
