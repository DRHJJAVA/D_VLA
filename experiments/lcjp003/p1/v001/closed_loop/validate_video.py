
import pathlib,json,time,traceback,datetime,os
from adapters import *
from video import overlay,encode_video
import cv2
def write(name,obj):
 (HERE/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
try:
 start=time.monotonic()
 eid='pick_up_the_bbq_sauce_and_place_it_in_the_basket__demo_0'
 write('validation_state.json',{'stage':'render_setup','time':stamp()})
 render=LiveRender(eid,HERE/('validation_video_'+str(time.time_ns())),capacity=20)
 old=OLD/'cache/rgbd'/(eid+'.h5')
 diffs={}
 with h5py.File(old,'r') as f:
  for cam in CAMS:
   for key in ['rgb','depth_m','K','camera_to_world','robot_mask']:
    a=render.out[cam+'/'+key][0];b=f[cam+'/'+key][0]
    diffs[cam+'/'+key]={'shape':list(a.shape),'max_abs':float(np.max(np.abs(a.astype(float)-b.astype(float))))}
 write('render_equivalence.json',{'time':stamp(),'differences':diffs})
 write('validation_state.json',{'stage':'segmentation_cpu','time':stamp()})
 proc=LiveProposals(device='cpu')
 n=proc.frame(render,0)
 write('validation_state.json',{'stage':'features_cpu','objects':n,'time':stamp()})
 adapter=LiveInput(eid,render.work,capacity=20,device='cpu')
 fr=adapter.frame(0);batch=adapter.batch(0)
 write('validation_state.json',{'stage':'model_cpu','time':stamp(),'schema':{k:list(v.shape) for k,v in batch.items()}})
 ckpt=torch.load(P/'step1750.pt',map_location='cpu',weights_only=False)
 model=LCJP003(ModelConfig()).eval();model.load_state_dict(ckpt['model'])
 gen=torch.Generator(device='cpu').manual_seed(20261026)
 records=[]
 writer=cv2.VideoWriter(str(HERE/'engineering_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1024,576))
 assert writer.isOpened()
 for t in range(4):
  astart=time.monotonic()
  if t:
   proc.frame(render,t);adapter.frame(t);batch=adapter.batch(t)
  with torch.inference_mode():
   noise=torch.randn(1,1,16,7,generator=gen)
   result=model(batch,noise)
  write('model_output_schema.json',{'type':str(type(result)),'keys':list(result) if isinstance(result,dict) else None,'shape':list(result.shape) if torch.is_tensor(result) else None})
  pred=result['actions'] if isinstance(result,dict) else result
  action=pred[0,0,0].cpu().numpy()*adapter.ds.action_std+adapter.ds.action_mean
  writer.write(overlay(fr if t==0 else adapter.ds.frame(eid,t),result,t,0,tag='ENGINEERING step1750 CPU'))
  success,done=render.step(action,t+1)
  records.append({'t':t,'action':action.tolist(),'success':success,'done':done,'seconds':time.monotonic()-astart})
  write('validation_state.json',{'stage':'closed_loop_cpu','time':stamp(),'completed_steps':t+1})
 writer.release()
 render.close()
 decoded=encode_video(HERE/'engineering_raw.mp4',HERE/'engineering_smoke.mp4',4)
 write('video_validation_complete.json',{'time':stamp(),'decoded_frames':decoded,'steps':records,'seconds':time.monotonic()-start,'engineering_only':True,'checkpoint_step':1750,'device':'cpu'})
except Exception as e:
 write('video_validation_failure.json',{'time':stamp(),'type':type(e).__name__,'message':str(e)[:800],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]})
