import argparse, json, hashlib, time, re, os
from pathlib import Path
import numpy as np
import h5py, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/mnt/nvme1/drh/D_VLA')
OLD=ROOT/'experiments/contact_frames/cf018'
OUT=ROOT/'experiments/contact_frames/cf019'
CASE='alphabet_soup_demo_0'
PAIR=OUT/'rgbd_panels'
ROWS=json.loads((OLD/'prepared_fine'/CASE/'sensors.json').read_text())['rows']
M=json.loads((OLD/'episode_manifest.json').read_text())
META=next(x for x in M['cases'] if x['case']==CASE)
N=int(META['frames'])
HISTORY=(12,6,2,0)
LIMITS={'agentview':(0.20,2.00),'robot0_eye_in_hand':(0.02,0.55)}
CAMERAS=('agentview','robot0_eye_in_hand')
FONT=ImageFont.load_default()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(p.suffix+'.tmp')
 q.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
 q.replace(p)
def history(t):
 return sorted({max(0,t-k) for k in HISTORY})
def depth_color(depth,camera):
 lo,hi=LIMITS[camera]
 v=np.clip((np.nan_to_num(depth,nan=0.0)-lo)/(hi-lo),0,1)
 a=(255*(1-v)).astype(np.uint8)
 b=cv2.applyColorMap(a,cv2.COLORMAP_TURBO)
 b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
 b[(~np.isfinite(depth))|(depth<=0)]=0
 return Image.fromarray(b)
def draw_panel(h,t):
 canvas=Image.new('RGB',(896,928),(12,19,28))
 d=ImageDraw.Draw(canvas)
 d.text((12,6),f'ORIGINAL FRAME t={t}  time={t/20:.2f}s  '+'RGB-D SOURCE FRAME',fill='white',font=FONT)
 for j,cam in enumerate(CAMERAS):
  rgb=np.asarray(h[cam]['rgb'][t])
  dep=np.asarray(h[cam]['depth_m'][t])
  assert rgb.shape==(448,448,3) and dep.shape==(448,448)
  canvas.paste(Image.fromarray(rgb),(448*j,32))
  canvas.paste(depth_color(dep,cam),(448*j,480))
  d.text((448*j+8,34),cam+' RGB',fill='white',stroke_width=1,stroke_fill='black',font=FONT)
  lo,hi=LIMITS[cam]
  d.text((448*j+8,482),f'{cam} depth {lo:.2f}-{hi:.2f}m near=red far=blue',fill='white',stroke_width=1,stroke_fill='black',font=FONT)
 return canvas
def world_patch(h,cam,t,row):
 if not row.get('state_valid'):return None
 info=row.get('camera',{}).get(cam,{})
 uv=info.get('ee_projected_uv')
 ee=row.get('ee_pos_m')
 if uv is None or ee is None:return None
 depth=np.asarray(h[cam]['depth_m'][t])
 H,W=depth.shape;u,v=uv
 x0=max(0,int(round(u))-24);x1=min(W,int(round(u))+25)
 y0=max(0,int(round(v))-24);y1=min(H,int(round(v))+25)
 if x1<=x0 or y1<=y0:return None
 sub=depth[y0:y1,x0:x1]
 yy,xx=np.mgrid[y0:y1,x0:x1]
 ok=np.isfinite(sub)&(sub>0)&((xx-u)**2+(yy-v)**2<=24**2)
 if not ok.any():return {'depth_points':0}
 K=np.asarray(h[cam]['K'][t],dtype=np.float64)
 C=np.asarray(h[cam]['camera_to_world'][t],dtype=np.float64)
 z=sub[ok].astype(np.float64);xp=xx[ok].astype(np.float64);yp=yy[ok].astype(np.float64)
 p=np.stack([(xp-K[0,2])*z/K[0,0],(yp-K[1,2])*z/K[1,1],z],1)
 world=(C[:3,:3]@p.T).T+C[:3,3]
 dist=np.linalg.norm(world-np.asarray(ee,dtype=np.float64),axis=1)
 return {'depth_points':int(len(z)),'camera_depth_median_m':round(float(np.median(z)),4),'distance_to_ee_3d_m_p10':round(float(np.percentile(dist,10)),4),'distance_to_ee_3d_m_median':round(float(np.median(dist)),4)}
def prepare():
 PAIR.mkdir(parents=True,exist_ok=True)
 features=[]
 with h5py.File(META['rgbd_path'],'r') as h, h5py.File(META['raw_path'],'r') as raw:
  joints=raw['data'][f"demo_{META['demo_index']}"]['obs']['joint_states']
  for t in range(N):
   p=PAIR/f'{t:04d}.jpg'
   if not p.exists():
    img=draw_panel(h,t)
    img.save(p,'JPEG',quality=83,subsampling=0)
   assert p.stat().st_size>10000
   row=ROWS[t]
   assert row['frame']==t and row.get('obs_index')==(t-1 if t else None)
   state={'frame':t,'time_s':round(t/20,2),'obs_index':row.get('obs_index'),'state_valid':row.get('state_valid',False)}
   if t:
    for key in ['ee_pos_m','ee_ori_raw','gripper_joint_m','action_raw']:
     state[key]=row.get(key)
    state['joint_states']=np.asarray(joints[t-1]).round(5).tolist()
    state['rgbd_3d_near_ee']={cam:world_patch(h,cam,t,row) for cam in CAMERAS}
   features.append(state)
 atomic_json(OUT/'frame_features.json',{'case':CASE,'fps':20,'history_offsets':HISTORY,'depth_scale_m':LIMITS,'source':'RGB-D replay t / official obs t-1','rows':features})
 atomic_json(OUT/'input_locked.json',{'rgbd_path':META['rgbd_path'],'rgbd_sha256':META['rgbd_sha256'],'raw_path':META['raw_path'],'raw_sha256':META['raw_sha256'],'sensor_json_sha256':sha(OLD/'prepared_fine'/CASE/'sensors.json'),'frame_features_sha256':sha(OUT/'frame_features.json'),'image_count':N,'first_image_sha256':sha(PAIR/'0000.jpg'),'last_image_sha256':sha(PAIR/f'{N-1:04d}.jpg')})
 print(json.dumps({'prepared_frames':N,'features':len(features),'first_image_bytes':(PAIR/'0000.jpg').stat().st_size,'last_image_bytes':(PAIR/f'{N-1:04d}.jpg').stat().st_size}))
def prompt(t,features,ids):
 head=("You are observing one LIBERO robot demonstration. Task instruction: "+META['instruction']+". "
 "Decide ONLY whether the robot gripper fingers physically touch an external object at CURRENT original video frame t="+str(t)+". "
 "This includes brief touch during an unsuccessful attempt and continued contact while carrying or operating. Lifting or task completion is NOT required. "
 "Do not infer contact from closing, proximity, task instruction, or eventual success alone. Do not count the robot touching itself. "
 "Earlier images are causal history, never future. Each image contains top row RGB (main camera left, wrist right) and bottom row aligned metric depth, with color scale printed on image. "
 "State rows align replay image t to official obs/action t-1. t=0 state is unavailable. "
 "The 3D numbers are metric depth observations near the projected end-effector reference, NOT measured finger contact; robot/background pixels may be included. "
 "Use current RGB-D and history, gripper state and actions/poses as evidence together. If contact surface is hidden and evidence is insufficient, choose unknown. "
 "Return one JSON object only with keys frame (integer), state (contact|no_contact|unknown), object (short string or null), reason (one short sentence). No interval and no score.\n")
 table={'frames':[features[i] for i in ids]}
 return head+'\nSensor and geometry observations (chronological JSON):\n'+json.dumps(table,ensure_ascii=False,separators=(',',':'))+'\nNow judge CURRENT t='+str(t)+'.'
def parse(s,t):
 m=re.search(r'\{.*?\}',s,re.S)
 if not m:return {'frame':t,'state':'unknown','object':None,'reason':'model response was not JSON','parse_status':'invalid'}
 try:d=json.loads(m.group())
 except:return {'frame':t,'state':'unknown','object':None,'reason':'model response JSON parse failed','parse_status':'invalid'}
 state=str(d.get('state','unknown')).lower().strip()
 if state not in ('contact','no_contact','unknown'):state='unknown'
 if d.get('frame')!=t:state='unknown';status='wrong_frame'
 else:status='ok'
 return {'frame':t,'state':state,'object':d.get('object'),'reason':str(d.get('reason',''))[:300],'parse_status':status}
def infer(only=None):
 import torch
 from transformers import AutoProcessor,AutoModelForImageTextToText
 from qwen_vl_utils import process_vision_info
 torch.manual_seed(2026)
 W=OLD/'weights'
 processor=AutoProcessor.from_pretrained(str(W),local_files_only=True)
 model=AutoModelForImageTextToText.from_pretrained(str(W),dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa',local_files_only=True).eval()
 features=json.loads((OUT/'frame_features.json').read_text())['rows']
 assert len(features)==N
 targets=[only] if only is not None else list(range(N))
 for t in targets:
  dest=OUT/'predictions'/f'{t:04d}.json'
  if dest.exists():continue
  ids=history(t)
  content=[{'type':'text','text':prompt(t,features,ids)}]
  for i in ids:
   content.append({'type':'text','text':f'Image for ORIGINAL t={i}'+(' CURRENT.' if i==t else ' HISTORY.')})
   content.append({'type':'image','image':'file://'+str(PAIR/f'{i:04d}.jpg')})
  messages=[{'role':'user','content':content}]
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  images,videos,kw=process_vision_info(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
  inputs=processor(text=[text],images=images,videos=videos,**kw,padding=True,return_tensors='pt').to(model.device)
  stamp=time.time()
  with torch.inference_mode():
   output=model.generate(**inputs,max_new_tokens=160,do_sample=False)
  raw=processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
  parsed=parse(raw,t)
  record={'case':CASE,'frame':t,'time_s':round(t/20,2),'history_frame_ids':ids,'state':parsed['state'],'object':parsed['object'],'reason':parsed['reason'],'parse_status':parsed['parse_status'],'raw_response':raw,'latency_s':round(time.time()-stamp,3),'input_image_sha256':{str(i):sha(PAIR/f'{i:04d}.jpg') for i in ids},'model_revision':'0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'}
  atomic_json(dest,record)
  atomic_json(OUT/'progress.json',{'completed':len(list((OUT/'predictions').glob('*.json'))),'total':N,'last_frame':t,'updated_epoch':time.time()})
  print(json.dumps({'frame':t,'state':parsed['state'],'parse':parsed['parse_status'],'latency_s':record['latency_s']},ensure_ascii=False),flush=True)
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('mode',choices=['prepare','infer'])
 ap.add_argument('--only',type=int)
 args=ap.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 if args.mode=='prepare':prepare()
 else:infer(args.only)
if __name__=='__main__':main()
