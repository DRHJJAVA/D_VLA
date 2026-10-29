import argparse,json,time,hashlib,re,os
from pathlib import Path
import torch
from transformers import AutoProcessor,AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
R=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf018')
W=R/'weights'
cases=json.loads((R/'episode_manifest.json').read_text())['cases']
arms=['A','B','C','D']
prompt_base=("You analyze an actual robot demonstration. Report intervals in original integer frame IDs (0-based, inclusive) where any gripper finger physically contacts any object or articulated part. Touching, pushing, sliding, grasping and sustained holding all count. A failed task can still involve contact. Closing, commanded action, lifting, hand/object co-motion, and task instructions alone do not prove contact. Do not infer a standard operation timeline. Inspect all objects and multiple attempts. If contact is occluded or cannot be decided, use unknown; distinguish direct visual support from temporal inference. Return only JSON with keys contact_intervals (array of {object, start_frame, end_frame, start_possible_range, end_possible_range, evidence_mode, evidence_frame_ids}), unknown_intervals (array of {start_frame,end_frame,reason}), and notes. Do not invent exact boundaries between frames that were not shown. ")
def frames_for(n,kind,start=None):
 if kind=='smoke':return [20,22]
 if kind=='fine':return list(range(start,start+40))
 if kind=='global':
  x=list(range(0,n,10))
  if x[-1]!=n-1:x.append(n-1)
  return x
 assert kind=='window'
 end=min(start+79,n-1)
 x=list(range(start,end+1,2))
 
 return x
def table(rows,indices):
 out=[]
 for t in indices:
  z=rows[str(t)] if isinstance(rows,dict) else rows[t//2 if t%2==0 else -1]
  if z['frame']!=t:raise RuntimeError('sensor frame mismatch')
  if not z['state_valid']:
   out.append(f'{t}: state=null')
   continue
  cams=z['camera']
  a=cams['agentview'];w=cams['robot0_eye_in_hand']
  out.append(f"{t}: ee_m={z['ee_pos_m']}; ee_ori_raw={z['ee_ori_raw']}; gripper_joint_m={z['gripper_joint_m']}; action_raw={z['action_raw']}; projected_EE_11x11_depth_medians_m=[{a.get('local_depth_median_m')},{w.get('local_depth_median_m')}]; depth_valid=[{a.get('local_depth_valid_fraction')},{w.get('local_depth_valid_fraction')}]")
 return '\n'.join(out)
def parse(text):
 s=text.strip()
 if s.startswith(chr(96)*3):s=s.split('\n',1)[-1].rsplit(chr(96)*3,1)[0].strip()
 try:return json.loads(s),'ok'
 except:pass
 a=s.find('{');b=s.rfind('}')
 if a>=0 and b>a:
  try:return json.loads(s[a:b+1]),'extracted_json'
  except:pass
 return None,'unparsed'
def one(model,processor,case,arm,kind,start=None):
 name=case['case'];n=case['frames']
 frame_ids=frames_for(n,kind,start)
 prepared=R/('prepared_fine' if kind=='fine' else 'prepared')/name
 sensor=json.loads((prepared/'sensors.json').read_text())['rows']
 rowmap={r['frame']:r for r in sensor}
 if any(t not in rowmap for t in frame_ids):
  raise RuntimeError('missing preprepared frame; fine layer needs separate preparation')
 prompt=prompt_base+f'Only frames {frame_ids[0]} through {frame_ids[-1]} are provided, with explicit labels. Return contact and unknown spans supported by these frames; do not answer about outside this clip.'
 if arm in ['B','D']:prompt+=' Official task instruction (intent only): '+case['instruction']+'.'
 if arm in ['C','D']:prompt+=' Sensor readings follow. Raw action is a command, not measured contact. Projected-EE depth can include robot or unrelated surfaces; object identity is unassigned. Missing values are not separation.\n'+table(rowmap,frame_ids)
 content=[{'type':'text','text':prompt}]
 for t in frame_ids:
  content.append({'type':'text','text':f'Original frame {t} at {t/20:.2f}s, main camera:'})
  content.append({'type':'image','image':'file://'+str(prepared/f'{t:04d}_agentview.jpg'),'resized_height':448,'resized_width':448})
  content.append({'type':'text','text':f'Original frame {t}, wrist camera:'})
  content.append({'type':'image','image':'file://'+str(prepared/f'{t:04d}_robot0_eye_in_hand.jpg'),'resized_height':448,'resized_width':448})
 messages=[{'role':'user','content':content}]
 text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 images,videos,kw=process_vision_info(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
 inputs=processor(text=[text],images=images,videos=videos,do_resize=False,return_tensors='pt').to(model.device)
 torch.cuda.reset_peak_memory_stats()
 t0=time.monotonic()
 with torch.inference_mode():
  output=model.generate(**inputs,max_new_tokens=2048,do_sample=False)
 latency=time.monotonic()-t0
 decoded=processor.batch_decode([output[0,inputs['input_ids'].shape[-1]:]],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
 obj,status=parse(decoded)
 out={'case':name,'arm':arm,'kind':kind,'start':start,'frame_ids':frame_ids,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'frame_count':len(frame_ids),'image_count':len(images),'text_input_tokens':int(inputs['input_ids'].shape[-1]),'generated_tokens':int(output.shape[-1]-inputs['input_ids'].shape[-1]),'latency_s':round(latency,3),'peak_gpu_gb':round(torch.cuda.max_memory_allocated()/1e9,3),'raw_response':decoded,'parse_status':status,'parsed':obj}
 del inputs,output
 torch.cuda.empty_cache()
 return out
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--smoke',action='store_true');ap.add_argument('--case',default=None);ap.add_argument('--arm',default=None);ap.add_argument('--phase',choices=['global','window','fine'],default='global')
 args=ap.parse_args()
 assert (R/'weights_locked.json').exists(),'weights not locked'
 torch.manual_seed(2026)
 processor=AutoProcessor.from_pretrained(str(W),local_files_only=True)
 model=AutoModelForImageTextToText.from_pretrained(str(W),dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa',local_files_only=True).eval()
 selected=cases[:1] if args.smoke else [x for x in cases if args.case is None or x['case']==args.case]
 selectedarms=['A'] if args.smoke else ([args.arm] if args.arm else arms)
 target=R/('smoke' if args.smoke else ('responses_fine' if args.phase=='fine' else 'responses'))
 target.mkdir(exist_ok=True)
 for case in selected:
  n=case['frames']
  for arm in selectedarms:
   windows=[None] if args.smoke or args.phase=='global' else ([w['start'] for c in json.loads((R/'fine_windows.json').read_text())['cases'] if c['case']==case['case'] for w in c['windows']] if args.phase=='fine' else list(range(0,n,40)))
   for start in windows:
    kind='smoke' if args.smoke else args.phase
    outputpath=target/f"{case['case']}__{arm}__{kind}__{start if start is not None else 'all'}.json"
    if outputpath.exists():continue
    result=one(model,processor,case,arm,kind,start)
    outputpath.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'case':case['case'],'arm':arm,'kind':kind,'start':start,'parse':result['parse_status'],'intervals':len((result['parsed'] or {}).get('contact_intervals',[])),'latency_s':result['latency_s'],'peak_gpu_gb':result['peak_gpu_gb']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
