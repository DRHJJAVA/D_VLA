
import os
os.environ['USE_TF']='0';os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='4'
import pathlib,json,time,traceback,hashlib,h5py,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from transformers import AutoProcessor,AutoModelForImageTextToText
R=pathlib.Path('/mnt/nvme1/drh/D_VLA');RUN=R/'experiments/lcjp003/r2/v0_v1/v001'
OUT=R/'docs/LCJP003_R2_V01_assets';OUT.mkdir(exist_ok=True)
C=R/'experiments/contact_joint_object/cache'
CAMS=['agentview','robot0_eye_in_hand']
def save(n,d):
 p=RUN/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 manifest=json.loads((R/'experiments/contact_joint_object/manifests/episodes.json').read_text())['episodes']
 rows=sorted([v for v in manifest if v['episode'] in ['demo_45','demo_46']],key=lambda z:(z['task'],z['episode']))
 assert len(rows)==20 and len(set(v['task'] for v in rows))==10
 cases=[]
 for row in rows:
  assert row['split']=='dev'
  for t in np.linspace(0,row['frames']-1,5,dtype=int):
   cases.append({'id':row['id'],'task':row['task'],'demo':row['episode'],'frame':int(t),'frames':row['frames'],'target':row['task'].removeprefix('pick_up_the_').split('_and_place_it_')[0].replace('_',' ')})
 save('v1_sampling_lock.json',{'cases':cases,'total_frames':100,'views':2,'seed':20261027,'selection':'all10 tasks dev45/46 each5 equally spaced frames; frozen before inference','no_heldout':True})
 weights=R/'experiments/lcjp003/weights/smolvlm2'
 save('v1_state.json',{'stage':'loading_frozen_vlm','completed':0,'total':100,'started':time.time()})
 processor=AutoProcessor.from_pretrained(weights,local_files_only=True)
 processor.tokenizer.padding_side='left'
 if hasattr(processor.image_processor,'do_image_splitting'):processor.image_processor.do_image_splitting=False
 model=AutoModelForImageTextToText.from_pretrained(weights,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').eval().to('cuda')
 for p in model.parameters():p.requires_grad_(False)
 yes=processor.tokenizer.encode('Yes',add_special_tokens=False);no=processor.tokenizer.encode('No',add_special_tokens=False)
 assert len(yes)==len(no)==1
 def query(entries):
  answer=[]
  for start in range(0,len(entries),8):
   part=entries[start:start+8];texts=[];ims=[]
   for full,crop,phrase in part:
    prompt='The first image marks one candidate region in red. The second image shows that candidate cropped out with a gray background. Consider only that candidate, not other objects in the scene. Is the candidate '+phrase+'? Answer Yes or No.'
    msgs=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':prompt}]}]
    texts.append(processor.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False));ims.append([full,crop])
   inp=processor(text=texts,images=ims,padding=True,return_tensors='pt').to('cuda')
   with torch.inference_mode():
    logits=model(**inp,logits_to_keep=1,use_cache=False).logits[:,-1]
    probabilities=logits[:,[yes[0],no[0]]].float().softmax(-1)[:,0].cpu().tolist()
   answer.extend(probabilities)
  return answer
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
 allrows=[];checks=[];times=[];started=time.time()
 for index,case in enumerate(cases):
  t0=time.time();eid=case['id'];t=case['frame'];items=[];raws=[];queries=[];mapping=[]
  with h5py.File(C/'rgbd'/f'{eid}.h5','r') as raw,h5py.File(C/'observations'/f'{eid}.h5','r') as obs:
   frame=obs[f'frames/{t}']
   for cam in CAMS:raws.append(raw[f'{cam}/rgb'][t])
   for name in sorted(k for k in frame if k.startswith('object_')):
    g=frame[name];obj={'id':int(name.split('_')[1]),'points':int(len(g['xyz'])),'quality':float(g.attrs.get('sam_quality',float('nan'))),'views':[]}
    for vi,rgb in enumerate(raws):
     if f'mask_{vi}' not in g:continue
     mask=np.unpackbits(g[f'mask_{vi}'][:])[:rgb.shape[0]*rgb.shape[1]].reshape(rgb.shape[:2]).astype(bool)
     if not mask.any():continue
     ys,xs=np.nonzero(mask);box=(max(0,int(xs.min())-3),max(0,int(ys.min())-3),min(rgb.shape[1],int(xs.max())+4),min(rgb.shape[0],int(ys.max())+4))
     full=Image.fromarray(rgb.copy());dr=ImageDraw.Draw(full);dr.rectangle(box,outline=(255,30,30),width=3)
     cut=rgb.copy();cut[~mask]=128;crop=Image.fromarray(cut).crop(box)
     obj['views'].append({'camera':vi,'area':int(mask.sum()),'bbox':list(box),'mask':mask,'full':full,'crop':crop})
     for role,phrase in [('target',case['target']),('goal','a basket')]:
      queries.append((full,crop,phrase));mapping.append((len(items),len(obj['views'])-1,role))
    items.append(obj)
   values=query(queries)
   for (oi,vi,role),v in zip(mapping,values):items[oi]['views'][vi][role]=v
   for o in items:
    o['target_prior']=float(np.mean([v['target'] for v in o['views']])) if o['views'] else None
    o['goal_prior']=float(np.mean([v['goal'] for v in o['views']])) if o['views'] else None
   usable=[o for o in items if o['target_prior'] is not None]
   best=max(usable,key=lambda x:x['target_prior']) if usable else None
   goal=max(usable,key=lambda x:x['goal_prior']) if usable else None
   if index%5==0 and best:
    v=best['views'][0];rgb=raws[v['camera']].copy();rgb[v['mask']]=128
    blankcrop=Image.new('RGB',v['crop'].size,(128,128,128));blank=query([(Image.fromarray(rgb),blankcrop,case['target'])])[0]
    checks.append({'case':index,'candidate':best['id'],'original':v['target'],'erased_region':blank,'absolute_delta':abs(v['target']-blank)})
   imagepaths=[]
   for vi,rgb in enumerate(raws):
    base=Image.fromarray(rgb.copy()).resize((512,512),Image.Resampling.NEAREST)
    arr=np.asarray(base).copy()
    for o in items:
     for v in o['views']:
      if v['camera']!=vi:continue
      mask=np.repeat(np.repeat(v['mask'],2,axis=0),2,axis=1)
      color=np.array((240,60,60) if best and o['id']==best['id'] else (40,130,230))
      arr[mask]=(arr[mask]*.76+color*.24).astype(np.uint8)
    im=Image.fromarray(arr);dr=ImageDraw.Draw(im)
    for o in items:
     for v in o['views']:
      if v['camera']!=vi:continue
      box=tuple(int(z*2) for z in v['bbox'])
      dr.rectangle(box,outline=(220,20,20) if best and o['id']==best['id'] else (40,120,220),width=2)
      label=f"#{o['id']} {v['target']:.2f}";dr.rectangle((box[0],box[1],min(512,box[0]+100),min(512,box[1]+22)),fill='white');dr.text((box[0]+2,box[1]),label,font=font,fill='black')
    filename=f'case_{index:03d}_view{vi}.jpg';im.save(OUT/filename,quality=92)
    rawname=f'case_{index:03d}_raw{vi}.jpg';Image.fromarray(rgb).resize((512,512)).save(OUT/rawname,quality=92)
    imagepaths.append({'overlay':filename,'raw':rawname})
   serial=[]
   for o in items:
    serial.append({k:v for k,v in o.items() if k!='views'}|{'views':[{k:v for k,v in q.items() if k not in ['mask','full','crop']} for q in o['views']]})
   row=case|{'index':index,'candidates':serial,'top_target':best['id'] if best else None,'top_goal':goal['id'] if goal else None,'images':imagepaths,'manual_target_coverage':'not_labeled','trained_target_gate':False}
   allrows.append(row);times.append(time.time()-t0)
   save('v1_results_partial.json',{'cases':allrows,'region_response_checks':checks})
   save('v1_state.json',{'stage':'frozen_prior_inference','completed':index+1,'total':100,'seconds':time.time()-started,'last_case_seconds':times[-1],'median_case_seconds':float(np.median(times)),'optimizer_updates':0})
 summary={'case_count':100,'camera_frames':200,'tasks':10,'episodes':20,'candidate_count':sum(len(c['candidates']) for c in allrows),'per_case_candidates':[len(c['candidates']) for c in allrows],'region_response_count':len(checks),'region_response_abs_mean':float(np.mean([x['absolute_delta'] for x in checks])),'region_response_abs_max':float(np.max([x['absolute_delta'] for x in checks])),'prior_definition':'next-token Yes vs No conditional likelihood normalized over the two tokens; not calibrated object probability','semantic_prior_accuracy':'not measured without independent object annotations','candidate_target_coverage':'requires visual review; no simulation object GT used','elapsed_seconds':time.time()-started,'model':'SmolVLM2-500M-Video-Instruct frozen','optimizer_updates':0,'heldout_read':False}
 save('v1_result.json',{'summary':summary,'cases':allrows,'region_response_checks':checks})
 (OUT/'v1_results.json').write_text(json.dumps({'summary':summary,'cases':allrows,'region_response_checks':checks},ensure_ascii=False))
 save('v1_complete.json',summary);save('v1_state.json',{'stage':'complete_pending_visual_review','completed':100,'total':100,'optimizer_updates':0})
if __name__=='__main__':
 try:main()
 except Exception as e:
  save('v1_failure.json',{'type':type(e).__name__,'summary':str(e).splitlines()[0][:400],'frames':[{'file':pathlib.Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]})
