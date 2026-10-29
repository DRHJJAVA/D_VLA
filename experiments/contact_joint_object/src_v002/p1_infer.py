import os
os.environ['USE_TF']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import json,pathlib,time,datetime,hashlib,traceback,math
import numpy as np,torch
from PIL import Image
from transformers import SiglipModel,SiglipImageProcessor,GemmaTokenizer
R=pathlib.Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/contact_joint_object'
RUN=B/'runs/v002/preflight/P1_semantics_v1';ASSET=R/'docs/lcjp002_p1_assets';MODEL=B/'dependencies/siglip2-base-patch16-224'
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def save(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False));q.replace(p)
def main():
 torch.set_num_threads(4);torch.manual_seed(2026)
 data=json.loads((RUN/'prepared.json').read_text());protocol=data['protocol']
 state=json.loads((MODEL/'prepare_state.json').read_text())
 assert state['files']['model.safetensors']['status']=='complete'
 assert state['files']['model.safetensors']['verified_lfs']
 model=SiglipModel.from_pretrained(str(MODEL),local_files_only=True).eval().to('cuda')
 processor=SiglipImageProcessor.from_pretrained(str(MODEL),local_files_only=True)
 tokenizer=GemmaTokenizer.from_pretrained(str(MODEL),local_files_only=True)
 labels=sorted({t['entity'] for t in data['tasks']})+['basket','table','robot gripper','background']
 prompts=['a photo of '+label+'.' for label in labels]
 texts=tokenizer(prompts,padding='max_length',max_length=64,truncation=True,return_tensors='pt')
 with torch.inference_mode():
  te=model.get_text_features(input_ids=texts['input_ids'].to('cuda'));te=te/te.norm(dim=-1,keepdim=True);scale=model.logit_scale.exp();bias=model.logit_bias
 pairs=[]
 for task in data['tasks']:
  for frame in task['frames']:
   for view in frame['views']:
    for candidate in view['candidates']:pairs.append((task,frame,view,candidate))
 scores=[];start=time.monotonic()
 for k in range(0,len(pairs),32):
  part=pairs[k:k+32];images=[Image.open(R/'docs'/x[3]['crop']).convert('RGB') for x in part]
  pixels=processor(images=images,return_tensors='pt')['pixel_values'].to('cuda')
  with torch.inference_mode():
   im=model.get_image_features(pixel_values=pixels);im=im/im.norm(dim=-1,keepdim=True);logits=(im@te.T*scale+bias).float().cpu().numpy()
  assert np.isfinite(logits).all();scores.append(logits)
  for (task,frame,view,c),row in zip(part,logits):
   c['entity_logits']={label:round(float(v),6) for label,v in zip(labels,row)}
   c['target_logit']=c['entity_logits'][task['entity']]
   order=np.argsort(-row);c['semantic_top']=[{'label':labels[i],'logit':round(float(row[i]),6)} for i in order[:4]]
  save(RUN/'inference_progress.json',{'time':now(),'candidates':min(k+32,len(pairs)),'total':len(pairs),'seconds':round(time.monotonic()-start,2)})
 allscores=np.concatenate(scores);np.savez_compressed(RUN/'semantic_logits.npz',logits=allscores,labels=np.array(labels))
 accepted=0;unknown=0;taskstats=[]
 for task in data['tasks']:
  ts={'id':task['id'],'name':task['name'],'views':0,'candidate':0,'unknown':0,'top_ids':{}}
  for frame in task['frames']:
   for view in frame['views']:
    view['entity_matches']={}
    for entity in task['entities']:
     label=entity['name'];candidates=sorted(view['candidates'],key=lambda c:(-c['entity_logits'][label],c['id']))
     if not candidates:
      match={'status':'unknown','reason':'没有分割候选','top_id':None,'runner_up_id':None,'top_logit':None,'margin':None}
     else:
      c=candidates[0];runner=candidates[1] if len(candidates)>1 else None
      margin=c['entity_logits'][label]-runner['entity_logits'][label] if runner else None
      reasons=[]
      if c['mask_area']<16:reasons.append('轮廓过小')
      if c['valid_depth_points']/c['mask_area']<.5:reasons.append('有效深度不足')
      if c['semantic_top'][0]['label']!=label:reasons.append('该候选更像'+c['semantic_top'][0]['label'])
      if margin is not None and margin<1:reasons.append('第一名与第二名接近')
      match={'status':'unknown' if reasons else 'candidate','reason':'；'.join(reasons) if reasons else '语义匹配已选出候选，等待画面核对','top_id':c['id'],'runner_up_id':runner['id'] if runner else None,'top_logit':c['entity_logits'][label],'margin':round(margin,6) if margin is not None else None}
     view['entity_matches'][label]=match
    view['match']=view['entity_matches'][task['entity']]
    ordered=sorted(view['candidates'],key=lambda c:(-c['target_logit'],c['id']))
    for i,c in enumerate(ordered):c['rank']=i+1
    if view['match']['status']=='candidate':accepted+=1;ts['candidate']+=1
    else:unknown+=1;ts['unknown']+=1
    ts['views']+=1;ts['top_ids'].setdefault(view['camera'],[]).append(view['match']['top_id'])
  taskstats.append(ts)
 data['summary'].update({'accepted':accepted,'unknown':unknown,'audit_correct':0,'audit_wrong':0,'audit_unknown':0,'audit_unreviewed':data['summary']['views']})
 data['protocol']['notes'] += ['Candidate status is not a pass; human image audit is listed separately.','No matching adapter was trained; this is a frozen pretrained matching baseline.']
 data['protocol']['model_sha256']=state['files']['model.safetensors']['sha256']
 data['protocol']['text_labels']=labels
 data['created']=now()
 save(RUN/'results_before_audit.json',data);save(ASSET/'results.json',data)
 save(RUN/'inference_complete.json',{'time':now(),'seconds':round(time.monotonic()-start,3),'summary':data['summary'],'task_summary':taskstats,'model_sha256':state['files']['model.safetensors']['sha256'],'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'heldout_read':False,'training_started':False})
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(RUN/'inference_failure.json',{'type':type(e).__name__,'locations':[{'file':pathlib.Path(v.filename).name,'line':v.lineno} for v in traceback.extract_tb(e.__traceback__)]});raise SystemExit(1)
