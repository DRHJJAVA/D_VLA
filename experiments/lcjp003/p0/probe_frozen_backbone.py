
import pathlib,json,os,time,datetime
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003')
state_path=B/'p0/frozen_backbone_probe.json'
def save(d):state_path.write_text(json.dumps(d,indent=2))
save({'status':'waiting_for_weight_download','pid':os.getpid()})
try:
 for _ in range(120):
  w=json.loads((B/'p0/weights_state.json').read_text())
  if w['smolvlm2']['status']=='downloaded':break
  if w['smolvlm2']['status']=='download_failed':raise RuntimeError('WeightDownloadFailed')
  time.sleep(2)
 else:raise TimeoutError('WeightWaitTimeout')
 import torch
 from transformers import AutoProcessor,AutoModelForImageTextToText
 torch.set_num_threads(2)
 path=w['smolvlm2']['path']
 processor=AutoProcessor.from_pretrained(path,local_files_only=True)
 model=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
 model.requires_grad_(False)
 core=model.model
 tokenized=processor.tokenizer('pick up the object',return_tensors='pt')
 with torch.inference_mode():
  txt=core.text_model(**tokenized).last_hidden_state
  vis=core.vision_model(pixel_values=torch.zeros(1,3,512,512)).last_hidden_state
 r={'status':'ready','scope':'engineering_import_and_tensor_interface_probe','checkpoint':path,'model_class':type(model).__name__,'vision_class':type(core.vision_model).__name__,'text_class':type(core.text_model).__name__,'text_shape':list(txt.shape),'vision_shape':list(vis.shape),'finite_text':bool(torch.isfinite(txt).all()),'finite_vision':bool(torch.isfinite(vis).all()),'parameter_count':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'config':{'vision_hidden':model.config.vision_config.hidden_size,'patch_size':model.config.vision_config.patch_size,'text_hidden':model.config.text_config.hidden_size,'text_layers':model.config.text_config.num_hidden_layers,'text_q_heads':model.config.text_config.num_attention_heads,'text_kv_heads':model.config.text_config.num_key_value_heads,'head_dim':model.config.text_config.head_dim}}
 save(r);print(json.dumps(r))
except Exception as e:
 import traceback
 save({'status':'failed','exception_type':type(e).__name__,'frames':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
 print(json.dumps({'status':'failed','exception_type':type(e).__name__}))
