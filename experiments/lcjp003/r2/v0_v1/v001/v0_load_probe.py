
import os
os.environ['USE_TF']='0';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='4'
import pathlib,json,traceback,importlib.util,time,torch
from safetensors.torch import load_file
R=pathlib.Path('/mnt/nvme1/drh/D_VLA');RUN=R/'experiments/lcjp003/r2/v0_v1/v001'
def save(n,d):(RUN/n).write_text(json.dumps(d,indent=2))
def main():
 torch.set_num_threads(4);torch.manual_seed(20261027)
 cfg=json.loads((R/'experiments/lcjp003/weights/smolvla_base/config.json').read_text())
 sp=importlib.util.spec_from_file_location('official_expert',RUN/'smolvlm_with_expert_official.py');mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
 model=mod.SmolVLMWithExpertModel(model_id=str(R/'experiments/lcjp003/weights/smolvlm2'),load_vlm_weights=False,train_expert_only=True,freeze_vision_encoder=True,attention_mode=cfg['attention_mode'],num_expert_layers=cfg['num_expert_layers'],num_vlm_layers=cfg['num_vlm_layers'],self_attn_every_n_layers=cfg['self_attn_every_n_layers'],expert_width_multiplier=cfg['expert_width_multiplier'],device='cpu')
 weights=load_file(R/'experiments/lcjp003/weights/smolvla_base/model.safetensors',device='cpu')
 prefix='model.vlm_with_expert.'
 state={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}
 shapes={k:list(v.shape) for k,v in model.state_dict().items()}
 missing,unexpected=model.load_state_dict(state,strict=False)
 exact=[k for k in state if k in model.state_dict() and torch.equal(state[k].float(),model.state_dict()[k].float())]
 summary={'model_type':type(model).__name__,'loaded_keys':len(state),'exact_keys':len(exact),'missing':missing,'unexpected':unexpected,'expert_hidden':model.expert_hidden_size,'vlm_hidden':model.config.text_config.hidden_size,'attention_heads':model.num_attention_heads,'kv_heads':model.num_key_value_heads,'expert_layers':len(model.lm_expert.layers),'expert_parameters':sum(p.numel() for p in model.lm_expert.parameters()),'source_version':'lerobot v0.4.4 official standalone module','base_action_semantics':'SO100; LIBERO projections must be adapted','source_initialization':'loaded smolvla_base tensors, no optimizer'}
 save('v0_weight_loading.json',summary);save('v0_weight_tensor_shapes.json',shapes)
 assert not unexpected and len(exact)==len(state)
 assert not [k for k in missing if k.startswith('lm_expert.')]
 for p in model.parameters():p.requires_grad_(False)
 model = model.float().to('cuda'); model.eval()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 a=torch.randn(1,40,summary['vlm_hidden'],device='cuda',requires_grad=True)
 b=torch.randn(1,16,summary['expert_hidden'],device='cuda',requires_grad=True)
 attn=torch.ones(1,56,56,dtype=torch.bool,device='cuda');attn[:,:40,40:]=False
 pos=torch.arange(56,device='cuda')[None]
 outputs,cache=model(attention_mask=attn,position_ids=pos,inputs_embeds=[a,b],use_cache=False,fill_kv_cache=False)
 y=outputs[1];y.square().mean().backward()
 save('v0_expert_interface.json',{'prefix':[1,40,summary['vlm_hidden']],'suffix':[1,16,summary['expert_hidden']],'output':list(y.shape),'finite':bool(torch.isfinite(y).all()),'prefix_grad_norm':float(a.grad.norm()),'suffix_grad_norm':float(b.grad.norm()),'optimizer_updates':0,'parameters_changed':False,'checkpoint_loaded':True})
if __name__=='__main__':
 try:main()
 except Exception as e:save('v0_load_failure.json',{'type':type(e).__name__,'summary':str(e).splitlines()[0][:400],'frames':[{'file':pathlib.Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]})
