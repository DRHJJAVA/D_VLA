import pathlib,sys,json,hashlib,datetime,time,traceback,contextlib,io,subprocess
root=pathlib.Path('/mnt/nvme1/drh/D_VLA')
r=root/'experiments/lcjp003';out=r/'p0';repo=root/'third_party/dinov3_lcjp003'
w=r/'weights/dinov3_vits16plus/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
report={'model':'dinov3_vits16plus','source':'Meta official local repository','loading_entry':'dinov3.hub.backbones.dinov3_vits16plus','hubconf_optional_eval_import':'bypassed via official backbone factory; source unchanged','device':'cpu','optimizer_updates':0,'input_scope':'existing P0 sample current frame only'}
log=io.StringIO()
try:
 with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  import numpy as np,torch
  torch.set_num_threads(2)
  lock=json.loads((out/'dinov3_source_lock.json').read_text())
  commit=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
  assert commit==lock['commit']
  sha=hashlib.sha256(w.read_bytes()).hexdigest()
  assert sha=='4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea'
  load_start=time.monotonic()
  sys.path.insert(0,str(repo))
  from dinov3.hub.backbones import dinov3_vits16plus
  model=dinov3_vits16plus(pretrained=False)
  weights=torch.load(str(w),map_location='cpu',weights_only=True)
  loaded=model.load_state_dict(weights,strict=True)
  model.eval().requires_grad_(False)
  sample=out/'samples/pick_up_the_bbq_sauce_and_place_it_in_the_basket__demo_0_frame0015.npz'
  with np.load(sample,allow_pickle=False) as z:
   rgb=z['rgb'][-1].copy();frame=int(z['frame_indices'][-1])
  x=torch.from_numpy(rgb).permute(0,3,1,2).float()/255.0
  mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
  x=(x-mean)/std
  forward_start=time.monotonic()
  with torch.inference_mode():
   f=model.forward_features(x);f2=model.forward_features(x)
  patch=f['x_norm_patchtokens'];cls=f['x_norm_clstoken'];storage=f['x_storage_tokens'] if 'x_storage_tokens' in f else f.get('x_norm_regtokens',None)
  tensor_schema={k:{'shape':list(v.shape),'dtype':str(v.dtype),'finite':bool(torch.isfinite(v).all()),'requires_grad':v.requires_grad} for k,v in f.items() if isinstance(v,torch.Tensor)}
  assert patch.shape==(2,256,384) and bool(torch.isfinite(patch).all())
  diff=float((patch-f2['x_norm_patchtokens']).abs().max())
  assert diff==0
  count=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
  assert trainable==0
  assert all(p.grad is None for p in model.parameters())
  report.update({'status':'passed','repository_commit':commit,'weights_file':str(w),'weights_sha256':sha,'weights_bytes':w.stat().st_size,'strict_load_missing_keys':len(loaded.missing_keys),'strict_load_unexpected_keys':len(loaded.unexpected_keys),'model_parameters':count,'trainable_parameters':trainable,'embed_dim':int(model.embed_dim),'patch_size':16,'depth':len(model.blocks),'storage_token_count':int(model.n_storage_tokens),'input_sample':sample.name,'frame_index':frame,'input_shape':list(x.shape),'rgb_uint8_range':[int(rgb.min()),int(rgb.max())],'normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225],'resize':'none: existing 256 by 256 RGB'},'outputs':tensor_schema,'patch_grid':[16,16],'feature_dimension':int(patch.shape[-1]),'patch_statistics':{'mean':float(patch.mean()),'std':float(patch.std()),'min':float(patch.min()),'max':float(patch.max())},'repeat_forward_max_abs_diff':diff,'load_seconds':forward_start-load_start,'two_forward_seconds':time.monotonic()-forward_start,'all_parameter_grads_none':True,'all_output_tensors_detached':all(not v.requires_grad for v in f.values() if isinstance(v,torch.Tensor))})
except Exception as e:
 report.update({'status':'failed','exception_type':type(e).__name__,'exception_summary':str(e)[:160],'sites':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
(out/'dinov3_probe.log').write_text(log.getvalue())
(out/'dinov3_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
p=out/'weights_state.json';d=json.loads(p.read_text());d['dinov3']['load_probe_status']=report['status'];d['dinov3']['load_probe']='p0/dinov3_probe.json'
if report['status']=='passed':d['dinov3'].update({'feature_dimension':report['feature_dimension'],'frozen_probe_verified':True})
tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
print(json.dumps(report))
