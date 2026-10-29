import pathlib,json,importlib.util,numpy as np,torch
R=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf007')
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
spec=importlib.util.spec_from_file_location('cf007model',R/'src/model_all_train_frames.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
rows=[]
for ep in ['tomato_sauce_demo_3','alphabet_soup_demo_1']:
 p=R/'evaluation_all_train_frames'/ep/'artifacts';d=np.load(p/'dataset.npz');saved=np.load(p/'frame_predictions.npz');ix=saved['context_indices']
 for j,seed in enumerate([2026,2027,2028]):
  ck=torch.load(R/'runs/all_train_frames'/('seed_'+str(seed))/'last.pt',map_location='cpu',weights_only=False)
  m=mod.ContactFrameNet(50).cuda().eval();m.load_state_dict(ck['model'])
  states=(d['states']-np.asarray(ck['state_mean'],np.float32))/np.asarray(ck['state_std'],np.float32)
  result=[]
  with torch.no_grad():
   for a in range(0,len(ix),16):
    k=ix[a:a+16]
    logits,_=m(torch.as_tensor(d['images'][k],device='cuda'),torch.as_tensor(d['points'][k],device='cuda'),torch.as_tensor(states[k],device='cuda'))
    result.append(torch.sigmoid(logits).cpu().numpy())
  result=np.concatenate(result).reshape(-1)
  err=float(np.abs(result-saved['probabilities'][j]).max());assert err<2e-6
  rows.append({'episode':ep,'seed':seed,'frames':len(result),'maximum_score_difference':err})
  del m;torch.cuda.empty_cache()
(R/'audits/independent_prediction.json').write_text(json.dumps({'passed':True,'rows':rows},indent=2))
print(json.dumps(rows))
