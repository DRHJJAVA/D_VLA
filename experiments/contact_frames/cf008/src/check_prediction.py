import pathlib,json,importlib.util,numpy as np,torch,hashlib
R=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_frames/cf008')
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
s=importlib.util.spec_from_file_location('m8',R/'src/model.py');m8=importlib.util.module_from_spec(s);s.loader.exec_module(m8);rows=[]
for ep in ['tomato_sauce_demo_3','alphabet_soup_demo_1','alphabet_soup_demo_0','porcelain_mug_demo_9']:
 p=R/'evaluation'/ep/'artifacts';d=np.load(p/'dataset.npz');z=np.load(p/'frame_predictions.npz');ix=z['context_indices']
 for j,seed in enumerate([2026,2027,2028]):
  ck=torch.load(R/'runs/evidence'/('seed_'+str(seed))/'last.pt',map_location='cpu',weights_only=False);m=m8.ContactFrameNet(50).cuda().eval();m.load_state_dict(ck['model'])
  st=(d['states']-ck['state_mean'])/ck['state_std'];prob=[];near=[]
  with torch.no_grad():
   for t in range(0,len(ix),16):
    k=ix[t:t+16];e,aux,n=m(torch.as_tensor(d['images'][k],device='cuda'),torch.as_tensor(d['points'][k],device='cuda'),torch.as_tensor(st[k],device='cuda'))
    prob.append(torch.sigmoid(e).cpu().numpy());near.append(torch.sigmoid(n).cpu().numpy())
  err=float(np.max(np.abs(np.concatenate(prob)-z['probabilities'][j])));errn=float(np.max(np.abs(np.concatenate(near)-z['proximity_probabilities'][j])))
  assert max(err,errn)<2e-6;rows.append({'episode':ep,'seed':seed,'frames':len(ix),'effective_max_diff':err,'near_max_diff':errn})
  del m;torch.cuda.empty_cache()
(R/'audits/independent_prediction.json').write_text(json.dumps({'passed':True,'rows':rows},indent=2));print(json.dumps(rows))
