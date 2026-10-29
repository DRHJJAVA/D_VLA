import pathlib,json,numpy as np,torch,h5py,hashlib
C=pathlib.Path(__file__).parent
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def summarize(rows,std):
 d=np.asarray(rows,dtype=np.float64)
 raw=d*std
 first=raw[:,:,0,:].reshape(-1,7) if raw.ndim==4 else raw.reshape(-1,7)
 return float(d.mean()),first
def qualify():
 p=json.loads((C/'action_consistency_protocol_v2.json').read_text())
 results={}
 for name in p['required_diagnostics']:
  x=json.loads((C/name).read_text())
  assert x['checkpoint_sha256']==p['weight_sha256']
  a=np.asarray([r['first_raw_abs_by_axis'] for r in x['rows']])
  pos=float(np.linalg.norm(a[:,:3]*.05,axis=1).max())
  rot=float(np.rad2deg(np.linalg.norm(a[:,3:6]*.5,axis=1)).max())
  flips=sum(np.sign(r['first_gripper_reference'])!=np.sign(r['first_gripper_changed']) for r in x['rows'])
  passed=bool(x['normalized_mae']<=p['normalized_sequence_mae_max'] and pos<=p['position_target_delta_m_max'] and rot<=p['rotation_vector_delta_deg_max'] and a[:,6].max()<=p['first_gripper_abs_max'] and flips==0 and x['same_input_repeat_max']<=p['same_input_prediction_repeat_max'] and np.isfinite(a).all())
  results[name]={'passed':passed,'normalized_mae':x['normalized_mae'],'position_target_delta_m_max':pos,'rotation_vector_delta_deg_max':rot,'gripper_sign_flips':int(flips),'sha256':digest(C/name)}
 repeat=json.loads((C/'sam_scoped_math_probe.json').read_text())
 stable=all(r['raw_logits_max_abs']==0 and r['binary_mask_xor']==0 for r in repeat[1:])
 result={'passed':all(v['passed'] for v in results.values()) and stable,'diagnostics':results,'sam_repeat_exact':stable,'protocol_sha256':digest(C/'action_consistency_protocol_v2.json')}
 (C/'action_compatibility_qualified.json').write_text(json.dumps(result,indent=2))
 assert result['passed'],'Action compatibility qualification failed'
 return p
def check_gate(model,adapter,ref,eid,work,device):
 p=qualify()
 old=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object/cache')
 raw_equal={}
 with h5py.File(old/'rgbd'/(eid+'.h5'),'r') as a,h5py.File(pathlib.Path(work)/'cache/rgbd'/(eid+'.h5'),'r') as b:
  def visit(name,obj):
   if isinstance(obj,h5py.Dataset):
    aa=obj[...] if name.startswith('finger_surface/') else obj[0]
    bb=b[name][...] if name.startswith('finger_surface/') else b[name][0]
    raw_equal[name]=bool(np.array_equal(aa,bb,equal_nan=True))
  a.visititems(visit)
 feature_equal={}
 with h5py.File(old/'observations'/(eid+'.h5'),'r') as a,h5py.File(pathlib.Path(work)/'cache/observations'/(eid+'.h5'),'r') as b:
  for name in ['features_0','features_1']:feature_equal[name]=bool(np.array_equal(a['frames']['0'][name][...],b['frames']['0'][name][...],equal_nan=True))
 live=adapter.batch(0);saved=adapter.frames[0];adapter.frames[0]=ref.frame(eid,0);cached=adapter.batch(0);adapter.frames[0]=saved
 exact_state=bool(torch.equal(live['state'],cached['state']))
 exact_text=bool(torch.equal(live['text_features'],cached['text_features']) and torch.equal(live['text_valid'],cached['text_valid']))
 norm=json.loads((C.parent/'inputs.json').read_text())['normalization'];std=np.asarray(norm['std']);mean=np.asarray(norm['mean'])
 diffs=[];first=[];flips=0;repeat_max=0.0
 for seed in range(20261026,20261046):
  noise=torch.randn(1,1,16,7,generator=torch.Generator(device='cpu').manual_seed(seed)).to(device)
  with torch.inference_mode():
   a=model(cached,noise)['actions'].detach().cpu().numpy();b=model(live,noise)['actions'].detach().cpu().numpy()
   if seed==20261026:repeat_max=float(np.max(np.abs(b-model(live,noise)['actions'].detach().cpu().numpy())))
  assert np.isfinite(a).all() and np.isfinite(b).all()
  diffs.append(np.abs(a-b));first.append(np.abs(a-b)[0,0,0]*std)
  flips+=int(np.sign((a*std+mean)[0,0,0,6])!=np.sign((b*std+mean)[0,0,0,6]))
 first=np.stack(first)
 result={'mode':'controller_scaled_action_compatibility_v2','raw_fields_exact':raw_equal,'dino_features_exact':feature_equal,'state_exact':exact_state,'text_exact':exact_text,'paired_predictions':20,'normalized_sequence_mae':float(np.mean(diffs)),'first_position_target_delta_m_max':float(np.linalg.norm(first[:,:3]*.05,axis=1).max()),'first_rotation_vector_delta_deg_max':float(np.rad2deg(np.linalg.norm(first[:,3:6]*.5,axis=1)).max()),'first_gripper_abs_max':float(first[:,6].max()),'gripper_sign_flips':flips,'repeat_max':repeat_max,'protocol_sha256':digest(C/'action_consistency_protocol_v2.json')}
 result['passed']=bool(all(raw_equal.values()) and all(feature_equal.values()) and exact_state and exact_text and result['normalized_sequence_mae']<=p['normalized_sequence_mae_max'] and result['first_position_target_delta_m_max']<=p['position_target_delta_m_max'] and result['first_rotation_vector_delta_deg_max']<=p['rotation_vector_delta_deg_max'] and result['first_gripper_abs_max']<=p['first_gripper_abs_max'] and flips==0 and repeat_max<=p['same_input_prediction_repeat_max'])
 (C/'gpu_input_compatibility_gate.json').write_text(json.dumps(result,indent=2))
 assert result['passed'],'Controller-scaled action compatibility gate failed'
 return result
