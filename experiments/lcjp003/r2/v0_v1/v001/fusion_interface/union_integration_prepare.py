import os,sys,json,time,hashlib,traceback,ast,copy
from pathlib import Path
ROOT=Path('/mnt/nvme1/drh/D_VLA')
HERE=Path(__file__).resolve().parent
OLD=HERE.parent
os.chdir(ROOT)
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'experiments/lcjp003/src'))
import numpy as np,h5py,torch
import data as p0data
from fusion_module import LEGACY,FusionFront,parameter_digest
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(n,d): (HERE/n).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def main():
 start=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(20261027);np.random.seed(20261027)
 case=json.loads((OLD/'v1_sampling_lock.json').read_text())['cases'][2]
 assert case['demo']=='demo_45' and case['frame']==74 and 'alphabet_soup' in case['id']
 t=case['frame'];eid=case['id'];base=ROOT/'experiments/contact_joint_object'
 rows=json.loads((base/'manifests/episodes.json').read_text())['episodes']
 e=next(r.copy() for r in rows if r['id']==eid);assert e['split']=='dev'
 e['source']=str(ROOT/e['raw']);obs=base/'cache/observations'/(eid+'.h5');rgb=base/'cache/rgbd'/(eid+'.h5')
 derived=HERE/'union_integration_case2_observations.h5';input_hashes={};meta=[]
 with h5py.File(obs,'r') as src,h5py.File(derived,'w') as out,h5py.File(rgb,'r') as r:
  fg=out.create_group(f'frames/{t}');og=src[f'frames/{t}']
  for c,name in enumerate(['agentview','robot0_eye_in_hand']):
   a=og['features_'+str(c)][:];fg.create_dataset('features_'+str(c),data=a);input_hashes['dense_DINO_view'+str(c)]=ahash(a)
   assert a.shape==(32,32,384)
   for k in ['rgb','depth_m','K','camera_to_world','robot_mask']:
    input_hashes[name+'/'+k]=ahash(r[name+'/'+k][t])
   maskfile=OLD/f'segmentation_diagnostic/m2m_all/case002_view{c}_union.npz';input_hashes[maskfile.name]=digest(maskfile)
   with np.load(maskfile) as masks: mm=masks['masks']
   assert mm.dtype==bool and mm.shape[1:]==(256,256)
   for mindex,m in enumerate(mm):
    v,u=np.nonzero(m);uv=np.stack([u,v],axis=1).astype(np.int64)
    assert len(uv)>0
    oi=len(meta);g=fg.create_group('object_'+str(oi))
    g.create_dataset('uv',data=uv);g.create_dataset('camera',data=np.full(len(uv),c,dtype=np.int8));g.create_dataset('mask_'+str(c),data=np.packbits(m))
    meta.append({'candidate':oi,'view':c,'sam_index':mindex,'pixels':len(uv)})
 with h5py.File(e['source'],'r') as f:
  actions=f['data/demo_45/actions'][:];assert len(actions)==case['frames']
 input_hashes['true_action_targets_74_89']=ahash(actions[t:t+16])
 inherited=json.loads((HERE/'inherited_action_normalization.json').read_text())
 assert inherited['development_used'] is False
 mu=np.array(inherited['mean'],np.float32);sd=np.array(inherited['std'],np.float32)
 d=p0data.P0Dataset()
 d.rows={eid:e};d.actions={eid:actions};d._cache.clear();d.read_log=[]
 original_factory=p0data.P0Dataset;original_file=h5py.File
 access=[]
 def checked_file(path,*args,**kwargs):
  mode=args[0] if args else kwargs.get('mode','r')
  resolved=Path(path).resolve() if isinstance(path,(str,Path)) else None
  if resolved==obs.resolve():
   assert mode=='r';access.append('new_segmentation_observation');return original_file(derived,*args,**kwargs)
  if resolved in [rgb.resolve(),Path(e['source']).resolve()]:
   assert mode=='r';access.append('original_readonly')
  return original_file(path,*args,**kwargs)
 p0data.P0Dataset=lambda:d;h5py.File=checked_file
 LEGACY['save']=lambda n,z:save('union_integration_'+n,z)
 try: obj,h,target,grip,idx=LEGACY['make_data'](e,t,mu,sd)
 finally: p0data.P0Dataset=original_factory;h5py.File=original_file
 assert len(obj)>0 and np.array_equal(np.asarray(idx),np.maximum(np.arange(t-15,t+1),0))
 x=torch.tensor(np.stack([o[1] for o in obj]),dtype=torch.float32)
 valid=torch.tensor(np.stack([o[2] for o in obj]),dtype=torch.bool)
 tree=ast.parse((OLD/'v0_corrected_check.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 class CPU(ast.NodeTransformer):
  def visit_Constant(self,node):
   if node.value=='cuda':return ast.copy_location(ast.Constant(value='cpu'),node)
   if node.value=='Pick up the bbq sauce and place it in the basket.':return ast.copy_location(ast.Constant(value='Pick up the alphabet soup and place it in the basket.'),node)
   return node
  def visit_Attribute(self,node):
   self.generic_visit(node)
   if node.attr=='cuda':node.attr='cpu'
   return node
 ns=dict(LEGACY);setup=CPU().visit(ast.Module(body=copy.deepcopy([n for n in fn.body if 136<=n.lineno<=147]),type_ignores=[]));exec(compile(ast.fix_missing_locations(setup),'<official_alphabet_text>', 'exec'),ns)
 words=ns['words'].detach().cpu();assert words.ndim==2 and words.shape[-1]==960
 del ns
 front=FusionFront().eval();before=parameter_digest(front);ht=torch.tensor(h,dtype=torch.float32)
 with torch.no_grad():
  o=front.encode(x,valid,ht,words)
  padvalid=valid.clone();padvalid[:,:,-8:]=False
  xp=x.clone();xp[~padvalid]=1e6
  oa=front.encode(x,padvalid,ht,words);ob=front.encode(xp,padvalid,ht,words)
  padding=max(float((oa[k]-ob[k]).abs().max()) for k in ['prefix','scores','lambda'])
  missing=front.encode(x,torch.zeros_like(valid),ht,words)
 assert padding==0 and torch.isfinite(missing['prefix']).all() and missing['lambda'].abs().max()==0
 assert before==parameter_digest(front)
 inputs={'x':x,'valid':valid,'h':ht,'words':words,'target':torch.tensor(target,dtype=torch.float32),'grip':torch.tensor(grip,dtype=torch.long),'history_indices':np.asarray(idx).tolist(),'candidate_ids':[o[0] for o in obj],'episode':eid,'frame':t}
 torch.save(inputs,HERE/'union_integration_inputs.pt')
 input_hashes.update({'derived_observations':digest(derived),'input_tensor_bundle':digest(HERE/'union_integration_inputs.pt'),'normalization':digest(HERE/'inherited_action_normalization.json'),'source_sampling_lock':digest(OLD/'v1_sampling_lock.json'),'official_alphabet_word_embeddings':ahash(words.numpy()),'fusion_module':digest(HERE/'fusion_module.py'),'P0_geometry':digest(ROOT/'experiments/lcjp003/src/data.py'),'legacy_part_assembly':digest(OLD/'v0_corrected_check.py')})
 result={'status':'passed','optimizer_updates':0,'case':case,'new_mask_candidates_by_view':[sum(m['view']==i for m in meta) for i in range(2)],'new_mask_candidates_total':len(meta),'assembled_candidates':len(obj),'assembled_candidate_ids':[o[0] for o in obj],'candidate_pixel_counts':meta,'shapes':{k:list(v.shape) for k,v in inputs.items() if torch.is_tensor(v)},'output_shapes':{k:list(o[k].shape) for k in ['base','part','common','prefix','scores','lambda']},'true_feature_protocol':'original same-frame full-image frozen DINO maps, original corrected RGBD backprojection/visibility_v3 and original part patch assembly','view_specific':True,'invented_identity_merge':False,'invented_visual_features':False,'semantic_token_source':'fresh official frozen word embeddings of the exact alphabet soup instruction','semantic_text':'Pick up the alphabet soup and place it in the basket.','history_indices':np.asarray(idx).tolist(),'future_observation_read':False,'raw_actions_for_loss_only':True,'read_log':d.read_log,'padding_perturbation_max':padding,'all_parts_missing_finite':True,'all_parts_missing_effective_lambda':0.0,'parameter_hash_before':before,'parameter_hash_after':parameter_digest(front),'input_hashes':input_hashes,'elapsed_seconds':time.perf_counter()-start,'input_redirect_only':str(obs),'input_redirect_destination':str(derived)}
 save('union_integration_prepare_result.json',result)
 print(json.dumps({'status':'prepared','candidate_count':len(obj),'shape':list(x.shape),'padding_max':padding,'elapsed':result['elapsed_seconds']}))
try:main()
except Exception as e:
 save('union_integration_failure.json',{'type':type(e).__name__,'message':str(e)[:250],'locations':[{'file':s.filename,'line':s.lineno} for s in traceback.extract_tb(e.__traceback__)]})
 print(json.dumps({'status':'failed','type':type(e).__name__}))
 raise SystemExit(1)
