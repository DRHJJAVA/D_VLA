from adapters import *
import importlib.util, traceback, datetime
C=pathlib.Path(__file__).parent
P=C.parent
def save(n,d): (C/n).write_text(json.dumps(d,ensure_ascii=False,indent=2))
try:
 spec=importlib.util.spec_from_file_location('exec_base',C/'executor.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
 tree=ast.parse((C/'executor.py').read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
 body=[n for n in run.body if n.lineno<=30 and n.lineno!=27]
 body.append(ast.Return(value=ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])))
 fun=ast.FunctionDef(name='boot',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
 mod=ast.Module(body=[fun],type_ignores=[]);ast.fix_missing_locations(mod);ns=dict(e.__dict__);exec(compile(mod,'<diagnostic_boot>','exec'),ns)
 env=ns['boot']();model=env['model'];encoder=env['encoder'];device=env['device']
 ref=Dataset(cached=True)
 norm=json.loads((P/'inputs.json').read_text())['normalization'];std=np.asarray(norm['std']);mean=np.asarray(norm['mean'])
 episodes=json.loads((P/'evaluation_lock.json').read_text())['closed_loop']['episodes'][:3]
 root=C/('action_history_sensitivity_'+str(time.time_ns()));root.mkdir()
 rows=[];raw_all=[];first_all=[];norm_all=[];flips=0;repeat_max=0.0
 from torch.nn.attention import sdpa_kernel,SDPBackend
 def stable_sam(proc):
  original=proc.proc.sam.generate
  def call(*args,**kwargs):
   with sdpa_kernel(SDPBackend.MATH):return original(*args,**kwargs)
  proc.proc.sam.generate=call
 def metrics(a,b):
  d=np.abs(a-b);raw=d*std
  return {'normalized_mae':float(d.mean()),'normalized_max':float(d.max()),'first_normalized_max':float(d[0,0,0].max()),'first_raw_abs_by_axis':raw[0,0,0].tolist(),'raw_mae_by_axis':raw.reshape(-1,7).mean(0).tolist(),'raw_max_by_axis':raw.reshape(-1,7).max(0).tolist()}
 for index,eid in enumerate(episodes):
  work=root/('demo'+str(index));work.mkdir()
  (work/'cache/rgbd').mkdir(parents=True)
  raw=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object/cache/rgbd')/(eid+'.h5')
  (work/'cache/rgbd'/(eid+'.h5')).symlink_to(raw)
  handle=h5py.File(raw,'r');render=types.SimpleNamespace(eid=eid,work=work,out=handle)
  proposals=LiveProposals(device=device);stable_sam(proposals)
  adapter=LiveInput(eid,work,capacity=32,device=device,encoder=encoder)
  for t in range(16):
   proposals.frame(render,t);fr=adapter.frame(t)
  live=adapter.batch(15);original_frames=adapter.frames.copy()
  for t in range(16):adapter.frames[t]=ref.frame(eid,t)
  cached=adapter.batch(15);rf=ref.frame(eid,15);adapter.frames=original_frames
  seeds=range(20261026,20261046) if index==0 else range(20261026,20261029)
  for seed in seeds:
   noise=torch.randn(1,1,16,7,generator=torch.Generator(device='cpu').manual_seed(seed)).to(device)
   with torch.inference_mode():
    a=model(cached,noise)['actions'].detach().cpu().numpy()
    b=model(live,noise)['actions'].detach().cpu().numpy()
    if index==0 and seed==20261026:
     repeat=model(live,noise)['actions'].detach().cpu().numpy();repeat_max=float(np.max(np.abs(repeat-b)))
   row={'episode':eid,'t':15,'seed':seed,**metrics(a,b)}
   aa=a*std+mean;bb=b*std+mean
   row['first_gripper_reference']=float(aa[0,0,0,6]);row['first_gripper_changed']=float(bb[0,0,0,6])
   row['confident_first_gripper_sign_flip']=bool(abs(aa[0,0,0,6])>.1 and np.sign(aa[0,0,0,6])!=np.sign(bb[0,0,0,6]))
   flips+=int(row['confident_first_gripper_sign_flip']);rows.append(row)
   raw_all.append(np.abs(a-b)*std);norm_all.append(np.abs(a-b));first_all.append(np.abs(a-b)[0,0,0]*std)
  np.savez_compressed(work/'frame_pair.npz',live_xyz=fr['xyz_base'],reference_xyz=rf['xyz_base'])
  handle.close();del adapter,proposals,live,cached;torch.cuda.empty_cache()
  save('action_history_sensitivity_progress.json',{'episodes_done':index+1,'pairs':len(rows)})
 d=np.concatenate([x.reshape(-1,7) for x in raw_all]);n=np.concatenate([x.reshape(-1,7) for x in norm_all]);first=np.stack(first_all)
 result={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'checkpoint_sha256':sha(P/'step3000.pt'),'episodes':3,'paired_predictions':len(rows),'same_input_repeat_max':repeat_max,'normalized_mae':float(n.mean()),'normalized_max':float(n.max()),'raw_mae_by_axis':d.mean(0).tolist(),'raw_max_by_axis':d.max(0).tolist(),'first_raw_max_by_axis':first.max(0).tolist(),'confident_gripper_sign_flips':flips,'rows':rows,'work':str(root),'scope':'3 training clips, observed frames0-15 only, evaluate t15; 20 seeds demo0 and3 each demo1/2; SAM math only; no optimizer'}
 save('action_history_sensitivity.json',result)
except Exception as ex:
 save('action_history_sensitivity_failure.json',{'type':type(ex).__name__,'message':str(ex)[:400],'frames':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(ex.__traceback__)]})
