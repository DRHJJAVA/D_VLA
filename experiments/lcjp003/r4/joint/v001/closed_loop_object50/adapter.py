import os,sys,pathlib,ast,types,copy,json,time,hashlib,traceback
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np, torch,h5py
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA')
RUN=ROOT/'experiments/lcjp003/r4/joint/v001'
OUT=RUN/'closed_loop_object50'
sys.path.insert(0,str(RUN/'src'))
import runtime
def stamp():
 import datetime
 return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(name,x):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def fail(stage,e):
 save('failure.json',{'stage':stage,'type':type(e).__name__,'time':stamp(),'frames':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
def module_nodes(p,nodes,name):
 m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),m.__dict__);return m
def render_module():
 p=ROOT/'experiments/lcjp003/p1/v001/closed_loop/adapters.py';t=ast.parse(p.read_text());body=[]
 for n in t.body:
  if isinstance(n,ast.Expr) and any(isinstance(a,ast.Call) and isinstance(a.func,ast.Attribute) and a.func.attr in ['set_num_threads','set_num_interop_threads'] for a in ast.walk(n)):continue
  if isinstance(n,ast.ImportFrom) and n.module in ['common','features','model']:continue
  if isinstance(n,ast.ClassDef) and n.name in ['LiveInput','LiveProposals']:continue
  body.append(n)
 return module_nodes(p,body,'r4_render')
def finger_function():
 p=ROOT/'experiments/lcjp003/r3/joint/v001/src/geometry.py';t=ast.parse(p.read_text())
 f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='load_objects');w=next(n for n in f.body if isinstance(n,ast.With))
 nodes=[copy.deepcopy(n) for n in w.body if 152<=n.lineno<=162]
 nodes+=ast.parse('return ee,rotation,fingers').body
 fn=ast.FunctionDef(name='finger_state',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='rgb'),ast.arg(arg='ix')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=nodes,decorator_list=[])
 ns={'np':np,'json':json}
 # Preserve the frozen original finger FPS implementation.
 fps=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='fps')
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fps,fn],type_ignores=[])),str(p),'exec'),ns)
 return ns['finger_state']
def history_function():
 p=ROOT/'experiments/lcjp003/r2/v0_v1/v001/v0_corrected_check.py';t=ast.parse(p.read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='make_data');w=next(n for n in f.body if isinstance(n,ast.With))
 nodes=[copy.deepcopy(n) for n in w.body if 36<=n.lineno<=38 or 74<=n.lineno<=77]
 nodes+=ast.parse('return np.asarray(hist,np.float32),hist_idx').body
 fn=ast.FunctionDef(name='history',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='rgb'),ast.arg(arg='t')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=nodes,decorator_list=[])
 ns={'np':np,'json':json};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(p),'exec'),ns);return ns['history']
class Front:
 def __init__(self):
  p=ROOT/'src/v3/stage2_extract_dino.py';t=ast.parse(p.read_text())
  nodes=[n for n in t.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign)) or isinstance(n,ast.FunctionDef) and n.name in ['model','preproc']]
  m=module_nodes(p,nodes,'r4_dino');self.preproc=m.preproc;self.dino=m.model()
  p=ROOT/'experiments/contact_joint_object/src/observations.py';t=ast.parse(p.read_text())
  node=next(n for n in ast.walk(t) if isinstance(n,ast.Assign) and n.lineno==67)
  self.feature_code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(p),'exec')
  self.finger=finger_function();self.history=history_function()
  p=RUN/'src/data.py';t=ast.parse(p.read_text());f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='get_frame')
  node=next(n for n in f.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='inputs' for a in n.targets))
  self.input_code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(p),'exec')
 def feature(self,im):
  ns={'self':types.SimpleNamespace(dino=self.dino),'preproc':self.preproc,'im':im,'torch':torch,'np':np}
  old=(torch.backends.cudnn.allow_tf32,torch.backends.cudnn.deterministic,torch.are_deterministic_algorithms_enabled(),torch.backends.cuda.matmul.allow_tf32)
  torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False);torch.backends.cuda.matmul.allow_tf32=False
  try:
   with torch.inference_mode():exec(self.feature_code,ns)
  finally:
   torch.backends.cudnn.allow_tf32=old[0];torch.backends.cudnn.deterministic=old[1];torch.use_deterministic_algorithms(old[2]);torch.backends.cuda.matmul.allow_tf32=old[3]
  return ns['ft']
 def inputs(self,rgb,t,words,actions):
  h,idx=self.history(rgb,t);previous=np.zeros((16,7),np.float32)
  for j,k in enumerate(idx):
   assert int(k)<=t
   if k>0:previous[j]=actions[int(k)-1]
  h=np.concatenate([h,previous],1)
  ee,rotation,fingers=self.finger(rgb,t)
  cams=['agentview','robot0_eye_in_hand'];images=[rgb[c+'/rgb'][t] for c in cams]
  ns={'torch':torch,'np':np,'rgb':images,'depth':[rgb[c+'/depth_m'][t] for c in cams],'robot':[rgb[c+'/robot_mask'][t] for c in cams],'K':[rgb[c+'/K'][t] for c in cams],'T':[rgb[c+'/camera_to_world'][t] for c in cams],'features':[self.feature(im) for im in images],'ee':ee,'meta':{'rotation':rotation,'fingers':fingers},'prev':{'h':torch.from_numpy(h),'words':words}}
  exec(self.input_code,ns);return ns['inputs']
def load_model():
 runtime.setup();runtime.seed(20261029)
 model=runtime.build('cuda');ck=torch.load(RUN/'checkpoints/step3000.pt',map_location='cpu',weights_only=False)
 assert ck['step']==3000
 model.load_state_dict(ck['model'],strict=True);model.eval();del ck
 return model
def predict(model,d,generator,plan=None):
 x=runtime.move(d,'cuda');noise=torch.randn((1,16,6),generator=generator).cuda()
 with torch.no_grad():
  encoded=model.encode(x,step=3000);motion,grip,sw=model.generate(encoded,noise,plan)
 cls=grip.argmax(-1)
 norm=json.loads((RUN/'normalization.json').read_text())
 raw=motion[0,0].cpu().numpy()*np.asarray(norm['std'])+np.asarray(norm['mean'])
 action=np.r_[raw,float(cls[0,0])*2-1].astype(np.float32)
 sequence=torch.cat([motion,cls.float().unsqueeze(-1)*2-1],-1)
 nextplan=torch.cat([sequence[:,1:],sequence[:,-1:]],1).detach()
 assert np.isfinite(action).all()
 return action,nextplan,encoded
